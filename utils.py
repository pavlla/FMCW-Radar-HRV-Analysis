import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend, lfilter, welch
from scipy.ndimage import gaussian_filter1d, label
import config

try:
    from vmdpy import VMD
    _HAS_VMD = True
except ImportError:
    _HAS_VMD = False


def bandpass(x, fs, f1, f2, order=4):
    sos = butter(order, [f1/(fs/2), f2/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def vmd_extract_respiratory(sig):
    """VMD decomposition – select mode whose dominant frequency falls in RR_BAND.
    Falls back to bandpass if vmdpy unavailable, VMD fails, or no mode matches."""
    if not _HAS_VMD:
        return bandpass(sig, config.FS, *config.RR_BAND)

    try:
        u, _, _ = VMD(sig, config.VMD_ALPHA, config.VMD_TAU, config.VMD_K,
                       0, 1, 1e-7)
    except Exception:
        return bandpass(sig, config.FS, *config.RR_BAND)

    best_mode = None
    best_power = -1.0

    for k in range(u.shape[0]):
        freqs, psd = welch(u[k], fs=config.FS, nperseg=min(len(u[k]), 512))
        mask = (freqs >= config.RR_BAND[0]) & (freqs <= config.RR_BAND[1])
        if not np.any(mask):
            continue
        in_band_power = np.max(psd[mask])
        if in_band_power > best_power:
            best_power = in_band_power
            best_mode = k

    if best_mode is None:
        return bandpass(sig, config.FS, *config.RR_BAND)

    return u[best_mode]


def open_memmap(path):
    return np.memmap(path, dtype=np.int16, mode="r")


def read_frame(mem, frame_idx):
    samples_per_frame = config.N_CHIRPS * config.N_RX * config.N_ADC * 2
    start = frame_idx * samples_per_frame
    raw = mem[start:start + samples_per_frame]
    raw = raw.reshape(-1, 4)

    I0, I1 = raw[:, 0], raw[:, 1]
    Q0, Q1 = raw[:, 2], raw[:, 3]

    c0 = I0 + 1j * Q0
    c1 = I1 + 1j * Q1

    stream = np.empty((c0.size * 2,), dtype=np.complex64)
    stream[0::2] = c0
    stream[1::2] = c1

    return stream.reshape(config.N_CHIRPS, config.N_RX, config.N_ADC)


def build_range_map(mem):
    half = config.N_ADC // 2
    range_map = np.zeros((config.OBS_FRAMES, half))

    for f in range(config.OBS_FRAMES):
        frame = read_frame(mem, f)
        fft_adc = np.fft.fft(frame, axis=-1)
        mag = np.mean(np.abs(fft_adc), axis=(0, 1))
        range_map[f] = mag[:half]

    return range_map



def find_motion_based_roi(range_map):
    # usunięcie cluttera (DC w czasie)
    rm_dc = range_map - range_map.mean(axis=0, keepdims=True)

    # energia ruchu
    motion_profile = np.var(rm_dc, axis=0)

    # wygładzenie
    motion_profile = gaussian_filter1d(motion_profile, sigma=2)

    # normalizacja
    motion_profile /= np.max(motion_profile)

    # adaptacyjny próg
    thr = np.mean(motion_profile) + 0.5 * np.std(motion_profile)
    mask = motion_profile > thr

    labels, nlab = label(mask)
    if nlab == 0:
        return None

    # wybór NAJSZERSZEGO obszaru
    best_label = None
    best_width = 0

    for lab in range(1, nlab + 1):
        idx = np.where(labels == lab)[0]
        if len(idx) > best_width:
            best_width = len(idx)
            best_label = lab

    roi_bins = np.where(labels == best_label)[0]
    return roi_bins.min(), roi_bins.max()


def pick_target_bin_motion(mem):
    range_map = build_range_map(mem)
    roi = find_motion_based_roi(range_map)

    if roi is None:
        return (config.N_ADC // 4)

    roi_start, roi_end = roi
    target_bin = int((roi_start + roi_end) / 2)

    return target_bin


def extract_displacement(mem, target_bin, mode="RR"):
    bins = [b for b in [target_bin - 1, target_bin, target_bin + 1]
            if 0 <= b < config.N_ADC // 2]

    # Per-Rx complex signals: shape (N_RX, OBS_FRAMES)
    s_rx = np.zeros((config.N_RX, config.OBS_FRAMES), dtype=np.complex64)

    for f in range(config.OBS_FRAMES):
        frame = read_frame(mem, f)
        fft_adc = np.fft.fft(frame, axis=-1)
        # Average over chirps and selected bins per Rx channel
        for rx in range(config.N_RX):
            s_rx[rx, f] = np.mean(fft_adc[:, rx, bins])

    # MRC: Maximal Ratio Combining across Rx channels
    h = np.mean(s_rx, axis=1)  # channel response (N_RX,)
    power = np.sum(np.abs(h) ** 2)
    if power > 0:
        w = np.conj(h) / power  # MRC weights (N_RX,)
    else:
        w = np.ones(config.N_RX, dtype=np.complex64) / config.N_RX
    s = w @ s_rx  # combined signal (OBS_FRAMES,)

    # DC removal
    s -= np.mean(s)

    # DACM: Differentiate And Cross-Multiply phase extraction
    I = np.real(s)
    Q = np.imag(s)
    eps = 1e-12
    dphi = (I[:-1] * Q[1:] - Q[:-1] * I[1:]) / (I[:-1]**2 + Q[:-1]**2 + eps)
    phase = np.concatenate(([0.0], np.cumsum(dphi)))
    phase = detrend(phase)

    return (config.LAMBDA / (4 * np.pi)) * phase


def estimate_bpm_series(sig, band, window, nfft=None):
    bpm = []

    for start in range(0, len(sig) - window + 1, config.SW):
        seg = sig[start:start + window]

        # AR(1) pre-whitening for RR path (nfft is set)
        if nfft is not None:
            seg_z = seg - np.mean(seg)
            r0 = np.dot(seg_z, seg_z)
            if r0 > 0:
                r1 = np.dot(seg_z[:-1], seg_z[1:])
                a1 = np.clip(r1 / r0, -0.99, 0.99)
            else:
                a1 = 0.0
            seg_w = lfilter([1, -a1], [1], seg_z)
            seg_nperseg = len(seg_w)
            n = nfft
        else:
            seg_w = seg
            seg_nperseg = len(seg) // 2
            n = len(seg)

        freqs, spec = welch(seg_w, fs=config.FS, nperseg=seg_nperseg,
                            nfft=n, window="hann")
        mask = (freqs >= band[0]) & (freqs <= band[1])
        freqs_band = freqs[mask]
        spec_band = spec[mask]

        if len(spec_band) < 3:
            bpm.append(np.nan)
            continue

        peak_idx = np.argmax(spec_band)

        # SNR gating for RR path
        if nfft is not None:
            median_level = np.median(spec_band)
            if median_level > 0:
                snr = spec_band[peak_idx] / median_level
            else:
                snr = 0.0
            if snr < config.RR_SNR_THR:
                bpm.append(np.nan)
                continue

        f_peak = freqs_band[peak_idx]

        # Harmonic check: prefer subharmonic f/2 or f/3 if plausible (RR path only)
        if nfft is not None:
            df = freqs_band[1] - freqs_band[0] if len(freqs_band) > 1 else 1.0
            peak_amp = spec_band[peak_idx]
            for divisor in [2, 3]:
                f_sub = f_peak / divisor
                if f_sub < band[0] or f_sub > band[1]:
                    continue
                sub_idx = int(round((f_sub - freqs_band[0]) / df))
                if sub_idx < 1 or sub_idx >= len(spec_band) - 1:
                    continue
                # Check local maximum
                if (spec_band[sub_idx] >= spec_band[sub_idx - 1] and
                        spec_band[sub_idx] >= spec_band[sub_idx + 1]):
                    # Check amplitude threshold (>= 25% of peak)
                    if spec_band[sub_idx] >= 0.25 * peak_amp:
                        f_peak = freqs_band[sub_idx]
                        peak_idx = sub_idx
                        break

        # Parabolic interpolation on log-spectrum (Jacobsen estimator)
        if 0 < peak_idx < len(spec_band) - 1:
            alpha = np.log(spec_band[peak_idx - 1] + 1e-30)
            beta  = np.log(spec_band[peak_idx]     + 1e-30)
            gamma = np.log(spec_band[peak_idx + 1] + 1e-30)
            delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            df = freqs_band[1] - freqs_band[0]
            f_peak = freqs_band[peak_idx] + delta * df
        else:
            f_peak = freqs_band[peak_idx]

        bpm.append(60.0 * f_peak)

    return np.array(bpm)



def load_polar_hr(csv_path):
    df = pd.read_csv(csv_path, skiprows=2)
    hr = []

    for _, row in df.iterrows():
        try:
            hr.append(float(row.iloc[2]))  
        except:
            pass

    return np.array(hr)


def load_br_ref_full(ods_path):

    raw = pd.read_excel(ods_path, engine="odf", header=None)

    rows = []

    current_participant = None
    current_scenario = None

    for i in range(len(raw)):

        row = raw.iloc[i]

        # Participant
        if isinstance(row[0], str) and "Participant" in row[0]:
            current_participant = int(row[0].split()[-1])
            continue

        # Scenario
        if isinstance(row[1], str) and "Scenario" in row[1]:

            if "Distance" in row[1]:
                current_scenario = "Distance"
            elif "Posture" in row[1]:
                current_scenario = "Orientation"
            elif "Angle" in row[1]:
                current_scenario = "Angle"
            continue

        if isinstance(row[1], str) and "Elevated" in row[1]:
            current_scenario = "Elevated"
            continue

        if isinstance(row[2], str):

            condition = row[2]

            for rec in range(4):

                val = row[3 + rec]

                if isinstance(val, (int, float)) and not pd.isna(val):

                    rows.append({
                        "participant": current_participant,
                        "scenario": current_scenario,
                        "condition": condition,
                        "recording": rec + 1,
                        "br_ref": float(val)
                    })

    return pd.DataFrame(rows)


def run_distance_scenario():

    all_rows = []

    for participant in range(1, 11):

        for scenario_label, scenario_folder in config.SCENARIOS.items():

            base_radar = f"/Volumes/X10 Pro/Human/Radar data/Participant {participant}/{scenario_folder}"
            base_hr_gt = f"HR_Ref_Values/Participant {participant}/{scenario_folder}"

            if not os.path.exists(base_radar):
                continue

            conditions = os.listdir(base_radar)

            for condition in conditions:
                for rec in config.RECORDINGS:

                    radar_path = os.path.join(base_radar, condition, rec, "data_Raw_0.bin")
                    if not os.path.exists(radar_path):
                        continue

                    mem = open_memmap(radar_path)

                    target_bin = pick_target_bin_motion(mem)

                    disp_rr = extract_displacement(mem, target_bin, mode="RR")
                    disp_hr = extract_displacement(mem, target_bin, mode="HR")

                    rr_sig = vmd_extract_respiratory(disp_rr)
                    hr_sig = bandpass(disp_hr, config.FS, *config.HR_BAND)

                    rr = estimate_bpm_series(rr_sig, config.RR_BAND, config.W_RR,
                                             nfft=config.RR_NFFT)
                    hr = estimate_bpm_series(hr_sig, config.HR_BAND, config.W_HR)

                    hr_csv = os.path.join(base_hr_gt, condition, f"R{rec}.CSV")
                    if not os.path.exists(hr_csv):
                        continue

                    hr_ref = load_polar_hr(hr_csv)

                    all_rows.append({
                        "participant": participant,
                        "scenario": scenario_label,
                        "condition": condition,
                        "recording": int(rec),
                        "target_bin": target_bin,
                        "rr_radar": np.nanmean(rr),
                        "hr_radar": np.nanmean(hr),
                        "hr_ref": np.nanmean(hr_ref)
                    })


    df = pd.DataFrame(all_rows)

    br_ref_full = load_br_ref_full(config.BR_REF_ODS)

    df["condition"] = df["condition"].astype(str).str.strip()
    br_ref_full["condition"] = br_ref_full["condition"].astype(str).str.strip()

    df["scenario"] = df["scenario"].str.strip()
    br_ref_full["scenario"] = br_ref_full["scenario"].str.strip()

    df = df.merge(
        br_ref_full,
        on=["participant","scenario","condition","recording"],
        how="left"
    )

    df["HR_abs_err"] = (df["hr_radar"] - df["hr_ref"]).abs()
    df["RR_abs_err"] = (df["rr_radar"] - df["br_ref"]).abs()

    df["HR_pct_err"] = (df["HR_abs_err"] / df["hr_ref"]) * 100
    df["RR_pct_err"] = (df["RR_abs_err"] / df["br_ref"]) * 100
    return df

