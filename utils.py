import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.ndimage import gaussian_filter1d, label
import config
from scipy.signal import welch


def bandpass(x, fs, f1, f2, order=4):
    sos = butter(order, [f1/(fs/2), f2/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


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
    s = np.zeros(config.OBS_FRAMES, dtype=np.complex64)

    bins = [b for b in [target_bin - 1, target_bin, target_bin + 1]
            if 0 <= b < config.N_ADC // 2]

    for f in range(config.OBS_FRAMES):
        frame = read_frame(mem, f)
        fft_adc = np.fft.fft(frame, axis=-1)
        s[f] = np.mean(fft_adc[:, :, bins])

    s -= np.mean(s)
    phase = np.unwrap(np.angle(s))
    phase = detrend(phase)

    return (config.LAMBDA / (4 * np.pi)) * phase


def estimate_bpm_series(sig, band, window, nfft=None):
    bpm = []

    for start in range(0, len(sig) - window + 1, config.SW):
        seg = sig[start:start + window]

        if nfft is not None:
            seg_nperseg = len(seg)
            n = nfft
        else:
            seg_nperseg = len(seg) // 2
            n = len(seg)

        freqs, spec = welch(seg, fs=config.FS, nperseg=seg_nperseg,
                            nfft=n, window="hann")
        mask = (freqs >= band[0]) & (freqs <= band[1])
        freqs_band = freqs[mask]
        spec_band = spec[mask]

        if len(spec_band) < 3:
            bpm.append(np.nan)
            continue

        peak_idx = np.argmax(spec_band)
        peak_val = spec_band[peak_idx]

        # SNR gating (only in high-resolution mode = RR)
        if nfft is not None:
            median_val = np.median(spec_band)
            if median_val > 0 and (peak_val / median_val) < config.RR_SNR_THR:
                bpm.append(np.nan)
                continue

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

                    rr_sig = bandpass(disp_rr, config.FS, *config.RR_BAND)
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

