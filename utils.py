import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from scipy.ndimage import gaussian_filter1d, label
import config

def bandpass(x, fs, f1, f2, order=4):
    b, a = butter(order, [f1/(fs/2), f2/(fs/2)], btype="band")
    return filtfilt(b, a, x)


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
        # awaryjnie – środek zakresu
        return (config.N_ADC // 4)

    roi_start, roi_end = roi
    target_bin = int((roi_start + roi_end) / 2)

    return target_bin


def extract_displacement(mem, target_bin, mode="RR"):
    s = np.zeros(config.OBS_FRAMES, dtype=np.complex64)

    if mode == "RR":
        bins = [target_bin]
    else:
        bins = [b for b in [target_bin-1, target_bin, target_bin+1]
                if 0 <= b < config.N_ADC//2]

    for f in range(config.OBS_FRAMES):
        frame = read_frame(mem, f)
        fft_adc = np.fft.fft(frame, axis=-1)
        s[f] = np.mean(fft_adc[:, :, bins])

    s -= np.mean(s)
    phase = np.unwrap(np.angle(s))
    phase = detrend(phase)

    return (config.LAMBDA / (4 * np.pi)) * phase


def estimate_bpm_series(sig, band, window):
    bpm = []

    for start in range(0, len(sig) - window + 1, config.SW):
        seg = sig[start:start + window]
        freqs = np.fft.rfftfreq(len(seg), 1/config.FS)
        spec = np.abs(np.fft.rfft(seg))**2

        mask = (freqs >= band[0]) & (freqs <= band[1])
        f_peak = freqs[mask][np.argmax(spec[mask])]
        bpm.append(60 * f_peak)

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


def load_br_ref(ods_path):
    raw = pd.read_excel(ods_path, engine="odf", header=None)
    rows = []

    for i, dist in enumerate(config.DISTANCES):
        vals = raw.iloc[2+i, 3:7].values
        for r in range(4):
            rows.append({
                "distance": dist,
                "recording": r+1,
                "br_ref": float(vals[r])
            })
    return pd.DataFrame(rows)


def run_distance_scenario():
    br_ref = load_br_ref(config.BR_REF_ODS)
    rows = []

    for dist in config.DISTANCES:
        for rec in config.RECORDINGS:
            radar_path = os.path.join(config.BASE_RADAR, dist, rec, "data_Raw_0.bin")
            if not os.path.exists(radar_path):
                continue

            mem = open_memmap(radar_path)

            target_bin = pick_target_bin_motion(mem)

            disp_rr = extract_displacement(mem, target_bin, mode="RR")
            disp_hr = extract_displacement(mem, target_bin, mode="HR")

            rr_sig = bandpass(disp_rr, config.FS, *config.RR_BAND)
            hr_sig = bandpass(disp_hr, config.FS, *config.HR_BAND)

            rr = estimate_bpm_series(rr_sig, config.RR_BAND, config.W_RR)
            hr = estimate_bpm_series(hr_sig, config.HR_BAND, config.W_HR)

            hr_csv = os.path.join(config.BASE_HR_GT, dist, f"R{rec}.CSV")
            hr_ref = load_polar_hr(hr_csv)

            rows.append({
                "distance": dist,
                "recording": int(rec),
                "target_bin": target_bin,
                "rr_radar": round(np.nanmean(rr), 1),
                "hr_radar": np.nanmean(hr),
                "hr_ref": np.nanmean(hr_ref)
            })

    df = pd.DataFrame(rows)
    df = df.merge(br_ref, on=["distance", "recording"])

    df["HR_abs_err"] = (df["hr_radar"] - df["hr_ref"]).abs()
    df["RR_abs_err"] = (df["rr_radar"] - df["br_ref"]).abs()

    return df