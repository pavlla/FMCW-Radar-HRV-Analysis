import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

FS = 20.0
NUM_CHIRPS = 128
NUM_RX = 4
NUM_ADC = 1657

def load_radar_bin(
    bin_path,
    num_chirps=NUM_CHIRPS,
    num_rx=NUM_RX,
    num_adc=NUM_ADC
):
    raw = np.fromfile(bin_path, dtype=np.int16)

    iq = raw[0::2] + 1j * raw[1::2]

    samples_per_frame = num_chirps * num_rx * num_adc
    num_frames = iq.shape[0] // samples_per_frame

    # obcinamy niepełną ramkę
    iq = iq[:num_frames * samples_per_frame]

    iq = iq.reshape(
        num_frames,
        num_chirps,
        num_rx,
        num_adc
    )

    return iq


def reshape_iq(iq, num_chirps=NUM_CHIRPS, num_rx=NUM_RX, num_adc=NUM_ADC):
    total_samples = iq.size
    samples_per_frame = num_chirps * num_rx * num_adc

    if total_samples % samples_per_frame != 0:
        raise ValueError(
            f"Złe parametry: total={total_samples}, "
            f"frame={samples_per_frame}"
        )

    num_frames = total_samples // samples_per_frame

    return iq.reshape(num_frames, num_chirps, num_rx, num_adc)


def compute_range_profile(iq, rx_id=0):
    window = np.hanning(iq.shape[-1])
    sig = iq[:, :, rx_id, :] * window
    fft = np.fft.fft(sig, axis=-1)
    return np.abs(fft)


def select_chest_bin(range_mag):
    mean_profile = range_mag.mean(axis=(0, 1))
    chest_bin = np.argmax(mean_profile)
    return chest_bin


def extract_phase(iq, chest_bin, rx_id=0):
    sig = iq[:, :, rx_id, chest_bin]
    phase = np.angle(sig)
    phase = np.unwrap(phase, axis=1)
    phase = phase.mean(axis=1)
    return phase


def phase_to_displacement(phase, wavelength=0.004):
    displacement = phase * wavelength / (4 * np.pi)
    return displacement


def bandpass(signal, fs, low, high):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, signal)


def detect_heart_beats(displacement, fs):
    heart_signal = bandpass(displacement, fs, 0.8, 2.5)

    peaks, _ = find_peaks(
        heart_signal,
        distance=int(fs * 0.4)
    )

    peak_times = peaks / fs
    return peak_times



def compute_hrv(peak_times):
    if len(peak_times) < 4:
        return None

    ibi = np.diff(peak_times)

    return {
        "num_beats": len(ibi),
        "mean_ibi": np.mean(ibi),
        "sdnn": np.std(ibi, ddof=1),
        "rmssd": np.sqrt(np.mean(np.diff(ibi)**2))
    }


def process_one_measurement(bin_path, fs=20.0):
    iq = load_radar_bin(bin_path)
    iq = reshape_iq(iq)

    range_mag = compute_range_profile(iq)
    chest_bin = select_chest_bin(range_mag)

    phase = extract_phase(iq, chest_bin)
    displacement = phase_to_displacement(phase)

    peak_times = detect_heart_beats(displacement, fs)
    return compute_hrv(peak_times)


