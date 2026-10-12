"""Load raw WESAD wrist data, clean it, cut it into labelled windows, save .npz.

Run from the repo root:  python -m src.load

Nothing is normalized here on purpose: normalization is one of the
personalization methods we compare later.
"""
import pickle
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from tqdm import tqdm

from src import config

# Column order of the processed signal array (last axis of X)
CHANNELS = ["BVP", "EDA", "TEMP", "ACC"]

ACC_COUNTS_PER_G = 64   # Empatica E4 stores ACC in units of 1/64 g
FILTER_ORDER = 4
BVP_BAND_HZ = (0.5, 8)  # keeps heart-rate range, removes drift and noise
EDA_CUTOFF_HZ = 1       # EDA changes slowly; anything faster is noise
MIN_WINDOWS_PER_LABEL = 30


def load_subject(subject):
    """Return (wrist, labels) for one subject, e.g. load_subject("S2")."""
    path = config.RAW_DIR / subject / f"{subject}.pkl"
    with open(path, "rb") as f:
        # The files were pickled with Python 2, so Python 3 needs latin1
        data = pickle.load(f, encoding="latin1")

    raw_wrist = data["signal"]["wrist"]
    wrist = {
        "ACC": raw_wrist["ACC"],            # stays (N, 3): x, y, z
        "BVP": raw_wrist["BVP"].ravel(),    # (N, 1) -> (N,)
        "EDA": raw_wrist["EDA"].ravel(),
        "TEMP": raw_wrist["TEMP"].ravel(),
    }
    labels = data["label"]                  # one label per 1/700 s
    return wrist, labels


def label_at_times(labels, times_sec):
    """Return the label that was active at each time (in seconds)."""
    # time -> position in the 700 Hz label array
    idx = (np.asarray(times_sec) * config.LABEL_FS).astype(int)
    # the wrist recording can run slightly past the labels, so stay in range
    idx = np.clip(idx, 0, len(labels) - 1)
    return labels[idx]


def _resample(signal, fs):
    """Resample a 1-D signal from fs to config.TARGET_FS."""
    # e.g. 64 -> 32 Hz is ratio 1/2 (up=1, down=2); 4 -> 32 Hz is 8/1
    ratio = Fraction(config.TARGET_FS, fs)
    # padtype="line" stops the ends from being pulled toward zero
    # (the default pads with zeros, which bends TEMP ~33 °C at the edges)
    return resample_poly(signal, ratio.numerator, ratio.denominator, padtype="line")


def preprocess_wrist(wrist):
    """Filter and resample the wrist signals -> array (n_samples, 4)."""
    # sosfiltfilt runs the filter forward then backward = zero phase shift
    bvp_sos = butter(FILTER_ORDER, BVP_BAND_HZ, btype="bandpass",
                     fs=config.WRIST_FS["BVP"], output="sos")
    bvp = sosfiltfilt(bvp_sos, wrist["BVP"])

    eda_sos = butter(FILTER_ORDER, EDA_CUTOFF_HZ, btype="lowpass",
                     fs=config.WRIST_FS["EDA"], output="sos")
    eda = sosfiltfilt(eda_sos, wrist["EDA"])

    temp = wrist["TEMP"]  # already smooth, no filter

    # ACC: raw counts -> g, then overall movement strength (drops direction)
    acc_g = wrist["ACC"] / ACC_COUNTS_PER_G
    acc = np.sqrt((acc_g ** 2).sum(axis=1))

    # Bring every channel to the same rate, in CHANNELS order
    resampled = [
        _resample(bvp, config.WRIST_FS["BVP"]),
        _resample(eda, config.WRIST_FS["EDA"]),
        _resample(temp, config.WRIST_FS["TEMP"]),
        _resample(acc, config.WRIST_FS["ACC"]),
    ]

    # Lengths can differ by a few samples, so cut all to the shortest
    n = min(len(channel) for channel in resampled)
    return np.column_stack([channel[:n] for channel in resampled]).astype(np.float32)


def make_windows(signals, labels):
    """Cut signals into overlapping windows -> (X, y, start_sec)."""
    window = config.WINDOW_SEC * config.TARGET_FS   # samples per window
    stride = config.STRIDE_SEC * config.TARGET_FS   # samples between starts

    # Label of every signal sample, matched by time (not by position)
    times_sec = np.arange(len(signals)) / config.TARGET_FS
    sample_labels = label_at_times(labels, times_sec)

    # idx[i] = the sample positions that belong to window i
    starts = np.arange(0, len(signals) - window + 1, stride)
    idx = starts[:, None] + np.arange(window)       # (n_windows, window)
    window_labels = sample_labels[idx]

    # Keep a window only if it has ONE label throughout, and we want that label
    first = window_labels[:, 0]
    one_label = (window_labels == first[:, None]).all(axis=1)
    keep = one_label & np.isin(first, config.KEEP_LABELS)

    X = signals[idx[keep]]                          # (n_kept, window, 4)
    y = first[keep]
    start_sec = starts[keep] / config.TARGET_FS
    return X, y, start_sec


def process_subject(subject):
    """Load, preprocess, window and save one subject. Returns (y, start_sec)."""
    wrist, labels = load_subject(subject)
    signals = preprocess_wrist(wrist)
    X, y, start_sec = make_windows(signals, labels)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.PROCESSED_DIR / f"{subject}.npz",
        X=X, y=y, start_sec=start_sec,
        subject=subject, channels=CHANNELS, fs=config.TARGET_FS,
    )
    return y, start_sec


def minutes_covered(start_sec):
    """Minutes of signal covered by these windows, counting overlaps once."""
    if len(start_sec) == 0:
        return 0.0
    # Tick off every second that falls inside at least one window
    covered = np.zeros(int(start_sec.max()) + config.WINDOW_SEC, dtype=bool)
    for start in start_sec.astype(int):
        covered[start:start + config.WINDOW_SEC] = True
    return covered.sum() / 60


def main():
    names = [config.LABEL_NAMES[label] for label in config.KEEP_LABELS]
    rows, warnings = [], []

    for subject in tqdm(config.SUBJECTS, desc="Processing"):
        y, start_sec = process_subject(subject)
        counts = [int((y == label).sum()) for label in config.KEEP_LABELS]
        minutes = [minutes_covered(start_sec[y == label]) for label in config.KEEP_LABELS]
        rows.append((subject, len(y), counts, minutes))

        for name, count in zip(names, counts):
            if count < MIN_WINDOWS_PER_LABEL:
                warnings.append(f"WARNING: {subject} has only {count} {name} windows")

    # Summary table: window counts, then minutes, per label
    header = f"{'subject':<8}{'total':>7}"
    header += "".join(f"{name + ' win':>15}" for name in names)
    header += "".join(f"{name + ' min':>15}" for name in names)
    print("\n" + header)
    print("-" * len(header))
    for subject, total, counts, minutes in rows:
        line = f"{subject:<8}{total:>7}"
        line += "".join(f"{count:>15}" for count in counts)
        line += "".join(f"{minute:>15.1f}" for minute in minutes)
        print(line)

    print(f"\nSaved to {config.PROCESSED_DIR}")
    for warning in warnings:
        print(warning)
    if not warnings:
        print(f"All subjects have at least {MIN_WINDOWS_PER_LABEL} windows per label.")


if __name__ == "__main__":
    main()
