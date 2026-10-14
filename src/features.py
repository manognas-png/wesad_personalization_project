"""Turn each 60 s window into a row of hand-crafted features -> results/features.csv.

Run from the repo root:  python -m src.features   (after python -m src.load)

Nothing is normalized here: scaling happens inside each model's pipeline.
"""
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

from src import config
from src.load import CHANNELS

FEATURES_CSV = config.RESULTS_DIR / "features.csv"

# Columns that describe the window (everything after these is a feature)
META_COLUMNS = ["subject", "start_sec", "condition_time_sec", "label"]

# Heartbeat timing at 32 Hz is only accurate to ~31 ms (1/32 s), which is as
# big as the beat-to-beat changes HRV measures. 128 Hz gives ~8 ms.
BVP_PEAK_FS = 128

# ---------- HRV cleaning ----------
# IBI = inter-beat interval, the time between two heartbeats in ms
IBI_RANGE_MS = (300, 1500)      # 1500 ms = 40 bpm, 300 ms = 200 bpm
IBI_MAX_MEDIAN_DEV = 0.20       # reject IBIs more than 20% away from the window median
MIN_IBI_VALID_FRAC = 0.60       # below this the window is too noisy for HRV
MIN_VALID_BEATS = 30            # below this there are too few beats for HRV or HR
NN50_MS = 50                    # pNN50 counts beat-to-beat changes bigger than 50 ms

# ---------- SCR cleaning ----------
SCR_MIN_AMPLITUDE_US = 0.01     # smallest sweat response we count, in microsiemens


def add_condition_time(start_sec, y, subject):
    """Seconds since the start of each window's own condition block."""
    condition_time_sec = np.zeros(len(start_sec))
    for label in config.KEEP_LABELS:
        mask = y == label
        starts = start_sec[mask]
        # One unbroken block means every window starts STRIDE_SEC after the last
        if not np.all(np.diff(starts) == config.STRIDE_SEC):
            print(f"WARNING: {subject} {config.LABEL_NAMES[label]} is not one "
                  f"contiguous block, condition_time_sec may be misleading")
        condition_time_sec[mask] = starts - starts.min()
    return condition_time_sec


def slope(signal):
    """Slope of the best straight line through the signal, in units per second."""
    times_sec = np.arange(len(signal)) / config.TARGET_FS
    # polyfit with degree 1 returns [slope, intercept]
    return np.polyfit(times_sec, signal, 1)[0]


def valid_ibi_mask(ibi_ms):
    """True for each IBI that looks like a real heartbeat interval."""
    low, high = IBI_RANGE_MS
    in_range = (ibi_ms >= low) & (ibi_ms <= high)
    if not in_range.any():
        return in_range
    # Median of the plausible IBIs only, so wild values cannot shift it.
    # A missed beat gives an IBI ~2x the median, an extra beat ~0.5x:
    # both are far outside 20%.
    median_ibi = np.median(ibi_ms[in_range])
    near_median = np.abs(ibi_ms - median_ibi) <= IBI_MAX_MEDIAN_DEV * median_ibi
    return in_range & near_median


def bvp_features(bvp):
    """Heart rate, heart-rate variability and signal quality from one BVP window."""
    features = {"bvp_hr_mean": np.nan, "bvp_sdnn": np.nan, "bvp_rmssd": np.nan,
                "bvp_pnn50": np.nan, "bvp_ibi_valid_frac": 0.0}
    try:
        # Upsample 32 -> 128 Hz BEFORE finding peaks, so each beat is timed finely
        bvp_fine = resample_poly(bvp, BVP_PEAK_FS // config.TARGET_FS, 1)
        # No neurokit artifact correction: it moves/invents beats. We reject
        # bad intervals ourselves below instead, so only measured beats count.
        _, info = nk.ppg_peaks(bvp_fine, sampling_rate=BVP_PEAK_FS, correct_artifacts=False)
    except Exception:
        # No clear beats in this window
        return features

    # Peak positions (samples) -> time between neighbouring peaks (ms)
    ibi_ms = np.diff(info["PPG_Peaks"]) / BVP_PEAK_FS * 1000
    if len(ibi_ms) == 0:
        return features

    valid = valid_ibi_mask(ibi_ms)
    valid_ibi = ibi_ms[valid]
    features["bvp_ibi_valid_frac"] = valid.mean()  # mean of True/False = fraction True

    if len(valid_ibi) < MIN_VALID_BEATS:
        return features
    features["bvp_hr_mean"] = 60000 / valid_ibi.mean()  # ms per beat -> beats/min

    if valid.mean() < MIN_IBI_VALID_FRAC:
        return features  # heart rate is still usable, variability is not

    # ddof=1 = sample standard deviation (same as neurokit's SDNN)
    features["bvp_sdnn"] = valid_ibi.std(ddof=1)

    # Beat-to-beat change, but ONLY between neighbours that are both valid.
    # Otherwise removing a bad IBI would glue two far-apart beats together
    # and create a fake jump.
    successive_diff = np.diff(ibi_ms)
    both_valid = valid[:-1] & valid[1:]
    pair_diff = successive_diff[both_valid]
    if len(pair_diff) > 0:
        features["bvp_rmssd"] = np.sqrt(np.mean(pair_diff ** 2))
        features["bvp_pnn50"] = 100 * np.mean(np.abs(pair_diff) > NN50_MS)
    return features


def scr_count(eda):
    """Number of sweat responses (SCRs) of at least SCR_MIN_AMPLITUDE_US."""
    eda_clean = nk.eda_clean(eda, sampling_rate=config.TARGET_FS)
    # Split EDA into slow level (tonic) + fast responses (phasic)
    phasic = nk.eda_phasic(eda_clean, sampling_rate=config.TARGET_FS)["EDA_Phasic"]
    # amplitude_min=0 turns off neurokit's RELATIVE threshold (10% of the
    # biggest response in the window), which made the count depend on the
    # window's largest peak. We apply a fixed threshold in µS instead.
    try:
        _, info = nk.eda_peaks(phasic, sampling_rate=config.TARGET_FS, amplitude_min=0)
    except ValueError:
        # neurokit raises ValueError when it finds no peaks at all.
        # That is a real answer (zero responses), not missing data.
        return 0
    amplitudes = np.asarray(info["SCR_Amplitude"], dtype=float)
    # NaN amplitudes (peak with no clear onset) compare False, so they are dropped
    return int(np.sum(amplitudes >= SCR_MIN_AMPLITUDE_US))


def eda_features(eda):
    """Skin conductance level, trend, and number of sweat responses (SCRs)."""
    try:
        eda_scr_count = scr_count(eda)
    except Exception:
        eda_scr_count = np.nan
    return {
        "eda_mean": eda.mean(),
        "eda_std": eda.std(),
        "eda_slope": slope(eda),
        "eda_scr_count": eda_scr_count,
    }


def temp_features(temp):
    return {"temp_mean": temp.mean(), "temp_std": temp.std(), "temp_slope": slope(temp)}


def acc_features(acc):
    return {"acc_mean": acc.mean(), "acc_std": acc.std()}


def window_features(window):
    """All features for one window of shape (window_samples, 4)."""
    # float64 because neurokit and polyfit are more reliable than with float32
    window = window.astype(np.float64)
    features = {}
    features.update(bvp_features(window[:, CHANNELS.index("BVP")]))
    features.update(eda_features(window[:, CHANNELS.index("EDA")]))
    features.update(temp_features(window[:, CHANNELS.index("TEMP")]))
    features.update(acc_features(window[:, CHANNELS.index("ACC")]))
    return features


def subject_features(subject):
    """Feature table (one row per window) for one subject."""
    data = np.load(config.PROCESSED_DIR / f"{subject}.npz")
    X, y, start_sec = data["X"], data["y"], data["start_sec"]

    rows = [window_features(window) for window in tqdm(X, desc=subject, leave=False)]
    table = pd.DataFrame(rows)

    # Put the describing columns first, in META_COLUMNS order
    table.insert(0, "subject", subject)
    table.insert(1, "start_sec", start_sec)
    table.insert(2, "condition_time_sec", add_condition_time(start_sec, y, subject))
    table.insert(3, "label", y)
    return table


def main():
    tables = [subject_features(subject) for subject in tqdm(config.SUBJECTS, desc="Subjects")]
    features = pd.concat(tables, ignore_index=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)

    print(f"\nSaved {FEATURES_CSV}")
    print(f"Shape: {features.shape} (rows = windows, columns = "
          f"{len(META_COLUMNS)} info + {features.shape[1] - len(META_COLUMNS)} features)")
    print("\n% NaN per feature:")
    nan_percent = features.drop(columns=META_COLUMNS).isna().mean() * 100
    print(nan_percent.round(2).to_string())


if __name__ == "__main__":
    main()
