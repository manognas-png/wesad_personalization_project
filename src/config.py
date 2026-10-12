"""Project settings: paths, subjects, sampling rates, labels, windowing."""
from pathlib import Path

# ---------- Paths ----------
# config.py is at repo/src/config.py -> parents[0] = src/, parents[1] = repo/
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "WESAD"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Processed data: Google Drive when running in Colab, local folder otherwise
try:
    import google.colab  # only exists inside Colab
    PROCESSED_DIR = Path("/content/drive/MyDrive/wesad_project/processed")
except ImportError:
    PROCESSED_DIR = ROOT / "data" / "processed"

# ---------- Subjects ----------
# S1 and S12 were dropped by the dataset authors (sensor malfunction)
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

# ---------- Sampling rates (Hz) ----------
LABEL_FS = 700                      # labels are stored at the chest rate
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

# ---------- Labels ----------
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
KEEP_LABELS = [1, 2, 3]             # drop meditation (4) and transitions (0, 5-7)
BINARY_MAP = {1: 0, 3: 0, 2: 1}     # stress vs. not stress
THREECLASS_MAP = {1: 0, 2: 1, 3: 2} # baseline / stress / amusement

# ---------- Windowing ----------
TARGET_FS = 32                      # resample all wrist signals to 32 Hz
WINDOW_SEC = 60                     # each window = 60 s of signal
STRIDE_SEC = 5                      # new window every 5 s (windows overlap)

# ---------- Personalization ----------
CALIBRATION_MINUTES = [0, 1, 2, 5, 10]  # personal data given to the model
CALIBRATION_GAP_SEC = WINDOW_SEC        # gap between calibration and test windows
                                        # so overlapping windows don't leak
RANDOM_SEED = 42