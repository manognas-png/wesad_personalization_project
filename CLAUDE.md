# WESAD Personalized Stress Detection

## Goal
Show how much a stress-detection model improves when it gets a few minutes of
a new person's own data. Key result: macro F1 vs. minutes of personal
calibration data per condition (0, 1, 2, 3), evaluated with
leave-one-subject-out (LOSO). Binary (stress vs. not) is the primary task;
3-class (baseline / stress / amusement) is secondary.

## Setup
- Python 3.12, venv at .venv (activate: source .venv/bin/activate)
- Run modules from repo root: python -m src.<module>
- All settings live in src/config.py. Never hardcode paths, rates, or labels.
- Local: preprocessing, features, RF/logistic baselines.
  Colab (T4 GPU): deep learning only. Processed .npz go to Google Drive.

## Data
- Raw: data/WESAD/S#/S#.pkl, 15 subjects (S2–S17, no S1 or S12).
  Load with pickle.load(f, encoding="latin1").
- Wrist only (Empatica E4): BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz (N, 3).
- Labels at 700 Hz: 1 baseline, 2 stress, 3 amusement. Drop 0, 4, 5, 6, 7.
- Processed: data/processed/S#.npz from src/load.py
  X: (n_windows, 1920, 4) at 32 Hz, channels [BVP, EDA, TEMP, ACC]
  y: raw labels (1/2/3), start_sec: window start time in the session.
  60 s windows, 5 s stride, pure-label windows only, NOT normalized.
- Usable data per subject: ~19.5 min baseline, ~10-12 min stress,
  ~6 min amusement (amusement is the bottleneck).
- NEVER commit data, .pkl, .npz, model weights, or tokens.

## Personalization design
- condition_time_sec = seconds since the start of that condition block
  for that subject (each condition is one contiguous block).
- Calibration amount k = minutes PER CONDITION, k in [0, 1, 2, 3].
  Calibration windows: condition_time_sec + 60 <= k*60
  (window must END before the cut-off).
- Test set is FIXED for every k: condition_time_sec >= TEST_START_MIN*60 (180 s).
  Same test windows at every k so the curve is comparable.
- Calibration and test windows never share samples (no overlap leakage).
- Methods to compare:
  1. Generic: train on other 14 subjects only.
  2. Baseline normalization: subtract the person's mean feature values from
     their own early baseline windows only (label-free calibration). No
     z-score: at k=1 there is only one baseline window, so no SD. Applied to
     every subject (training and test) the same way.
  3. Fine-tune last layer (deep models) / add calibration data with
     higher weight (RF).
  4. Fine-tune full model (deep models).
  5. Person-only: train only on the person's calibration windows.
- Report per-subject results and mean ± SD across subjects.

## Rules
- Align labels by TIME (index = t × fs), never by sample position.
- Do NOT normalize during preprocessing. Normalization is one of the
  personalization methods being compared.
- Fit imputers/scalers/models on training data only. Never on test data.
- For HRV, upsample BVP windows to 128 Hz before peak detection.
- Random seed 42 everywhere.
- Keep code simple, readable, and commented. I am learning, so explain
  non-obvious lines with short comments.
- Don't install packages or change config.py without asking.
- Don't commit. I review and commit myself.