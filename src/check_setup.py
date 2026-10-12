"""Sanity checks: packages installed, data present, inspect one subject."""
import importlib
import pickle

import numpy as np

from src import config


def check_packages():
    print("=== Packages ===")
    packages = ["numpy", "pandas", "scipy", "sklearn", "matplotlib",
                "neurokit2", "torch", "tqdm"]
    for name in packages:
        try:
            importlib.import_module(name)
            print(f"OK       {name}")
        except ImportError:
            print(f"MISSING  {name}")


def check_data():
    print("\n=== Data ===")
    missing = []
    for subject in config.SUBJECTS:
        path = config.RAW_DIR / subject / f"{subject}.pkl"
        if not path.exists():
            missing.append(subject)
    print(f"Found {len(config.SUBJECTS) - len(missing)}/{len(config.SUBJECTS)} subjects")
    if missing:
        print("Missing:", missing)


def inspect_one_subject(subject="S2"):
    print(f"\n=== Inspecting {subject} ===")
    path = config.RAW_DIR / subject / f"{subject}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")   # WESAD was saved in Python 2

    print("Keys:", list(data.keys()))

    print("\nWrist signals:")
    for name, signal in data["signal"]["wrist"].items():
        fs = config.WRIST_FS[name]
        minutes = signal.shape[0] / fs / 60
        print(f"  {name:5s} shape={signal.shape}  fs={fs} Hz  ≈ {minutes:.1f} min")

    print("\nChest signals:")
    for name, signal in data["signal"]["chest"].items():
        print(f"  {name:5s} shape={signal.shape}")

    labels = data["label"]
    print(f"\nLabels: shape={labels.shape}  ≈ {labels.shape[0] / config.LABEL_FS / 60:.1f} min")
    values, counts = np.unique(labels, return_counts=True)
    for v, c in zip(values, counts):
        name = config.LABEL_NAMES.get(v, "ignored")
        print(f"  label {v} ({name}): {c / config.LABEL_FS / 60:.1f} min")


if __name__ == "__main__":
    check_packages()
    check_data()
    inspect_one_subject("S2")