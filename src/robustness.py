"""Robustness checks for the RF personalization result -> results/personalization_robustness.csv.

Run from the repo root:  python -m src.robustness   (after python -m src.features)

1. Feature check: does the gain hold with EDA only (no temperature)?
2. Temporal gap check: does it hold when the test set starts 1 minute after
   calibration ends, instead of right after it?
3. Temperature drift: does skin temperature just rise with time in the session?
4. Why did baseline normalization hurt S2, S3 and S17?
Checks 1 and 2 are saved to the CSV; 3 and 4 are printed only.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

from src import config
from src.baseline import TASKS
from src.features import FEATURES_CSV
from src.personalize_rf import (BASELINE_LABEL, baseline_normalize, fit_and_score,
                                is_calibration, is_test, weighted_union)

ROBUSTNESS_CSV = config.RESULTS_DIR / "personalization_robustness.csv"
PERSONALIZATION_CSV = config.RESULTS_DIR / "personalization_rf.csv"

K = max(config.CALIBRATION_MINUTES)          # all checks use the largest k (3 min)
GAP_TEST_START_SEC = K * 60 + 60             # 240 s: a 1-minute gap after calibration
EDA_FEATURES = [column for column in config.MAIN_FEATURES if column.startswith("eda_")]
METHODS = ["generic", "generic_plus_calib", "norm_plus_calib", "person_only"]
NORM_PROBLEM_SUBJECTS = ["S2", "S3", "S17"]  # baseline_norm hurt these three most
STRESS_LABEL = next(label for label, name in config.LABEL_NAMES.items() if name == "stress")

# variant name -> (feature columns, test start in seconds)
VARIANTS = {
    # Reference: same as personalize_rf.py, so it should reproduce its numbers
    "main_test180": (config.MAIN_FEATURES, config.TEST_START_MIN * 60),
    "eda_only_test180": (EDA_FEATURES, config.TEST_START_MIN * 60),
    "main_test240": (config.MAIN_FEATURES, GAP_TEST_START_SEC),
}


# ---------- Checks 1 and 2: rerun the methods ----------

def evaluate_subject(features, normalized, feature_columns, test_start_sec, test_subject):
    """generic + 3 personalization methods at k = K for one held-out subject."""
    is_person = features["subject"] == test_subject
    test = features[is_person & is_test(features, test_start_sec)]
    norm_test = normalized[is_person & is_test(normalized, test_start_sec)]
    calib = features[is_person & is_calibration(features, K)]
    norm_calib = normalized[is_person & is_calibration(normalized, K)]

    # Training sets for the two "add the person's windows" methods
    plus_train, plus_weights = weighted_union(features[~is_person], calib)
    norm_train, norm_weights = weighted_union(normalized[~is_person], norm_calib)

    rows = []
    for task, label_map in TASKS.items():
        scores = {
            "generic": fit_and_score(features[~is_person], test, label_map,
                                     feature_columns=feature_columns),
            "generic_plus_calib": fit_and_score(plus_train, test, label_map, plus_weights,
                                                feature_columns),
            "norm_plus_calib": fit_and_score(norm_train, norm_test, label_map, norm_weights,
                                             feature_columns),
            "person_only": fit_and_score(calib, test, label_map,
                                         feature_columns=feature_columns),
        }
        for method, score in scores.items():
            rows.append({"subject": test_subject, "task": task, "method": method, "k": K,
                         **score, "n_calib_windows": 0 if method == "generic" else len(calib),
                         "n_test_windows": len(test)})
    return rows


def print_gains(results):
    """Per variant and task: F1 per method and its gain over generic."""
    for task in TASKS:
        print(f"\n===== {task}, k = {K} min per condition (mean ± SD over subjects) =====")
        print(f"{'variant':<18}{'method':<20}{'F1':>15}{'gain':>9}{'better':>8}{'worse':>7}{'p':>8}")
        for variant in VARIANTS:
            block = results[(results["variant"] == variant) & (results["task"] == task)]
            wide = block.pivot(index="subject", columns="method", values="f1_macro")
            for method in METHODS:
                f1 = f"{wide[method].mean():.3f} ± {wide[method].std():.3f}"
                if method == "generic":
                    print(f"{variant:<18}{method:<20}{f1:>15}")
                    continue
                diff = wide[method] - wide["generic"]
                p_value = np.nan if np.all(diff == 0) else wilcoxon(wide[method], wide["generic"]).pvalue
                print(f"{'':<18}{method:<20}{f1:>15}{diff.mean():>+9.3f}"
                      f"{(diff > 0).sum():>8}{(diff < 0).sum():>7}{p_value:>8.3f}")


def check_reference_matches(results):
    """The reference variant must reproduce personalize_rf.csv exactly."""
    old = pd.read_csv(PERSONALIZATION_CSV)
    old = old[(old["k"] == K) & old["method"].isin(METHODS)]
    new = results[results["variant"] == "main_test180"]
    merged = new.merge(old, on=["subject", "task", "method"], suffixes=("", "_old"))
    max_diff = (merged["f1_macro"] - merged["f1_macro_old"]).abs().max()
    print(f"\nReference check: {len(merged)} rows matched personalize_rf.csv, "
          f"largest F1 difference = {max_diff:.4f}")


def print_test_counts(features):
    """Test windows per class with the normal and the gapped test start."""
    print("\nTest windows per class, summed over subjects (min per subject in brackets)")
    print(f"{'test start':<12}" + "".join(f"{config.LABEL_NAMES[l]:>18}" for l in config.KEEP_LABELS))
    for start in [config.TEST_START_MIN * 60, GAP_TEST_START_SEC]:
        test = features[is_test(features, start)]
        counts = test.groupby(["subject", "label"]).size().unstack("label")
        cells = "".join(f"{f'{counts[l].sum()} ({counts[l].min()})':>18}" for l in config.KEEP_LABELS)
        print(f"{f'>= {start} s':<12}{cells}")


# ---------- Check 3: temperature drift ----------

def print_temperature_drift(features):
    """Does temp_mean simply rise with time? Looked at within the baseline block."""
    print("\nTemperature drift. rho = Spearman(temp_mean, time) within baseline;")
    print("change = last minus first baseline window (°C); order = condition order in session")
    names = {label: config.LABEL_NAMES[label] for label in config.KEEP_LABELS}
    rows = []
    for subject in config.SUBJECTS:
        sub = features[features["subject"] == subject]
        base = sub[sub["label"] == BASELINE_LABEL].sort_values("start_sec")
        row = {"subject": subject,
               "rho": base["temp_mean"].corr(base["condition_time_sec"], method="spearman"),
               "change": base["temp_mean"].iloc[-1] - base["temp_mean"].iloc[0]}
        for label, name in names.items():
            row[f"temp_{name}"] = sub.loc[sub["label"] == label, "temp_mean"].mean()
        # Order the conditions by when their block starts in the session
        block_start = sub.groupby("label")["start_sec"].min().sort_values()
        row["order"] = "-".join(names[label][0].upper() for label in block_start.index)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("subject")
    print(table.round(2).to_string())
    print(f"median rho = {table['rho'].median():.2f}; "
          f"rho > 0.5 in {(table['rho'] > 0.5).sum()}/{len(table)} subjects")
    stress = table[f"temp_{names[STRESS_LABEL]}"]
    base = table[f"temp_{names[BASELINE_LABEL]}"]
    print(f"stress warmer than baseline in {(stress > base).sum()}/{len(table)} subjects")


# ---------- Check 4: early vs full baseline ----------

def print_early_baseline(features):
    """Is the first 1-3 minutes of baseline a good stand-in for the whole baseline?

    'offset / stress effect' = (early mean - full baseline mean) / (stress mean
    - full baseline mean). Near 0: the early minutes are representative. Near
    +-1 or beyond: subtracting the early mean shifts this person's data by as
    much as stress itself does, so their baseline can look like stress.
    """
    columns = config.MAIN_FEATURES
    ratios = {}
    for subject in config.SUBJECTS:
        sub = features[features["subject"] == subject]
        base = sub[sub["label"] == BASELINE_LABEL]
        full = base[columns].mean()
        stress = sub.loc[sub["label"] == STRESS_LABEL, columns].mean()
        table = pd.DataFrame({f"first_{k}min": base.loc[is_calibration(base, k), columns].mean()
                              for k in config.CALIBRATION_MINUTES if k > 0})
        table["full_baseline"] = full
        table["stress"] = stress
        table["offset/stress_effect"] = (table[f"first_{K}min"] - full) / (stress - full)
        ratios[subject] = table["offset/stress_effect"]
        if subject in NORM_PROBLEM_SUBJECTS:
            print(f"\n{subject}: mean features, early baseline vs full baseline vs stress")
            print(table.round(3).to_string())

    others = pd.DataFrame(ratios).drop(columns=NORM_PROBLEM_SUBJECTS)
    print(f"\nFor context, median |offset / stress effect| at k = {K} in the other "
          f"{others.shape[1]} subjects:")
    print(others.abs().median(axis=1).round(2).to_string())


def main():
    features = pd.read_csv(FEATURES_CSV)

    rows = []
    for variant, (feature_columns, test_start_sec) in VARIANTS.items():
        normalized = baseline_normalize(features, K, feature_columns)
        for subject in tqdm(config.SUBJECTS, desc=variant, leave=False):
            for row in evaluate_subject(features, normalized, feature_columns,
                                        test_start_sec, subject):
                rows.append({"variant": variant, "test_start_sec": test_start_sec,
                             "n_features": len(feature_columns), **row})
    results = pd.DataFrame(rows)
    results.to_csv(ROBUSTNESS_CSV, index=False)
    print(f"Saved {ROBUSTNESS_CSV}")

    check_reference_matches(results)
    print_test_counts(features)
    print_gains(results)
    print_temperature_drift(features)
    print_early_baseline(features)


if __name__ == "__main__":
    main()
