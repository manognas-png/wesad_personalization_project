"""Feature-set ablation for the generic LOSO baseline -> results/ablation_feature_sets.csv.

Run from the repo root:  python -m src.ablation   (after python -m src.features)

Question: how much of the baseline score comes from physiology, and how much
from "the wrist was moving / the signal was noisy"? We re-run the same LOSO
baseline with different groups of features and compare.
"""
import pandas as pd
from sklearn.inspection import permutation_importance
from tqdm import tqdm

from src import config
from src.baseline import TASKS, evaluate_subject, make_models, make_pipeline
from src.features import FEATURES_CSV, META_COLUMNS

ABLATION_CSV = config.RESULTS_DIR / "ablation_feature_sets.csv"

HRV_COLUMNS = ["bvp_sdnn", "bvp_rmssd", "bvp_pnn50"]
QUALITY_COLUMN = "bvp_ibi_valid_frac"
PERMUTATION_REPEATS = 10    # shuffle each feature this many times and average
IMPORTANCE_SETS = ["A_all", "C_physiology"]


def make_feature_sets(all_columns):
    """Feature-set name -> list of feature columns."""
    eda = [column for column in all_columns if column.startswith("eda_")]
    temp = [column for column in all_columns if column.startswith("temp_")]
    acc = [column for column in all_columns if column.startswith("acc_")]
    heart = ["bvp_hr_mean"] + HRV_COLUMNS  # peak-based heart features
    return {
        "A_all": list(all_columns),
        "B_no_valid_frac": [column for column in all_columns if column != QUALITY_COLUMN],
        "C_physiology": eda + temp + heart,
        # Same as C but without the columns that are mostly NaN under stress.
        # If C is only good because of WHERE the NaNs are, this should be worse.
        "C_no_hrv": eda + temp + ["bvp_hr_mean"],
        # Like C_no_hrv, but heart rate comes from the spectrum, so it is never
        # missing (rejected as main set: it is often wrong, and adds nothing to D)
        "C2_spectral_hr": eda + temp + ["bvp_hr_spectral"],
        "D_eda_temp": eda + temp,
        "E_acc_only": acc,
        "F_valid_frac_only": [QUALITY_COLUMN],
    }


def importance_for_subject(features, feature_columns, test_subject):
    """Permutation importance of each feature for RF, on one test subject.

    Shuffle one feature's values in the TEST set and see how much macro F1
    drops. A big drop means the model really relied on that feature.
    """
    is_test_subject = features["subject"] == test_subject
    is_late = features["condition_time_sec"] >= config.TEST_START_MIN * 60
    train = features[~is_test_subject]
    test = features[is_test_subject & is_late]

    rows = []
    for task, label_map in TASKS.items():
        pipeline = make_pipeline(make_models()["random_forest"])
        pipeline.fit(train[feature_columns], train["label"].map(label_map))
        # Shuffling happens BEFORE the pipeline, so imputing and scaling are
        # applied to the shuffled values just like for real data
        result = permutation_importance(
            pipeline, test[feature_columns], test["label"].map(label_map),
            scoring="f1_macro", n_repeats=PERMUTATION_REPEATS,
            random_state=config.RANDOM_SEED)
        for column, drop in zip(feature_columns, result.importances_mean):
            rows.append({"subject": test_subject, "task": task,
                         "feature": column, "f1_drop": drop})
    return rows


def print_score_table(results):
    """Feature set x task x model -> macro F1 mean ± SD across subjects."""
    summary = results.groupby(["feature_set", "task", "model"])["f1_macro"].agg(["mean", "std"])
    summary["text"] = summary["mean"].map("{:.3f}".format) + " ± " + summary["std"].map("{:.3f}".format)
    # unstack turns the task/model index levels into columns: one row per feature set
    table = summary["text"].unstack(["task", "model"])
    print("\nMacro F1, LOSO over subjects (mean ± SD)")
    print(table.to_string())


def print_motion_correlation(features):
    """Does more wrist movement go with fewer valid heartbeats?"""
    print(f"\nCorrelation between acc_std and {QUALITY_COLUMN} (Spearman)")
    print(f"{'condition':<12}{'all windows pooled':>20}{'median within subject':>24}")
    for label in config.KEEP_LABELS:
        block = features[features["label"] == label]
        # Spearman = correlation of ranks, so a few huge movements don't dominate
        pooled = block["acc_std"].corr(block[QUALITY_COLUMN], method="spearman")
        # Within one person at a time, so differences BETWEEN people can't fake it
        per_subject = block.groupby("subject")[["acc_std", QUALITY_COLUMN]].apply(
            lambda rows: rows["acc_std"].corr(rows[QUALITY_COLUMN], method="spearman"))
        print(f"{config.LABEL_NAMES[label]:<12}{pooled:>20.2f}{per_subject.median():>24.2f}")


def main():
    features = pd.read_csv(FEATURES_CSV)
    all_columns = [column for column in features.columns if column not in META_COLUMNS]
    feature_sets = make_feature_sets(all_columns)

    # ---------- Scores for every feature set ----------
    rows = []
    for set_name, columns in feature_sets.items():
        for subject in tqdm(config.SUBJECTS, desc=set_name, leave=False):
            for row in evaluate_subject(features, columns, subject):
                rows.append({"feature_set": set_name, "n_features": len(columns), **row})
    results = pd.DataFrame(rows)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ABLATION_CSV, index=False)
    print(f"\nSaved {ABLATION_CSV}")
    print_score_table(results)

    # ---------- RF permutation importance ----------
    for set_name in IMPORTANCE_SETS:
        columns = feature_sets[set_name]
        rows = []
        for subject in tqdm(config.SUBJECTS, desc=f"importance {set_name}", leave=False):
            rows.extend(importance_for_subject(features, columns, subject))
        importance = pd.DataFrame(rows)
        # Average the F1 drop over the 15 test subjects, one column per task
        table = importance.groupby(["feature", "task"])["f1_drop"].mean().unstack("task")
        print(f"\nRF permutation importance, set {set_name} "
              f"(mean drop in macro F1 over test subjects)")
        print(table.sort_values("binary", ascending=False).round(3).to_string())

    print_motion_correlation(features)


if __name__ == "__main__":
    main()
