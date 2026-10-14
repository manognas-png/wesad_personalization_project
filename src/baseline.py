"""Generic (non-personalized) baselines with leave-one-subject-out (LOSO).

Run from the repo root:  python -m src.baseline   (after python -m src.features)

For each subject: train on the other 14, test on that subject's fixed test set.
This is the "0 minutes of personal data" starting point.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import config
from src.features import FEATURES_CSV, META_COLUMNS

BASELINE_CSV = config.RESULTS_DIR / "baseline_generic.csv"

# task name -> how raw labels (1/2/3) become class numbers
TASKS = {"binary": config.BINARY_MAP, "3class": config.THREECLASS_MAP}


def make_models():
    """Fresh, untrained models (a new set is needed for every fold)."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=config.RANDOM_SEED, n_jobs=-1),
        "logistic": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
    }


def make_pipeline(model):
    """Fill NaNs -> scale -> model. Fitting the pipeline fits all three steps."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # NaN -> training median
        ("scale", StandardScaler()),                   # mean 0, std 1 (training stats)
        ("model", model),
    ])


def evaluate_subject(features, feature_columns, test_subject):
    """Train on everyone else, test on one subject. Returns a list of result rows."""
    is_test_subject = features["subject"] == test_subject
    train = features[~is_test_subject]
    # Fixed test set: only windows starting late enough into each condition,
    # so the early minutes stay free to be used as calibration data later
    is_late = features["condition_time_sec"] >= config.TEST_START_MIN * 60
    test = features[is_test_subject & is_late]

    rows = []
    for task, label_map in TASKS.items():
        y_train = train["label"].map(label_map)
        y_test = test["label"].map(label_map)
        for model_name, model in make_models().items():
            pipeline = make_pipeline(model)
            # fit() only ever sees training subjects, so nothing about the
            # test subject (medians, means, stds) leaks into the model
            pipeline.fit(train[feature_columns], y_train)
            predicted = pipeline.predict(test[feature_columns])
            rows.append({
                "subject": test_subject,
                "task": task,
                "model": model_name,
                "accuracy": accuracy_score(y_test, predicted),
                # macro = average the F1 of each class, so small classes count equally
                "f1_macro": f1_score(y_test, predicted, average="macro"),
            })
    return rows


def main():
    features = pd.read_csv(FEATURES_CSV)
    feature_columns = [column for column in features.columns if column not in META_COLUMNS]

    rows = []
    for subject in tqdm(config.SUBJECTS, desc="LOSO"):
        rows.extend(evaluate_subject(features, feature_columns, subject))
    results = pd.DataFrame(rows)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(BASELINE_CSV, index=False)
    print(f"\nSaved {BASELINE_CSV}")

    # Mean and spread ACROSS SUBJECTS for each task/model pair
    summary = results.groupby(["task", "model"])[["f1_macro", "accuracy"]].agg(["mean", "std"])
    print(f"\nGeneric baseline, LOSO over {results['subject'].nunique()} subjects (mean ± SD)")
    print(f"{'task':<8}{'model':<16}{'F1 macro':>16}{'accuracy':>16}")
    print("-" * 56)
    for (task, model_name), row in summary.iterrows():
        f1 = f"{row[('f1_macro', 'mean')]:.3f} ± {row[('f1_macro', 'std')]:.3f}"
        accuracy = f"{row[('accuracy', 'mean')]:.3f} ± {row[('accuracy', 'std')]:.3f}"
        print(f"{task:<8}{model_name:<16}{f1:>16}{accuracy:>16}")


if __name__ == "__main__":
    main()
