"""Personalization experiment with Random Forest on config.MAIN_FEATURES.

Run from the repo root:  python -m src.personalize_rf   (after python -m src.features)

For each held-out subject (LOSO), task and k (minutes PER CONDITION of the
person's own data), compare five ways of using that data. The test windows
are the same for every k, so the scores at different k are comparable.
See "Personalization design" in CLAUDE.md.
"""
import matplotlib
matplotlib.use("Agg")  # draw straight to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src import config
from src.baseline import TASKS, make_pipeline
from src.features import FEATURES_CSV

RESULTS_CSV = config.RESULTS_DIR / "personalization_rf.csv"
FEATURES = config.MAIN_FEATURES

METHODS = ["generic", "baseline_norm", "generic_plus_calib", "norm_plus_calib", "person_only"]
PERSON_WEIGHT_SHARE = 0.25   # person's windows = 25% of the total training weight
SANITY_SUBJECT = "S2"
HIGHLIGHT_SUBJECTS = ["S14", "S10"]  # the two weakest subjects in the generic baseline

# Look the baseline label up by name instead of writing the number 1
BASELINE_LABEL = next(label for label, name in config.LABEL_NAMES.items() if name == "baseline")

# Chart colors: generic is the gray reference line, methods take fixed palette slots
METHOD_STYLE = {
    "generic":            {"color": "#898781", "marker": "o", "linestyle": "--"},
    "baseline_norm":      {"color": "#2a78d6", "marker": "s", "linestyle": "-"},
    "generic_plus_calib": {"color": "#eb6834", "marker": "^", "linestyle": "-"},
    "norm_plus_calib":    {"color": "#1baf7a", "marker": "D", "linestyle": "-"},
    "person_only":        {"color": "#eda100", "marker": "v", "linestyle": "-"},
}
INK_PRIMARY, INK_MUTED, GRID = "#1f1e1c", "#898781", "#e6e5e0"


# ---------- Which windows go where ----------

def is_calibration(features, k):
    """Windows that END within the first k minutes of their condition block."""
    return features["condition_time_sec"] + config.WINDOW_SEC <= k * 60


def is_test(features, test_start_sec=config.TEST_START_MIN * 60):
    """The fixed test set: windows starting >= test_start_sec into their condition."""
    return features["condition_time_sec"] >= test_start_sec


def check_no_overlap(features):
    """Assert that no calibration window shares a single second with a test window."""
    for subject, rows in features.groupby("subject"):
        test_start = rows.loc[is_test(rows), "start_sec"].to_numpy()
        reference_test = None
        for k in config.CALIBRATION_MINUTES:
            calib_start = rows.loc[is_calibration(rows, k), "start_sec"].to_numpy()
            # Two windows [a, a+60) and [b, b+60) overlap when |a - b| < 60.
            # [:, None] makes a (n_calib, n_test) grid of every pair.
            gaps = np.abs(calib_start[:, None] - test_start[None, :])
            assert np.all(gaps >= config.WINDOW_SEC), f"{subject} k={k}: calib overlaps test"
            # The test set must not depend on k
            test_index = rows.index[is_test(rows)]
            if reference_test is None:
                reference_test = test_index
            assert test_index.equals(reference_test), f"{subject}: test set changes with k"
    print("Check passed: no calibration window overlaps a test window, "
          "and the test set is identical for every k.")


def print_sanity_counts(features):
    """Window counts for one subject, so we can eyeball the split."""
    rows = features[features["subject"] == SANITY_SUBJECT]
    print(f"\nWindow counts for {SANITY_SUBJECT} (per condition: baseline/stress/amusement)")
    print(f"{'k (min)':<9}{'n_calib':>9}{'calib per cond':>17}{'n_test':>9}{'test per cond':>17}")
    for k in config.CALIBRATION_MINUTES:
        calib = rows[is_calibration(rows, k)]
        test = rows[is_test(rows)]
        per_condition = lambda block: "/".join(
            str((block["label"] == label).sum()) for label in config.KEEP_LABELS)
        print(f"{k:<9}{len(calib):>9}{per_condition(calib):>17}"
              f"{len(test):>9}{per_condition(test):>17}")


# ---------- Personalization tricks ----------

def baseline_normalize(features, k, feature_columns=FEATURES):
    """Subtract each subject's mean over their own first k minutes of BASELINE.

    Label-free onboarding: "sit still for k minutes", then everything that
    person produces is measured relative to their own resting values. Every
    subject (training and test) is treated the same way, and each subject's
    reference uses only that subject's own calibration windows, never test ones.
    """
    is_reference = (features["label"] == BASELINE_LABEL) & is_calibration(features, k)
    reference_mean = features[is_reference].groupby("subject")[feature_columns].mean()
    normalized = features.copy()
    # reference_mean.loc[subjects] repeats each subject's mean once per window
    normalized[feature_columns] = (features[feature_columns].to_numpy()
                            - reference_mean.loc[features["subject"]].to_numpy())
    return normalized


def person_weight(n_others, n_person):
    """Weight per person window so that person windows = PERSON_WEIGHT_SHARE of total.

    Solve  n_person*w / (n_others + n_person*w) = share  for w.
    """
    share = PERSON_WEIGHT_SHARE
    return share / (1 - share) * n_others / n_person


# ---------- Training and scoring ----------

def fit_and_score(train, test, label_map, sample_weight=None, feature_columns=FEATURES):
    """Fit a fresh imputer -> scaler -> RF pipeline on train, score it on test."""
    model = RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_SEED, n_jobs=-1)
    pipeline = make_pipeline(model)
    # "model__sample_weight" passes the weights to the step named "model" only
    pipeline.fit(train[feature_columns], train["label"].map(label_map),
                 model__sample_weight=sample_weight)
    predicted = pipeline.predict(test[feature_columns])
    y_test = test["label"].map(label_map)
    return {"f1_macro": f1_score(y_test, predicted, average="macro"),
            "accuracy": accuracy_score(y_test, predicted)}


def weighted_union(others, person):
    """Stack other subjects + the person's windows, with the person up-weighted."""
    train = pd.concat([others, person])
    weights = np.concatenate([np.ones(len(others)),
                              np.full(len(person), person_weight(len(others), len(person)))])
    return train, weights


def evaluate_subject(features, normalized_by_k, test_subject):
    """All methods x tasks x k for one held-out subject. Returns result rows."""
    is_person = features["subject"] == test_subject
    test = features[is_person & is_test(features)]
    rows = []

    def add(task, method, k, scores, n_calib):
        rows.append({"subject": test_subject, "task": task, "method": method, "k": k,
                     **scores, "n_calib_windows": n_calib, "n_test_windows": len(test)})

    for task, label_map in TASKS.items():
        # Method 1: generic does not use the person's data, so one fit serves every k
        generic = fit_and_score(features[~is_person], test, label_map)
        for k in config.CALIBRATION_MINUTES:
            add(task, "generic", k, generic, 0)

        for k in config.CALIBRATION_MINUTES:
            if k == 0:
                continue  # no personal data -> only generic exists
            calib = features[is_person & is_calibration(features, k)]
            n_calib_baseline = (calib["label"] == BASELINE_LABEL).sum()

            # Method 2: everyone measured relative to their own early baseline
            normalized = normalized_by_k[k]
            norm_others = normalized[~is_person]
            norm_test = normalized[is_person & is_test(normalized)]
            add(task, "baseline_norm", k,
                fit_and_score(norm_others, norm_test, label_map), n_calib_baseline)

            # Method 3: add the person's labelled windows, up-weighted
            train, weights = weighted_union(features[~is_person], calib)
            add(task, "generic_plus_calib", k,
                fit_and_score(train, test, label_map, weights), len(calib))

            # Method 4: methods 2 + 3 together (calibration windows normalized too)
            norm_calib = normalized[is_person & is_calibration(normalized, k)]
            train, weights = weighted_union(norm_others, norm_calib)
            add(task, "norm_plus_calib", k,
                fit_and_score(train, norm_test, label_map, weights), len(calib))

            # Method 5: only the person's own calibration windows
            add(task, "person_only", k, fit_and_score(calib, test, label_map), len(calib))
    return rows


# ---------- Reporting ----------

def print_summary(results):
    """Mean ± SD macro F1 across subjects, method x k, per task."""
    for task in TASKS:
        block = results[results["task"] == task]
        stats = block.groupby(["method", "k"])["f1_macro"].agg(["mean", "std"])
        text = stats["mean"].map("{:.3f}".format) + " ± " + stats["std"].map("{:.3f}".format)
        table = text.unstack("k").reindex(METHODS).fillna("-")
        print(f"\nMacro F1, {task}, mean ± SD over {block['subject'].nunique()} subjects")
        print("(columns = minutes of personal data per condition)")
        print(table.to_string())


def print_wilcoxon(results):
    """Paired Wilcoxon signed-rank test of each method vs generic, per k."""
    print("\nPaired Wilcoxon signed-rank test vs generic (two-sided, n = subjects).")
    print("No correction for multiple tests: treat p-values near 0.05 with care.")
    for task in TASKS:
        block = results[results["task"] == task]
        # One row per subject, one column per (method, k)
        wide = block.pivot_table(index="subject", columns=["method", "k"], values="f1_macro")
        print(f"\n{task}")
        print(f"{'method':<20}{'k':>3}{'mean diff':>11}{'better':>8}{'worse':>7}{'p':>9}")
        for method in METHODS[1:]:
            for k in config.CALIBRATION_MINUTES[1:]:
                diff = wide[(method, k)] - wide[("generic", k)]
                if np.all(diff == 0):
                    p_value = np.nan  # identical scores: nothing to test
                else:
                    p_value = wilcoxon(wide[(method, k)], wide[("generic", k)]).pvalue
                print(f"{method:<20}{k:>3}{diff.mean():>+11.3f}{(diff > 0).sum():>8}"
                      f"{(diff < 0).sum():>7}{p_value:>9.3f}")


def style_axes(ax):
    """Recessive grid and axes so the data lines stand out."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_PRIMARY)


def plot_f1_vs_minutes(results, task):
    """Mean macro F1 (± standard error across subjects) vs minutes per condition."""
    block = results[results["task"] == task]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method in METHODS:
        scores = block[block["method"] == method].groupby("k")["f1_macro"]
        mean = scores.mean()
        # Standard error = SD / sqrt(number of subjects)
        standard_error = scores.std() / np.sqrt(scores.count())
        style = METHOD_STYLE[method]
        ax.fill_between(mean.index, mean - standard_error, mean + standard_error,
                        color=style["color"], alpha=0.12, linewidth=0)
        ax.plot(mean.index, mean, color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=2, markersize=7, label=method)
    ax.set_xticks(config.CALIBRATION_MINUTES)
    ax.set_xlabel("Minutes of personal data per condition", color=INK_PRIMARY)
    ax.set_ylabel("Macro F1", color=INK_PRIMARY)
    ax.set_title(f"RF personalization, {task} (mean ± SE over subjects)",
                 color=INK_PRIMARY, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"f1_vs_minutes_{task}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_per_subject(results, task="binary"):
    """Per-subject change vs generic at the largest k, for the best method."""
    k = max(config.CALIBRATION_MINUTES)
    block = results[(results["task"] == task) & (results["k"] == k)]
    mean_f1 = block.groupby("method")["f1_macro"].mean().drop("generic")
    best = mean_f1.idxmax()

    wide = block.pivot(index="subject", columns="method", values="f1_macro")
    change = (wide[best] - wide["generic"]).sort_values()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = [METHOD_STYLE[best]["color"] if subject in HIGHLIGHT_SUBJECTS else "#b7d3f6"
              for subject in change.index]
    ax.bar(change.index, change.to_numpy(), color=colors, width=0.7)
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    for position, (subject, value) in enumerate(change.items()):
        if subject in HIGHLIGHT_SUBJECTS:
            # Label only the highlighted bars, just past the end of the bar
            offset = 0.01 if value >= 0 else -0.01
            ax.text(position, value + offset, f"{value:+.2f}", ha="center",
                    va="bottom" if value >= 0 else "top", color=INK_PRIMARY, fontsize=9)
    ax.set_xlabel("Subject (sorted by change)", color=INK_PRIMARY)
    ax.set_ylabel(f"Macro F1 change vs generic", color=INK_PRIMARY)
    ax.set_title(f"{best} at k = {k} min per condition, {task} "
                 f"(highlighted: {', '.join(HIGHLIGHT_SUBJECTS)})",
                 color=INK_PRIMARY, loc="left", fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"per_subject_{task}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    features = pd.read_csv(FEATURES_CSV)

    # Sanity checks BEFORE spending time on training
    print_sanity_counts(features)
    check_no_overlap(features)

    # Normalized tables depend only on k (each subject uses only its own
    # calibration windows), so compute them once instead of once per fold
    normalized_by_k = {k: baseline_normalize(features, k)
                       for k in config.CALIBRATION_MINUTES if k > 0}

    rows = []
    for subject in tqdm(config.SUBJECTS, desc="LOSO"):
        rows.extend(evaluate_subject(features, normalized_by_k, subject))
    results = pd.DataFrame(rows)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {RESULTS_CSV}")

    print_summary(results)
    print_wilcoxon(results)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        plot_f1_vs_minutes(results, task)
    plot_per_subject(results)


if __name__ == "__main__":
    main()
