"""
Predicting treatment success from pre-treatment urodynamic (UDS) metrics.

Steps:
  1. Load and clean the transcribed patient data
  2. Compare pre vs. post treatment metrics (paired t-tests)
  3. Save plots to the figures/ folder
  4. Train a random forest to predict treatment success
"""

import os

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Transcribed_Data.csv"
FIGURES_DIR = "figures"
METRICS = ["Qmax", "Pdet_max", "PdetQmax", "VH2O_cap"]


# Step 1: Load and process data
def load_and_process_data(file_name):
    raw_data = pd.read_csv(file_name, header=None)
    patients = []
    current_patient = {}

    for _, row in raw_data.iterrows():
        key = row[0]
        pre = row[1]
        post = row[2] if len(row) > 2 else None

        # A blank row means the current patient's block has ended
        if pd.isna(key) and pd.isna(pre):
            if current_patient:
                patients.append(current_patient)
                current_patient = {}

        # Start of a new patient
        elif isinstance(key, str) and "Patient No." in key:
            if current_patient:
                patients.append(current_patient)
            current_patient = {"Patient_No": pre}

        # A metric row
        elif isinstance(key, str):
            name = key.split(":")[0].strip()
            name = (name.replace("Pre-", "").replace("Post-", "")
                        .replace(" ", "_").replace("(", "").replace(")", "")
                        .replace("/", "_per_"))

            # Use short, consistent names for the four main metrics
            # (PdetQmax is checked first so it isn't mistaken for Qmax)
            if "Detrusor_pressure_at_maximum_flow_rate" in name:
                name = "PdetQmax"
            elif "Maximum_flow_rate" in name:
                name = "Qmax"
            elif "Maximum_detrusor_pressure" in name:
                name = "Pdet_max"
            elif "Bladder_capacity" in name:
                name = "VH2O_cap"

            # Some capacity cells hold several readings ("#1 = ... \ #2 = ...");
            # keep the first one
            if isinstance(pre, str) and "#1" in pre:
                pre = float(pre.split("#1 = ")[1].split("\\")[0])
            if isinstance(post, str) and "#1" in post:
                post = float(post.split("#1 = ")[1].split("\\")[0])

            if pd.notna(pre) and str(pre).strip():
                current_patient[f"Pre_{name}"] = pre
            if post is not None and pd.notna(post) and str(post).strip():
                current_patient[f"Post_{name}"] = post

    if current_patient:
        patients.append(current_patient)

    df = pd.DataFrame(patients)

    for metric in METRICS:
        for col in (f"Pre_{metric}", f"Post_{metric}"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# Step 2: Pre vs. post statistics
def analyze_uds_metrics(df):
    results = {}

    for metric in METRICS:
        pre_col, post_col = f"Pre_{metric}", f"Post_{metric}"
        if pre_col not in df.columns or post_col not in df.columns:
            continue

        df[f"{metric}_diff"] = df[post_col] - df[pre_col]
        pct = (df[post_col] - df[pre_col]) / df[pre_col] * 100
        df[f"{metric}_pct_change"] = pct.replace([np.inf, -np.inf], np.nan)

        pairs = df[[pre_col, post_col]].dropna()
        if len(pairs) >= 2:
            _, p_value = stats.ttest_rel(pairs[pre_col], pairs[post_col])
        else:
            p_value = np.nan

        results[metric] = {
            "mean_pre": df[pre_col].mean(),
            "std_pre": df[pre_col].std(),
            "mean_post": df[post_col].mean(),
            "std_post": df[post_col].std(),
            "mean_diff": df[f"{metric}_diff"].mean(),
            "mean_pct_change": df[f"{metric}_pct_change"].mean(),
            "p_value": p_value,
            "n_pairs": len(pairs),
        }

    return df, results


# Step 3: Plots (saved to figures/)
def create_visualizations(df, results):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Pre vs. post boxplots
    plt.figure(figsize=(12, 9))
    for i, metric in enumerate(METRICS, 1):
        if metric not in results:
            continue
        plt.subplot(2, 2, i)
        plt.boxplot([df[f"Pre_{metric}"].dropna(), df[f"Post_{metric}"].dropna()])
        plt.xticks([1, 2], ["Pre", "Post"])
        r = results[metric]
        plt.title(f"{metric}  (n={r['n_pairs']}, p={r['p_value']:.3f})")
        plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/pre_post_boxplots.png", dpi=150)
    plt.close()

    # Correlation between pre-treatment metrics
    pre_cols = [f"Pre_{m}" for m in METRICS if f"Pre_{m}" in df.columns]
    if pre_cols:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[pre_cols].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Between Pre-Treatment Metrics")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150)
        plt.close()


# Step 4: Predictive model
def build_predictive_model(df):
    features = [f"Pre_{m}" for m in METRICS if f"Pre_{m}" in df.columns]
    if not features:
        print("Error: no feature columns found")
        return None
    if "Qmax_pct_change" not in df.columns or "VH2O_cap_pct_change" not in df.columns:
        print("Error: need both Qmax and VH2O_cap pre/post values")
        return None

    # Only keep patients where we can tell whether treatment succeeded
    data = df.dropna(subset=["Qmax_pct_change", "VH2O_cap_pct_change"]).copy()

    # Success = flow rate up >20% AND bladder capacity up >10%
    data["Success"] = ((data["Qmax_pct_change"] > 20) &
                       (data["VH2O_cap_pct_change"] > 10)).astype(int)

    X = data[features]
    y = data["Success"]

    print(f"Patients used: {len(y)}  (success: {y.sum()}, no success: {len(y) - y.sum()})")
    min_class = y.value_counts().min()
    if y.nunique() < 2 or min_class < 2:
        print("Error: need at least 2 patients in each group to train a model")
        return None

    # Pipeline: missing values and scaling are learned from training data only,
    # so no information from the test set leaks into training
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42)),
    ])

    folds = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)

    # Cross-validation gives a more reliable score than one split on a small dataset
    cv_scores = cross_val_score(pipeline, X, y, cv=folds, scoring="f1")
    print(f"Cross-validated F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Tune on a training split, then check on held-out patients
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 3],
    }
    train_min_class = y_train.value_counts().min()
    inner_folds = StratifiedKFold(n_splits=max(2, min(5, train_min_class)),
                                  shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid, cv=inner_folds, scoring="f1")
    search.fit(X_train, y_train)
    best = search.best_estimator_

    y_pred = best.predict(X_test)
    print("\nBest settings:", search.best_params_)
    print("\nTest-set results:")
    print(classification_report(y_test, y_pred, zero_division=0))

    importance = pd.DataFrame({
        "feature": features,
        "importance": best.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)

    print("Feature importance:")
    print(importance.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(importance["feature"], importance["importance"])
    plt.title("Feature Importance for Predicting Treatment Success")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_importance.png", dpi=150)
    plt.close()

    return best


if __name__ == "__main__":
    print("Loading data...")
    df = load_and_process_data(DATA_FILE)
    print(f"Loaded {len(df)} patients")

    print("\nPre vs. post results:")
    df, results = analyze_uds_metrics(df)
    for metric, r in results.items():
        print(f"\n{metric} (n={r['n_pairs']} pairs)")
        print(f"  Pre:    {r['mean_pre']:.2f} ± {r['std_pre']:.2f}")
        print(f"  Post:   {r['mean_post']:.2f} ± {r['std_post']:.2f}")
        print(f"  Change: {r['mean_diff']:.2f} ({r['mean_pct_change']:.1f}%)")
        print(f"  p-value: {r['p_value']:.3f}")

    print("\nSaving plots...")
    create_visualizations(df, results)

    print("\nTraining model...")
    build_predictive_model(df)

    print(f"\nDone. Plots saved in the '{FIGURES_DIR}' folder.")
