"""Evaluate a Logistic Regression baseline using development data only.

Run from project root:
    python -m classification.train_baseline

The reserved 20% test set is NOT used for model selection or evaluation here.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import issparse

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from classification.features import select_classification_features
from classification.preprocessing import build_preprocessor
from classification.split import DATA_PATH, split_data, make_cv_folds

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "classification" / "tables"
FIGURES = ROOT / "outputs" / "classification" / "figures"

# Evaluates both correct classification and detection of cancellations (class 1).
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# Choose which baseline models to run.
MODELS_TO_RUN = [
    "random_forest",
    "hist_gradient_boosting",
]

# Which model should we use for feature importance?
IMPORTANCE_MODEL = "random_forest"

def to_dense(X):
    """Convert sparse preprocessing output to dense if necessary."""
    return X.toarray() if issparse(X) else X


def build_model(model_name):
    """Return a fresh, unfitted preprocessing + classifier pipeline."""

    if model_name == "logreg":
        classifier = LogisticRegression(max_iter=1000)

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps = [
        ("preprocessing", build_preprocessor(scale_numeric=True))
    ]

    # Histogram Gradient Boosting requires dense input.
    if model_name == "hist_gradient_boosting":
        steps.append((
            "to_dense",
            FunctionTransformer(to_dense)
        ))

    steps.append(("classifier", classifier))

    return Pipeline(steps)



def evaluate_cv(X_dev, y_dev,model_name, group_aware=False):
    """Evaluate on 5 validation folds taken only from development data."""
    folds = make_cv_folds(X_dev, y_dev, group_aware=group_aware)
    results = cross_validate(
        build_model(model_name), X_dev, y_dev,
        cv=folds,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )

    # One row per fold. Keep values as fractions (0 to 1), not percentages.
    return pd.DataFrame({
        "fold": range(1, len(folds) + 1),
        **{name: results[f"test_{name}"] for name in SCORING},
        "fit_seconds": results["fit_time"],
    })

def analyze_feature_importance(X_dev, y_dev, model_name):
    """Calculate importance using one development validation fold."""

    folds = make_cv_folds(X_dev, y_dev)

    # Use the first fold for a preliminary importance experiment.
    train_pos, val_pos = folds[0]

    X_train = X_dev.iloc[train_pos]
    X_val = X_dev.iloc[val_pos]

    y_train = y_dev.iloc[train_pos]
    y_val = y_dev.iloc[val_pos]

    model = build_model(model_name)

    # Fit only on this fold's training data.
    model.fit(X_train, y_train)

    # Randomly shuffle each original feature in validation data.
    importance = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )

    table = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print(f"\n=== {model_name.upper()} FEATURE IMPORTANCE ===")
    print(table.round(4).to_string(index=False))

    table.to_csv(
        TABLES / f"{model_name}_permutation_importance.csv",
        index=False,
    )

    # Create a figure for our report.
    plot_table = table.sort_values("importance_mean")

    plt.figure(figsize=(9, 6))
    plt.barh(
        plot_table["feature"],
        plot_table["importance_mean"],
    )
    plt.xlabel("Mean decrease in validation ROC-AUC")
    plt.title(f"Permutation Importance — {model_name}")
    plt.tight_layout()

    plt.savefig(
        FIGURES / f"{model_name}_permutation_importance.png",
        dpi=150,
    )
    plt.close()

def main():
    df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    X, y = select_classification_features(df)
    X_dev, X_test, y_dev, y_test = split_data(X, y)

    print("Dataset:", DATA_PATH.name)
    print("Development bookings:", len(X_dev))
    print("Reserved test bookings:", len(X_test))
    print("Final test set is NOT evaluated.")

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Compare the selected baseline models.
    for model_name in MODELS_TO_RUN:

        print(f"\n========== {model_name.upper()} ==========")

        for strategy, group_aware in [
            ("stratified", False),
            ("group_aware", True),
        ]:

            fold_table = evaluate_cv(
                X_dev,
                y_dev,
                model_name,
                group_aware=group_aware,
            )

            summary = fold_table[list(SCORING)].agg(["mean", "std"])

            print(f"\n{strategy} 5-fold CV:")
            print(summary.round(4).to_string())

            fold_table.to_csv(
                TABLES / f"{model_name}_{strategy}_folds.csv",
                index=False,
            )

            summary.to_csv(
                TABLES / f"{model_name}_{strategy}_summary.csv"
            )

    # Feature importance is calculated separately from CV scores.
    if IMPORTANCE_MODEL is not None:
        analyze_feature_importance(
            X_dev,
            y_dev,
            IMPORTANCE_MODEL,
        )

    print("\nAll experiments complete. Final test set untouched.")


if __name__ == "__main__":
    main()
