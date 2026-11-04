from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from classification.features import select_classification_features
from classification.preprocessing import build_preprocessor
from classification.split import DATA_PATH, split_data, make_cv_folds

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "classification" / "tables"

# Evaluates both correct classification and detection of cancellations (class 1).
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def build_model():
    """Return a fresh, unfitted preprocessing + classifier pipeline."""
    return Pipeline([
        ("preprocessing", build_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def evaluate_cv(X_dev, y_dev, group_aware=False):
    """Evaluate on 5 validation folds taken only from development data."""
    folds = make_cv_folds(X_dev, y_dev, group_aware=group_aware)
    results = cross_validate(
        build_model(), X_dev, y_dev,
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


def main():
    df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    X, y = select_classification_features(df)
    X_dev, X_test, y_dev, y_test = split_data(X, y)

    print(f"Development bookings: {len(X_dev):,}")
    print(f"Reserved test bookings: {len(X_test):,} (NOT evaluated)")
    print("Model: Logistic Regression (baseline, no tuning)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, group_aware in [
        ("stratified", False),
        ("group_aware", True),
    ]:
        print(f"\n=== {name.upper()} 5-FOLD CV ===")
        fold_table = evaluate_cv(X_dev, y_dev, group_aware=group_aware)
        summary = fold_table[list(SCORING)].agg(["mean", "std"])
        print(summary.round(4).to_string())

        fold_table.to_csv(OUTPUT_DIR / f"logreg_{name}_folds.csv", index=False)
        summary.to_csv(OUTPUT_DIR / f"logreg_{name}_summary.csv")

    print("\nSaved fold results and summaries in:", OUTPUT_DIR)
    print("Test set remains untouched. No final model has been selected.")


if __name__ == "__main__":
    main()
