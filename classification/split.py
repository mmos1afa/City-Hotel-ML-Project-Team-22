"""Reproducible classification holdout and cross-validation fold definitions.

Run from the repository root: python -m classification.split
No models or preprocessing are fitted by this file.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from classification.features import (select_classification_features)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned" / "H2_cleaned3.csv"
OUTPUT_PATH = ROOT / "outputs" / "classification" / "tables"
SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5


def split_data(X, y):
    """Reserve a stratified 20% holdout; return 80% development and 20% test."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y 
    )#stratify tells the function to split the data based on y (iscanceled) so that the proportion of canceled and not canceled bookings is the same in both the development and test sets.



def get_feature_groups(X):
    """Give identical rows of selected predictors the same group ID."""
    patterns = pd.MultiIndex.from_frame(X)
    group_ids, _ = pd.factorize(patterns, sort=False)
    return group_ids



def make_cv_folds(X_dev, y_dev, group_aware=False):
    """Return 5 (train_positions, validation_positions) pairs within development."""
    if group_aware:
        groups = get_feature_groups(X_dev)
        cv = StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=SEED
        )
        return list(cv.split(X_dev, y_dev, groups=groups))

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(cv.split(X_dev, y_dev))


def main():
    df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    X, y = select_classification_features(df)
    X_dev, X_test, y_dev, y_test = split_data(X, y)

    normal_folds = make_cv_folds(X_dev, y_dev)
    group_folds = make_cv_folds(X_dev, y_dev, group_aware=True)
    groups = get_feature_groups(X_dev)

    print(f"Total bookings: {len(X):,}")
    print(f"Development: {len(X_dev):,} | Test: {len(X_test):,}")
    print(f"Cancellation rates — full: {y.mean():.2%}, "
          f"development: {y_dev.mean():.2%}, test: {y_test.mean():.2%}")
    print(f"Distinct selected-feature patterns in development: {len(set(groups)):,}")

    # Save row assignments (not predictions or fitted model artifacts).
    manifest = pd.DataFrame(index=X.index)
    manifest.index.name = "source_row_index"
    manifest["subset"] = "test"
    manifest.loc[X_dev.index, "subset"] = "development"
    manifest["stratified_cv_fold"] = pd.NA
    manifest["group_cv_fold"] = pd.NA

    for label, folds, column in [
        ("Standard stratified", normal_folds, "stratified_cv_fold"),
        ("Group-aware", group_folds, "group_cv_fold"),
    ]:
        print(f"\n{label} 5-fold validation (development data only):")
        for fold_number, (train_pos, val_pos) in enumerate(folds, start=1):
            val_index = X_dev.iloc[val_pos].index
            manifest.loc[val_index, column] = fold_number
            print(f"  Fold {fold_number}: training={len(train_pos):,}, "
                  f"validation={len(val_pos):,}, "
                  f"validation cancellation rate={y_dev.iloc[val_pos].mean():.2%}")
            if label == "Group-aware":
                assert set(groups[train_pos]).isdisjoint(set(groups[val_pos]))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / "split_manifest.csv"
    manifest.reset_index().to_csv(output_file, index=False)
    print(f"\nSplit assignments saved to: {output_file}")
    print("No preprocessing or model was fitted. Test labels are for split checks only.")


if __name__ == "__main__":
    main()
