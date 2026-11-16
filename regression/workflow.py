from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REGRESSION_DIR = Path(__file__).resolve().parent
DATASET_PATH = (
    REGRESSION_DIR.parent
    / "data"
    / "cleaned"
    / "H2_regression_without_duplicates.csv"
)
ARTIFACT_DIR = REGRESSION_DIR.parent / "artifacts" / "regression"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _load_split():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    reg_df = pd.read_csv(DATASET_PATH)
    X = reg_df.drop(columns=["ADR"])
    y = reg_df["ADR"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def load_training_data():
    X_train, _, y_train, _ = _load_split()
    return X_train, y_train


def load_test_data():
    _, X_test, _, y_test = _load_split()
    return X_test, y_test


def get_model_artifact_path(workflow_name, model_name):
    model_filename = model_name.lower().replace(" ", "_") + ".joblib"
    return ARTIFACT_DIR / workflow_name / model_filename
