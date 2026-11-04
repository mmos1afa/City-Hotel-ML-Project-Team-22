import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import prepare_regression_data
from models import build_regression_pipeline
from evaluation import evaluate_regression_model

# 1. Correct Project Root Path (1 level up from regression/)
project_dir = CURRENT_DIR.parent
raw_cleaned_path = project_dir / "data" / "cleaned" / "H2_cleaned3.csv"

# Verify dataset existence before reading
if not raw_cleaned_path.exists():
    raise FileNotFoundError(f"Dataset not found at: {raw_cleaned_path}. Make sure data/cleaned/H2_cleaned3.csv exists!")

df = pd.read_csv(raw_cleaned_path)
reg_df = prepare_regression_data(df)

X = reg_df.drop(columns=["ADR"])
y = reg_df["ADR"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = build_regression_pipeline(X_train)
pipeline.fit(X_train, y_train)

metrics = evaluate_regression_model(pipeline, X_val, y_val, model_name="Linear Regression (Baseline)")