import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from joblib import load
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(model, X_val, y_val, model_name="Baseline Model"):
    """
    Evaluates a fitted regression pipeline on the supplied data.
    """
    y_pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    metrics = {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }
    
    print(f"\n===== {model_name} Evaluation Results =====")
    print(f"MAE  : {metrics['MAE']}")
    print(f"RMSE : {metrics['RMSE']}")
    print(f"R²   : {metrics['R2']}")
    
    return metrics


def main():
    from models import get_regression_models
    from workflow import get_model_artifact_path, load_test_data

    X_test, y_test = load_test_data()
    model_names = get_regression_models()
    workflows = ("baseline", "tuned")
    missing_artifacts = [
        get_model_artifact_path(workflow_name, model_name)
        for workflow_name in workflows
        for model_name in model_names
        if not get_model_artifact_path(workflow_name, model_name).exists()
    ]
    if missing_artifacts:
        missing_paths = "\n".join(str(path) for path in missing_artifacts)
        raise FileNotFoundError(
            "Run train_baseline.py and tuning.py before test evaluation. "
            f"Missing model artifacts:\n{missing_paths}"
        )

    for workflow_name in workflows:
        for model_name in model_names:
            artifact_path = get_model_artifact_path(workflow_name, model_name)
            model = load(artifact_path)
            evaluate_regression_model(
                model,
                X_test,
                y_test,
                model_name=f"{model_name} - {workflow_name.title()} (Test)",
            )


if __name__ == "__main__":
    main()