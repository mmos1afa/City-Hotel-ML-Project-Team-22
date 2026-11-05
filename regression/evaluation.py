import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression_model(model, X_val, y_val, model_name="Baseline Model"):
    """
    Evaluates the fitted model pipeline on validation data.
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