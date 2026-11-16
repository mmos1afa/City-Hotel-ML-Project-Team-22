import sys
from pathlib import Path
from joblib import dump

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from sklearn.model_selection import (
    KFold,
    cross_validate,
)

from models import build_regression_pipeline, get_regression_models
from workflow import get_model_artifact_path, load_training_data

cross_validation = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}


def main():
    X_train, y_train = load_training_data()
    models = get_regression_models()
    for model_name, model in models.items():
        pipeline = build_regression_pipeline(
            X_train,
            model=model,
        )
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring=scoring,
        )

        print(f"\n===== {model_name} (Baseline 5-Fold CV) =====")
        fold_scores = {}
        for metric in ("MAE", "RMSE", "R2"):
            scores = cv_results[f"test_{metric}"]
            if metric in ("MAE", "RMSE"):
                scores = -scores
            fold_scores[metric] = scores

        for fold_index in range(cross_validation.n_splits):
            print(
                f"Fold {fold_index + 1}: "
                f"MAE={fold_scores['MAE'][fold_index]:.2f}, "
                f"RMSE={fold_scores['RMSE'][fold_index]:.2f}, "
                f"R2={fold_scores['R2'][fold_index]:.4f}"
            )

        print("CV summary:")
        for metric, scores in fold_scores.items():
            digits = 4 if metric == "R2" else 2
            print(
                f"{metric}: {scores.mean():.{digits}f} "
                f"± {scores.std():.{digits}f}"
            )

        pipeline.fit(X_train, y_train)
        artifact_path = get_model_artifact_path("baseline", model_name)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        dump(pipeline, artifact_path)
        print(f"Saved baseline model to: {artifact_path}")


if __name__ == "__main__":
    main()
