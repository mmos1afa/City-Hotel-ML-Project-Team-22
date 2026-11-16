import sys
from pathlib import Path
from joblib import dump

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import optuna
from sklearn.model_selection import KFold, cross_val_score

from models import (
    build_regression_model,
    build_regression_pipeline,
    get_regression_models,
    suggest_regression_parameters,
)
from workflow import get_model_artifact_path, load_training_data

N_TRIALS = 10
CROSS_VALIDATION = KFold(n_splits=5, shuffle=True, random_state=42)


def main():
    X_train, y_train = load_training_data()

    for model_name in get_regression_models():
        def objective(trial):
            parameters = suggest_regression_parameters(trial, model_name)
            model = build_regression_model(model_name, parameters)
            pipeline = build_regression_pipeline(X_train, model=model)
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=CROSS_VALIDATION,
                scoring="neg_mean_absolute_error",
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=N_TRIALS)
        print(
            f"\n===== {model_name} (Optuna 5-Fold CV) =====\n"
            f"Best MAE: {study.best_value:.2f}\n"
            f"Best hyperparameters: {study.best_params}"
        )

        best_model = build_regression_model(model_name, study.best_params)
        pipeline = build_regression_pipeline(X_train, model=best_model)
        pipeline.fit(X_train, y_train)

        artifact_path = get_model_artifact_path("tuned", model_name)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        dump(pipeline, artifact_path)
        print(f"Saved tuned model to: {artifact_path}")


if __name__ == "__main__":
    main()