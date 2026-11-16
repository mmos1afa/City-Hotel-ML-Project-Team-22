from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor


def suggest_regression_parameters(trial, model_name):
    if model_name == "Linear Regression":
        return {
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept", [True, False]
            ),
        }
    if model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 10, 20, 30]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 1.0]
            ),
        }
    if model_name == "Gradient Boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
        }
    raise ValueError(f"Unknown regression model: {model_name}")


def build_regression_model(model_name, parameters=None):
    parameters = parameters or {}
    if model_name == "Linear Regression":
        return LinearRegression(**parameters)
    if model_name == "Random Forest":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **parameters,
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            random_state=42,
            **parameters,
        )
    if model_name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **parameters,
        )
    raise ValueError(f"Unknown regression model: {model_name}")


def get_regression_models():
    return {
        "Linear Regression": build_regression_model("Linear Regression"),
        "Random Forest": build_regression_model(
            "Random Forest", {"n_estimators": 100}
        ),
        "Gradient Boosting": build_regression_model(
            "Gradient Boosting",
            {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
            },
        ),
        "XGBoost": build_regression_model(
            "XGBoost",
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        ),
    }

def build_regression_pipeline(X_train, model=None):
    """
    Builds a full ML pipeline with preprocessing and an baseline regressor.
    Ensures preprocessing parameters are fitted ONLY on training data.
    """
    if model is None:
        model = LinearRegression()

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return full_pipeline