"""Define classification preprocessing (unfitted).

Build this pipeline now, but fit it ONLY after splitting the data.
No training or fitting happens when this module is run.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "LeadTime",
    "TotalGuests",
    "TotalNights",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
]

CATEGORICAL_FEATURES = [
    "ArrivalDateMonth",
    "MarketSegment",
    "DistributionChannel",
    "CustomerType",
    "ReservedRoomType",
    "Meal",
]

BINARY_FEATURES = ["IsRepeatedGuest"]


def build_preprocessor(scale_numeric=True): #law scale numeric b false me4 ha3mel scaling lel data
    """Return a fresh, UNFITTED ColumnTransformer for our 12 features."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("binary", "passthrough", BINARY_FEATURES),
    ])


if __name__ == "__main__":
    print("Numeric:", NUMERIC_FEATURES)
    print("Categorical:", CATEGORICAL_FEATURES)
    print("Binary:", BINARY_FEATURES)
    print("Created an unfitted preprocessor; no split or model fitting performed.")
    preprocessor = build_preprocessor()
    print(preprocessor)
