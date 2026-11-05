import pandas as pd

NUMERIC_FEATURES = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
]

BINARY_FEATURES = ["IsRepeatedGuest"]

CATEGORICAL_FEATURES = [
    "Meal",
    "ReservedRoomType",
    "MarketSegment",
    "DistributionChannel",
]

CLUSTERING_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def get_feature_config() -> dict:
    """Return the numerical, binary, and categorical features used for clustering."""
    return {
        "numeric": NUMERIC_FEATURES.copy(),
        "binary": BINARY_FEATURES.copy(),
        "categorical": CATEGORICAL_FEATURES.copy(),
    }


def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the selected booking features from the cleaned dataset."""
    return df.loc[:, CLUSTERING_FEATURES].copy()
