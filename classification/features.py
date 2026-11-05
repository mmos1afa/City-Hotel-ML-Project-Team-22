"""Provisional cancellation-prediction inputs for H2 City Hotel.

No data splitting, missing-value imputation, encoding, scaling, or model fitting.
Run: python -m classification.features
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned" / "H2_cleaned2.csv"
TARGET = "IsCanceled"

# Initial shortlist; not a claim of learned feature importance.
FEATURES = [
    "LeadTime",
    "TotalGuests",              # Engineered from Adults + Children + Babies
    "TotalNights",              # Engineered from weekend + weekday nights
    "ArrivalDateMonth",
    "MarketSegment",
    "DistributionChannel",
    "CustomerType",
    "IsRepeatedGuest",          # Check that history is known at prediction time
    "PreviousCancellations",    # Audit timing and duplicated records
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "Meal",
]


def select_classification_features(df: pd.DataFrame):
    """Return X and y without altering the source DataFrame."""
    needed = (set(FEATURES) - {"TotalGuests", "TotalNights"}) | {
        TARGET, "Adults", "Children", "Babies",
        "StaysInWeekendNights", "StaysInWeekNights",
    }
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing source columns: {missing}")

    work = df.copy()
    work["TotalGuests"] = work["Adults"] + work["Children"] + work["Babies"]
    work["TotalNights"] = work["StaysInWeekendNights"] + work["StaysInWeekNights"]

    # Some categories in H2_cleaned2.csv contain trailing spaces (e.g., 'A   ', 'BB   ').
    # Strip them so training and future inputs use the same category names.
    for column in [
        "ArrivalDateMonth", "MarketSegment", "DistributionChannel",
        "CustomerType", "ReservedRoomType", "Meal",
    ]:
        work[column] = work[column].astype("string").str.strip().fillna("Unknown").astype(object)

    return work[FEATURES].copy(), work[TARGET].copy()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    X, y = select_classification_features(df)
    print(f"Dataset: {df.shape[0]:,} bookings, {df.shape[1]} original columns")
    print(f"Selected inputs X: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Target y: {y.name}, values = {sorted(y.dropna().unique().tolist())}")
    print("\nSelected features:")
    for feature in X.columns:
        print(" -", feature)
    print("\nNo split or model fitting performed.")
