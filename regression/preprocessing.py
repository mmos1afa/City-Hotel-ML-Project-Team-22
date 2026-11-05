
import pandas as pd
from pathlib import Path

from features import create_regression_features


# Columns that must not be used to predict ADR at booking time
LEAKAGE_COLUMNS = [
    "IsCanceled",
    "ReservationStatus",
    "ReservationStatusDate",
    "AssignedRoomType",
    "BookingChanges",
    "DaysInWaitingList"
]


def prepare_regression_data(df):

    df = df.copy()

    # Create our three engineered features
    df = create_regression_features(df)

    # ADR is our prediction target
    y = df["ADR"]

    # Keep only allowed input features
    X = df.drop(columns=LEAKAGE_COLUMNS + ["ADR"])

    # Combine features and target for saving
    regression_df = X.copy()
    regression_df["ADR"] = y

    return regression_df


if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data" / "cleaned"

    datasets = [
        (
            "H2_cleaned2.csv",
            "H2_regression_with_duplicates.csv"
        ),
        (
            "H2_cleaned3.csv",
            "H2_regression_without_duplicates.csv"
        )
    ]

    for input_name, output_name in datasets:

        # Load cleaned dataset
        df = pd.read_csv(data_dir / input_name)

        # Prepare regression dataset
        regression_df = prepare_regression_data(df)

        # Save to a separate CSV
        output_path = data_dir / output_name
        regression_df.to_csv(output_path, index=False)

        print(f"\n===== {output_name} =====")
        print("Total rows:", len(regression_df))
        print("Total columns:", regression_df.shape[1])
        print("Zero-night bookings:",
              (regression_df["TotalNights"] == 0).sum())

        print("New features:",
              ["TotalNights", "TotalGuests", "IsFamilyBooking"])

        print("WeekendRatio included:",
              "WeekendRatio" in regression_df.columns)

        print("Saved successfully to:", output_path)
