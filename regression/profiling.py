
import pandas as pd
from pathlib import Path


def profile_adr(df, dataset_name):

    print(f"\n========== {dataset_name} ==========")

    # 1. Basic information
    print("\nTotal bookings:", len(df))

    print("\nADR Statistics:")
    print(df["ADR"].describe())

    # 2. Zero ADR bookings
    print("\nBookings with ADR = 0:",
          (df["ADR"] == 0).sum())

    # 3. ADR by room type
    print("\nAverage ADR by ReservedRoomType:")
    print(
        df.groupby("ReservedRoomType")["ADR"]
        .mean()
        .sort_values(ascending=False)
    )

    # 4. ADR by arrival month
    print("\nAverage ADR by ArrivalDateMonth:")
    print(
        df.groupby("ArrivalDateMonth")["ADR"]
        .mean()
    )

    # 5. Total stay duration
    total_nights = (
        df["StaysInWeekendNights"]
        + df["StaysInWeekNights"]
    )

    print("\nZero-night bookings:", (total_nights == 0).sum())


if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parent.parent

    datasets = [
        ("With duplicates", "H2_cleaned2.csv"),
        ("Without duplicates", "H2_cleaned3.csv")
    ]

    for name, filename in datasets:

        path = project_dir / "data" / "cleaned" / filename

        df = pd.read_csv(path)

        profile_adr(df, name)
