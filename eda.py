
from pathlib import Path

import pandas as pd


def check_missing_values(df):
    print("\n===== MISSING VALUES =====")
    print(df.isna().sum())

def check_statistics(df):
    print("\n===== STATISTICS =====")
    print(df.describe())


def check_duplicates(df):
    # Print a heading so this section is easy to identify in the console output.
    print("\n===== DUPLICATE ANALYSIS =====")

    # Count rows that match an earlier row across all columns.
    duplicate_count = df.duplicated().sum()

    # Get the total number of rows so the duplicate count can be put in context.
    total_rows = len(df)

    # Report the dataset size and the number of duplicate rows.
    print("Total rows:", total_rows)
    print("Duplicate rows:", duplicate_count)

    # Calculate and display duplicates as a percentage of all rows.
    print("Duplicate percentage:",
          round(duplicate_count / total_rows * 100, 2), "%")

    # Select every row belonging to a duplicate group, including the first copy.
    duplicates = df[df.duplicated(keep=False)]

    # Show up to the first ten duplicate rows as examples.
    print("\nSample duplicate records:")
    print(duplicates.head(10))

    # Return the duplicate count so callers can use it in further analysis.
    return duplicate_count

def analyze_duplicate_patterns(df):

    print("\n===== DUPLICATE PATTERNS =====")

    # Group all identical rows and count repetitions
    duplicates = (
        df.groupby(
            list(df.columns),
            dropna=False
        )
        .size()
        .reset_index(name="Frequency")
    )

    # Keep only repeated combinations
    duplicates = duplicates[
        duplicates["Frequency"] > 1
    ]

    duplicates = duplicates.sort_values(
        by="Frequency",
        ascending=False
    )

    print("\nNumber of repeated booking patterns:", len(duplicates))

    print("\nTop 10 repeated patterns:")
    print(
        duplicates[
            [
                "LeadTime",
                "Adults",
                "Children",
                "ADR",
                "IsCanceled",
                "Frequency"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nMaximum repetition:", duplicates["Frequency"].max())

    return duplicates
def check_invalid_values(df):

    print("\n===== INVALID VALUES ANALYSIS =====")

    # Calculate total guests
    total_guests = (
        df["Adults"] +
        df["Children"].fillna(0) +
        df["Babies"]
    )

    # Calculate total nights
    total_nights = (
        df["StaysInWeekendNights"] +
        df["StaysInWeekNights"]
    )

    # Define suspicious conditions
    conditions = {
        "Negative ADR": df["ADR"] < 0,
        "Negative LeadTime": df["LeadTime"] < 0,
        "Negative Adults": df["Adults"] < 0,
        "Negative Children": df["Children"] < 0,
        "Negative Babies": df["Babies"] < 0,
        "Zero Total Guests": total_guests == 0,
        "Zero Total Nights": total_nights == 0,
        "Invalid IsCanceled": ~df["IsCanceled"].isin([0, 1])
    }

    # Count each problem and display affected rows
    for problem, condition in conditions.items():

        invalid_rows = df[condition]

        print(f"\n===== {problem} =====")
        print("Number of records:", len(invalid_rows))

        if len(invalid_rows) > 0:
            print("\nSample affected records:")
            print(invalid_rows.head(10).to_string())

    print("\n===== NUMERICAL RANGES =====")
    print("ADR:", df["ADR"].min(), "to", df["ADR"].max())
    print("LeadTime:", df["LeadTime"].min(), "to", df["LeadTime"].max())

def check_data_types(df):

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== OBJECT COLUMNS =====")
    print(df.select_dtypes(include="object").columns.tolist())

    print("\n===== NUMERICAL COLUMNS =====")
    print(df.select_dtypes(include="number").columns.tolist())

    print("\n===== UNIQUE VALUES =====")
    print(df.nunique())
    #benshoof el unique values bta3t kol column 
def check_categorical_values(df):

    columns = [
        "Meal",
        "Country",
        "MarketSegment",
        "DistributionChannel",
        "ReservedRoomType",
        "AssignedRoomType",
        "DepositType",
        "Agent",
        "Company",
        "CustomerType"
    ]

    print("\n===== CATEGORICAL ANALYSIS =====")

    for col in columns:
        print(f"\n===== {col} =====")

        print("Unique values:", df[col].nunique())

        print("\nMost frequent categories:")
        print(df[col].value_counts(dropna=False).head(10))


def validate_numerical_counts(df):

    print("\n===== NUMERICAL COUNT VALIDATION =====")

    columns = [
        "Adults",
        "Children",
        "Babies",
        "StaysInWeekendNights",
        "StaysInWeekNights",
        "PreviousCancellations",
        "PreviousBookingsNotCanceled",
        "BookingChanges",
        "DaysInWaitingList",
        "RequiredCarParkingSpaces",
        "TotalOfSpecialRequests"
    ]

    for col in columns:

        values = df[col].dropna()

        negative = (values < 0).sum()
        fractional = (values % 1 != 0).sum()
        missing = df[col].isna().sum()

        print(f"\nColumn: {col}")
        print("Negative values:", negative)
        print("Fractional values:", fractional)
        print("Missing values:", missing)

    print("\n===== DATE VALIDATION =====")

    parsed_dates = pd.to_datetime(
        df["ReservationStatusDate"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    print("Invalid dates:",
          (df["ReservationStatusDate"].notna()
           & parsed_dates.isna()).sum())


def analyze_zero_nights(df):

    print("\n===== ZERO-NIGHT ANALYSIS =====")

    # Calculate total nights
    total_nights = (
        df["StaysInWeekendNights"]
        + df["StaysInWeekNights"]
    )

    # Select zero-night bookings
    zero_nights = df[total_nights == 0]

    print("Total records:", len(df))
    print("Zero-night records:", len(zero_nights))

    print("\n===== RESERVATION STATUS =====")
    print(zero_nights["ReservationStatus"].value_counts())

    print("\n===== CANCELLATION DISTRIBUTION =====")
    print(zero_nights["IsCanceled"].value_counts())

    print("\n===== ADR ANALYSIS =====")
    print("ADR = 0:", (zero_nights["ADR"] == 0).sum())
    print("ADR > 0:", (zero_nights["ADR"] > 0).sum())

    print("\n===== STATUS VS ADR =====")
    print(pd.crosstab(
        zero_nights["ReservationStatus"],
        zero_nights["ADR"].eq(0).map({
            True: "ADR = 0",
            False: "ADR != 0"
        })
    ))

    print("\n===== SAMPLE RECORDS =====")
    columns = [
        "Adults",
        "Children",
        "StaysInWeekendNights",
        "StaysInWeekNights",
        "ADR",
        "IsCanceled",
        "ReservationStatus"
    ]

    print(zero_nights[columns].head(15).to_string())

    return zero_nights
#neshoof el outliers bta3t el adr 
def investigate_adr(df):

    print("\n===== ADR OUTLIER INVESTIGATION =====")

    # Show ADR percentiles
    print("\nADR Percentiles:")
    print(df["ADR"].quantile(
        [0.50, 0.90, 0.95, 0.99, 0.999, 1.00]
    ))

    # Count extreme ADR values
    print("\nExtreme ADR Counts:")
    for threshold in [300, 500, 1000, 2000]:
        print(f"ADR > {threshold}:",
              (df["ADR"] > threshold).sum())

    # Examine the highest prices
    columns = [
        "ADR",
        "Adults",
        "Children",
        "LeadTime",
        "ReservedRoomType",
        "MarketSegment",
        "StaysInWeekendNights",
        "StaysInWeekNights",
        "IsCanceled"
    ]

    print("\nTop 15 Highest ADR Records:")
    print(
        df.nlargest(15, "ADR")[columns]
        .to_string(index=True)
    )
def verify_cleaned_data(df):
    print("\n===== FINAL DATA VERIFICATION =====")

    print("Total rows:", len(df))
    print("Missing values:\n", df.isna().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Maximum ADR:", df["ADR"].max())

    total_guests = df["Adults"] + df["Children"] + df["Babies"]
    print("Zero-guest bookings:", (total_guests == 0).sum())

    total_nights = (
        df["StaysInWeekendNights"]
        + df["StaysInWeekNights"]
    )
    print("Zero-night bookings:", (total_nights == 0).sum())

    print("\nCategorical placeholders:")
    for column in ["Agent", "Company", "MarketSegment", "DistributionChannel"]:
        print(column, df[column].astype("string").str.strip().isin(
            ["NULL", "Undefined"]
        ).sum())
def main():
    project_dir = Path(__file__).resolve().parent.parent
    dataset_names = ("H2_cleaned2.csv", "H2_cleaned3.csv")
    analyses = (
        check_missing_values,
        check_duplicates,
        analyze_duplicate_patterns,
        check_invalid_values,
        check_statistics,
        check_data_types,
        check_categorical_values,
        validate_numerical_counts,
        analyze_zero_nights,
        investigate_adr,
        verify_cleaned_data,
    )

    for dataset_name in dataset_names:
        print(f"\n========== {dataset_name} ==========")
        df = pd.read_csv(project_dir / "data" / dataset_name)
        print("Dataset shape:", df.shape)

        for analysis in analyses:
            analysis(df)


if __name__ == "__main__":
    main()