
from pathlib import Path

import matplotlib.pyplot as plt
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
    project_dir = Path(__file__).resolve().parent
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
        df = pd.read_csv(project_dir / "data" / "cleaned" / dataset_name)
        print("Dataset shape:", df.shape)

        for analysis in analyses:
            analysis(df)
    # Generate EDA graphs for the ORIGINAL dataset
    original_path = project_dir / "data" / "precleaned" / "H2.csv"
    output_dir = project_dir / "outputs" / "eda"

    original_df = pd.read_csv(
        original_path,
        keep_default_na=False,
        na_values=[""]
    )

    create_eda_graphs(original_df, output_dir)

def create_eda_graphs(df, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    numeric_columns = [
        "ADR",
        "LeadTime",
        "IsCanceled",
        "StaysInWeekendNights",
        "StaysInWeekNights",
        "Adults",
        "Children",
        "Babies",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 1. Missing values, including literal NULL placeholders
    missing = df.isna().sum()

    for col in ["Agent", "Company"]:
        missing[col] += (
            df[col].astype("string").str.strip().eq("NULL").sum()
        )

    missing = missing[missing > 0].sort_values()

    plt.figure(figsize=(10, 5))
    missing.plot(kind="barh")
    plt.title("Missing Values in Original H2 Dataset")
    plt.xlabel("Number of Missing Values")
    plt.tight_layout()
    plt.savefig(output_dir / "01_missing_values.png")
    plt.close()

    # 2. ADR histogram
    plt.figure(figsize=(9, 5))
    plt.hist(df["ADR"].dropna(), bins=60, edgecolor="black")
    plt.title("ADR Distribution")
    plt.xlabel("Average Daily Rate")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig(output_dir / "02_adr_histogram.png")
    plt.close()

    # 3. ADR boxplot
    plt.figure(figsize=(9, 4))
    plt.boxplot(df["ADR"].dropna(), vert=False)
    plt.title("ADR Outlier Detection")
    plt.xlabel("ADR")
    plt.tight_layout()
    plt.savefig(output_dir / "03_adr_boxplot.png")
    plt.close()

    # 4. LeadTime histogram
    plt.figure(figsize=(9, 5))
    plt.hist(df["LeadTime"], bins=50, edgecolor="black")
    plt.title("Lead Time Distribution")
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Bookings")
    plt.tight_layout()
    plt.savefig(output_dir / "04_leadtime_histogram.png")
    plt.close()

    # 5. LeadTime boxplot
    plt.figure(figsize=(9, 4))
    plt.boxplot(df["LeadTime"], vert=False)
    plt.title("Lead Time Outliers")
    plt.xlabel("Lead Time (Days)")
    plt.tight_layout()
    plt.savefig(output_dir / "05_leadtime_boxplot.png")
    plt.close()

    # 6. Numerical correlation heatmap
    corr = df.select_dtypes(include="number").corr()

    plt.figure(figsize=(14, 11))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90,
        fontsize=8
    )
    plt.yticks(
        range(len(corr.columns)),
        corr.columns,
        fontsize=8
    )
    plt.title("Numerical Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "06_correlation_heatmap.png")
    plt.close()

    # 7. Cancellation distribution
    df["IsCanceled"].value_counts().sort_index().plot(
        kind="bar", figsize=(7, 5)
    )
    plt.title("Cancellation Distribution")
    plt.xlabel("IsCanceled (0 = No, 1 = Yes)")
    plt.ylabel("Bookings")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "07_cancellation.png")
    plt.close()

    # 8. Arrival month distribution
    months = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    df["ArrivalDateMonth"].value_counts().reindex(months).plot(
        kind="bar", figsize=(11, 5)
    )
    plt.title("Bookings by Arrival Month")
    plt.xlabel("Month")
    plt.ylabel("Bookings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "08_arrival_month.png")
    plt.close()

    # 9. Total nights distribution
    total_nights = (
        df["StaysInWeekendNights"]
        + df["StaysInWeekNights"]
    )

    plt.figure(figsize=(9, 5))
    plt.hist(total_nights, bins=40, edgecolor="black")
    plt.title("Total Nights Distribution")
    plt.xlabel("Nights")
    plt.ylabel("Bookings")
    plt.tight_layout()
    plt.savefig(output_dir / "09_total_nights.png")
    plt.close()

    # 10. Total guests distribution
    total_guests = (
        df["Adults"]
        + df["Children"].fillna(0)
        + df["Babies"]
    )

    plt.figure(figsize=(9, 5))
    plt.hist(total_guests, bins=20, edgecolor="black")
    plt.title("Total Guests Distribution")
    plt.xlabel("Guests")
    plt.ylabel("Bookings")
    plt.tight_layout()
    plt.savefig(output_dir / "10_total_guests.png")
    plt.close()

    print("All 10 EDA graphs generated successfully!")
    print("Saved in:", output_dir)


if __name__ == "__main__":
    main()