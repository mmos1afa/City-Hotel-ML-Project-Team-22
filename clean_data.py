from anyio import Path
import pandas as pd
import pandas as pd
import pathlib
def clean_missing_values(df):

    # Create a copy to preserve the original DataFrame
    df = df.copy()

    print("===== BEFORE CLEANING =====")
    print("Missing Children:", df["Children"].isna().sum())
    print("Missing Country:", df["Country"].isna().sum())

    # Replace missing Children values with median
    children_median = df["Children"].median()
    df["Children"] = df["Children"].fillna(children_median)

    # Replace missing Country values with Unknown
    df["Country"] = df["Country"].fillna("Unknown")

    print("\n===== AFTER CLEANING =====")
    print("Missing Children:", df["Children"].isna().sum())
    print("Missing Country:", df["Country"].isna().sum())

    return df
def remove_zero_guest_bookings(df):

    df = df.copy()

    total_guests = (
        df["Adults"] +
        df["Children"].fillna(0) +
        df["Babies"]
    )

    print("Rows before:", len(df))

    # Remove rows with zero total guests
    df = df[total_guests > 0].copy()

    print("Rows after:", len(df))

    return df
def validate_dates(df):
    df = df.copy()

    parsed_dates = pd.to_datetime(
        df["ReservationStatusDate"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    invalid_dates = (
        df["ReservationStatusDate"].notna()
        & parsed_dates.isna()
    )

    print("Invalid reservation dates:", invalid_dates.sum())

    df["ReservationStatusDate"] = parsed_dates

    return df


import pandas as pd
from pathlib import Path


def validate_dates(df):
    df = df.copy()

    parsed_dates = pd.to_datetime(
        df["ReservationStatusDate"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    invalid_dates = (
        df["ReservationStatusDate"].notna()
        & parsed_dates.isna()
    )

    print("Invalid reservation dates:", invalid_dates.sum())

    df["ReservationStatusDate"] = parsed_dates

    return df


#3dlna hena el file el asasy le H2_cleaned2.csv w
if __name__ == "__main__":

    from pathlib import Path
    import pandas as pd

    project_dir = pathlib.Path(__file__).resolve().parent.parent

    # Always start from the original dataset
    raw_path = project_dir / "data" / "H2.csv"

    # New cleaned dataset (duplicates retained)
    clean_path = project_dir / "data" / "H2_cleaned2.csv"

    # 1. Load original data
    df = pd.read_csv(raw_path)

    print("Original rows:", len(df))

    # 2. Clean missing values
    df = clean_missing_values(df)

    # 3. Remove zero-guest bookings
    df = remove_zero_guest_bookings(df)

    # 4. Validate and convert dates
    df = validate_dates(df)

    # 5. Remove extreme ADR = 5400
    df = df[df["ADR"] != 5400].copy()

    # IMPORTANT: Keep all duplicate records
    # Do not use df.drop_duplicates()

    # 6. Save the new cleaned dataset
    df.to_csv(clean_path, index=False)

    print("\n===== FINAL RESULTS =====")
    print("Remaining rows:", len(df))
    print("Duplicates retained:", df.duplicated().sum())
    print("Missing Children:", df["Children"].isna().sum())
    print("Missing Country:", df["Country"].isna().sum())
    print("Maximum ADR:", df["ADR"].max())

    print("\nSaved to:", clean_path)
 #3mlnaa hena remove duplicates w save le file gdida
if __name__ == "__main__":

    from pathlib import Path
    import pandas as pd

    project_dir = pathlib.Path(__file__).resolve().parent.parent

    # Start from the cleaned file that still has duplicates
    input_path = project_dir / "data" / "H2_cleaned2.csv"

    # Save the deduplicated version in a new file
    output_path = project_dir / "data" / "H2_cleaned3.csv"

    # Load dataset
    df = pd.read_csv(input_path)

    print("Rows before duplicate removal:", len(df))
    print("Duplicates before:", df.duplicated().sum())

    # Remove exact duplicate rows and keep only one copy
    df = df.drop_duplicates().copy()

    print("Rows after duplicate removal:", len(df))
    print("Duplicates after:", df.duplicated().sum())

    # Save new file
    df.to_csv(output_path, index=False)

    print("\nSaved to:", output_path)
## two mains mohmen awel wahda le H3_cleaned3.csv w el tani le H2_cleaned2.csv
if __name__ == "__main__":

    from pathlib import Path
    import pandas as pd

    project_dir = pathlib.Path(__file__).resolve().parent.parent

    input_path = project_dir / "data" / "H2_cleaned3.csv"
    output_path = project_dir / "data" / "H2_cleaned3.csv"

    df = pd.read_csv(input_path, keep_default_na=False)

    print("===== BEFORE CATEGORICAL CLEANING =====")
    print("Agent NULL:", (df["Agent"].astype(str).str.strip() == "NULL").sum())
    print("Company NULL:", (df["Company"].astype(str).str.strip() == "NULL").sum())

    # Replace placeholders without deleting bookings
    for column in ["Agent", "Company"]:
        df[column] = df[column].astype(str).str.strip()
        df[column] = df[column].replace("NULL", "Unknown")

    print("\n===== AFTER CATEGORICAL CLEANING =====")
    print("Agent Unknown:", (df["Agent"] == "Unknown").sum())
    print("Company Unknown:", (df["Company"] == "Unknown").sum())
    print("Remaining duplicates:", df.duplicated().sum())
    print("Total rows:", len(df))

    df.to_csv(output_path, index=False)
    print("\nSaved to:", output_path)
if __name__ == "__main__":

    from pathlib import Path
    import pandas as pd

    project_dir = pathlib.Path(__file__).resolve().parent.parent

    # Read and update the same CSV file
    file_path = project_dir / "data" / "H2_cleaned2.csv"

    df = pd.read_csv(file_path, keep_default_na=False)

    print("===== BEFORE CLEANING =====")
    print("Agent NULL:", (df["Agent"].astype(str).str.strip() == "NULL").sum())
    print("Company NULL:", (df["Company"].astype(str).str.strip() == "NULL").sum())

    # Replace NULL with Unknown
    for column in ["Agent", "Company"]:
        df[column] = df[column].astype(str).str.strip()
        df[column] = df[column].replace("NULL", "Unknown")

    print("\n===== AFTER CLEANING =====")
    print("Agent Unknown:", (df["Agent"] == "Unknown").sum())
    print("Company Unknown:", (df["Company"] == "Unknown").sum())
    print("Total rows:", len(df))
    print("Duplicates retained:", df.duplicated().sum())

    # Save changes directly to H2_cleaned2.csv
    df.to_csv(file_path, index=False)

    print("\nUpdated successfully:", file_path)

