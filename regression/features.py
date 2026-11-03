def create_regression_features(df):

    df = df.copy()

    # Total number of nights
    df["TotalNights"] = (
        df["StaysInWeekendNights"]
        + df["StaysInWeekNights"]
    )

    # Total number of guests
    df["TotalGuests"] = (
        df["Adults"]
        + df["Children"]
        + df["Babies"]
    )

    # Identify bookings with children or babies
    df["IsFamilyBooking"] = (
        (df["Children"] + df["Babies"]) > 0
    ).astype(int)

    return df