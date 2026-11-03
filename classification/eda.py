"""Simple, cancellation-focused EDA for the City Hotel H2 dataset.

Run from the project root: python -m classification.eda
No training, splitting, or changes to the input CSV.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned" / "H2_cleaned2.csv"
TABLES = ROOT / "outputs" / "classification" / "tables"
FIGURES = ROOT / "outputs" / "classification" / "figures"

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
LEAD_GROUPS = ["0–7", "8–30", "31–90", "91–180", "181+"]


def summarize(df, columns, name):
    """Count bookings and calculate the percentage canceled in each group."""
    result = (df.groupby(columns, observed=True)["IsCanceled"]
                .agg(bookings="size", canceled="sum").reset_index())
    result["rate_pct"] = (100 * result["canceled"] / result["bookings"]).round(2)
    result.to_csv(TABLES / f"{name}.csv", index=False)
    print(f"\n=== {name.replace('_', ' ').title()} ===")
    print(result.to_string(index=False))
    return result


def bar_chart(table, column, name):
    """Simple cancellation-rate bar chart, with booking counts shown."""
    # Very tiny categories remain in CSV tables, but are hidden in plots.
    table = table[table["bookings"] >= 30].copy()
    fig, ax = plt.subplots(figsize=(max(7, len(table) * 0.9), 4.5))
    labels = table[column].astype(str)
    bars = ax.bar(labels, table["rate_pct"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_title(name.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=35 if len(table) > 5 else 0)
    for bar, rate, count in zip(bars, table["rate_pct"], table["bookings"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2,
                f"{rate:.1f}%\nn={count:,}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=140)
    plt.close(fig)


def line_chart(table, x_col, group_col, groups, x_labels, name):
    """Compare cancellation rates across groups on one plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for group in groups:
        part = table[table[group_col] == group].copy()
        part[x_col] = part[x_col].astype(str)
        rates = part.set_index(x_col)["rate_pct"].reindex(x_labels)
        ax.plot(x_labels, rates, marker="o", label=str(group))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_xlabel(x_col)
    ax.set_title(name.replace("_", " ").title())
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=140)
    plt.close(fig)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, skipinitialspace=True)
    print(f"Analyzing {len(df):,} bookings. Descriptive EDA only: no split or training.")

    # Two new features: total guests and planned total nights.
    df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]

    # Group numeric values so cancellation rates are easy to compare.
    df["LeadTimeGroup"] = pd.cut(df["LeadTime"], [-1, 7, 30, 90, 180, float("inf")], labels=LEAD_GROUPS)
    df["TotalGuestsGroup"] = pd.cut(df["TotalGuests"], [-1, 1, 2, 3, 4, float("inf")], labels=["0–1", "2", "3", "4", "5+"])
    df["TotalNightsGroup"] = pd.cut(df["TotalNights"], [-1, 0, 1, 3, 7, float("inf")], labels=["0", "1", "2–3", "4–7", "8+"])
    df["PreviousCancellationsGroup"] = pd.cut(df["PreviousCancellations"], [-1, 0, 1, float("inf")], labels=["0", "1", "2+"])
    df["PreviousBookingsNotCanceledGroup"] = pd.cut(df["PreviousBookingsNotCanceled"], [-1, 0, 1, float("inf")], labels=["0", "1", "2+"])
    df["ArrivalDateMonth"] = pd.Categorical(df["ArrivalDateMonth"], categories=MONTHS, ordered=True)
    df["MarketSegment"] = df["MarketSegment"].astype(str).str.strip()
    df["GuestType"] = df["IsRepeatedGuest"].map({0: "First-time", 1: "Repeated"})

    # Part 1: One feature versus the cancellation outcome.
    topics = [
        ("LeadTimeGroup", "lead_time"),
        ("TotalGuestsGroup", "total_guests"),
        ("TotalNightsGroup", "total_nights"),
        ("GuestType", "repeated_guest"),
        ("PreviousCancellationsGroup", "previous_cancellations"),
        ("PreviousBookingsNotCanceledGroup", "previous_non_canceled"),
        ("MarketSegment", "market_segment"),
        ("ArrivalDateMonth", "arrival_month"),
        ("DistributionChannel", "distribution_channel"),
        ("CustomerType", "customer_type"),
    ]
    for column, name in topics:
        table = summarize(df, [column], name)
        if name in {"lead_time", "total_guests", "total_nights", "repeated_guest",
                    "previous_cancellations", "market_segment", "arrival_month"}:
            bar_chart(table, column, name)

    # Part 2: Explore interactions among features.
    lead_seg = summarize(df, ["MarketSegment", "LeadTimeGroup"], "lead_time_by_segment")
    major_segments = df["MarketSegment"].value_counts().head(4).index.tolist()
    line_chart(lead_seg, "LeadTimeGroup", "MarketSegment", major_segments,
               LEAD_GROUPS, "lead_time_by_segment")

    # Exactly one previous cancellation: compare first-time and repeat guests.
    one_previous = df[df["PreviousCancellations"] == 1]
    repeated = summarize(one_previous, ["GuestType"], "one_previous_by_repeated")
    bar_chart(repeated, "GuestType", "one_previous_by_repeated")
    summarize(one_previous, ["MarketSegment", "GuestType"], "one_previous_by_segment_and_repeated")

    # Arrival-month patterns must be checked separately for each year.
    month_year = summarize(df, ["ArrivalDateYear", "ArrivalDateMonth"], "arrival_month_by_year")
    line_chart(month_year, "ArrivalDateMonth", "ArrivalDateYear",
               sorted(df["ArrivalDateYear"].unique()), MONTHS, "arrival_month_by_year")

    # Does the Groups segment stay cancellation-prone in every year?
    segment_year = summarize(df, ["MarketSegment", "ArrivalDateYear"], "segment_by_year")
    line_chart(segment_year, "ArrivalDateYear", "MarketSegment", major_segments,
               sorted(df["ArrivalDateYear"].unique()), "segment_by_year")

    print(f"\nDone. Tables: {TABLES}\nGraphs: {FIGURES}")
    print("Input CSV unchanged. No train/test split performed.")


if __name__ == "__main__":
    main()
