"""Describe the original booking data inside each cluster."""

from __future__ import annotations

from typing import Any

import pandas as pd

from clustering.features import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def attach_cluster_labels(df: pd.DataFrame, labels: Any) -> pd.DataFrame:
    """Add one K-Means label to each original, unscaled booking row."""
    if len(df) != len(labels):
        raise ValueError("The number of labels must match the number of booking rows.")

    # Copy the readable original data so profiling never changes the input DataFrame.
    labeled_df = df.copy()

    # list(labels) assigns labels by row position: first label to first booking, etc.
    labeled_df["cluster"] = list(labels)
    return labeled_df


def build_cluster_profiles(labeled_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return readable size, numeric, and category summaries for each cluster."""
    if "cluster" not in labeled_df.columns:
        raise ValueError("Add cluster labels before building profiles.")

    # Count bookings in each cluster and calculate each cluster's dataset share.
    cluster_sizes = labeled_df.groupby("cluster").size().rename("count").reset_index()
    cluster_sizes["percentage"] = cluster_sizes["count"] / len(labeled_df) * 100

    # Show median booking values. Medians are less affected by unusually large values.
    numeric_medians = (
        labeled_df.groupby("cluster")[NUMERIC_FEATURES].median().reset_index()
    )

    # IsRepeatedGuest is 0 or 1, so its mean becomes the percentage of repeated guests.
    repeated_guest_percentage = (
        labeled_df.groupby("cluster")[BINARY_FEATURES].mean().mul(100).reset_index()
    )

    profiles = {
        "cluster_sizes": cluster_sizes,
        "numeric_medians": numeric_medians,
        "repeated_guest_percentage": repeated_guest_percentage,
    }

    # For every category, show the percentage of each cluster using each value.
    for column in CATEGORICAL_FEATURES:
        category_percentages = (
            labeled_df.groupby("cluster")[column]
            .value_counts(normalize=True)
            .mul(100)
            .rename("percentage")
            .reset_index()
        )
        profiles[f"{column}_percentages"] = category_percentages

    return profiles


def describe_cluster_differences(profiles: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return short evidence statements for each cluster; names stay human-chosen."""
    sizes = profiles["cluster_sizes"].set_index("cluster")
    medians = profiles["numeric_medians"].set_index("cluster")
    descriptions = {}

    for cluster in sizes.index:
        row = medians.loc[cluster]
        total_nights = row["StaysInWeekendNights"] + row["StaysInWeekNights"]

        # This reports facts only. Give the cluster a business name after reviewing all profiles.
        descriptions[str(cluster)] = (
            f"Cluster {cluster}: {int(sizes.loc[cluster, 'count'])} bookings "
            f"({sizes.loc[cluster, 'percentage']:.2f}%). "
            f"Median lead time: {row['LeadTime']:.1f} days; "
            f"median stay: {total_nights:.1f} nights; "
            f"median adults: {row['Adults']:.1f}; "
            f"median children: {row['Children']:.1f}."
        )

    return descriptions
