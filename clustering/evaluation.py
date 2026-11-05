"""Internal clustering metrics and booking counts by cluster."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score


def compute_cluster_metrics(
    X: pd.DataFrame,  # processed data after scaling and one-hot encoding
    labels: Any,  # assigned cluster labels for each row in X
    algorithm: Any,  # fitted model
    sample_size: int
    | None = 5000,  # maximum number of samples to use for silhouette score calculation
    seed: int = 42,  # random seed for reproducibility of silhouette score sampling
) -> dict[
    str, float
]:  # returns a dictionary with silhouette, Davies-Bouldin, and inertia scores
    """Return silhouette, Davies-Bouldin, and inertia for fitted K-Means.

    X is the processed DataFrame; labels must match its row order.
    algorithm is the fitted K-Means model for these rows. Only silhouette
    uses a sample. DBSCAN noise handling will be added in its own step.
    """
    number_of_clusters = len(set(labels))  # [0,1,2,0,0,1,3]set-> [0,1,2,3]len->4
    if number_of_clusters < 2 or number_of_clusters >= len(
        labels
    ):  # number of clusters must be at least 2 and less than the number of rows
        raise ValueError(
            "Evaluation needs at least two clusters and fewer clusters than rows."
        )

    silhouette = silhouette_score(  # For each booking: is it close to other bookings in its cluster & is it far from bookings in other clusters?
        X,
        labels,
        sample_size=sample_size,  # uses sample size (max 5000) to speed up calculation for large datasets
        random_state=seed,  # ensures reproducibility of the sample selection
    )  # outputs a score between -1 and 1. 1: strong separation , 0: overlapping clusters, -1: misclassified points
    davies_bouldin = davies_bouldin_score(
        X, labels
    )  # measures how similar clusters are to each other. Lower values indicate better clustering. Higher means clusters overlap.
    inertia = algorithm.inertia_  # is calculated during training. Its the total squared distance of samples to their own cluster center. Lower inertia means better clustering. Decreases as clusters increase

    return {
        "silhouette": float(silhouette),
        "davies_bouldin": float(davies_bouldin),
        "inertia": float(inertia),
    }


def summarize_cluster_sizes(labels: Any) -> pd.DataFrame:
    """Return per-cluster counts and percentages, including noise."""
    # Count how many bookings received each label, then order labels as 0, 1, 2, ...
    counts = pd.Series(labels, name="cluster").value_counts().sort_index()

    # Convert the counts into a normal table with columns named "cluster" and "count".
    summary = counts.rename("count").rename_axis("cluster").reset_index()

    # Divide each cluster count by the total number of bookings to get its percentage.
    summary["percentage"] = summary["count"] / counts.sum() * 100
    return summary


def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn metric dictionaries from several clustering runs into one table."""
    # Each dictionary becomes one row. Its keys become table column names.
    comparison = pd.DataFrame(results)

    # Put the model name first when it was included in each result dictionary.
    if "model" in comparison.columns:
        other_columns = [column for column in comparison.columns if column != "model"]
        comparison = comparison[["model", *other_columns]]

    return comparison
