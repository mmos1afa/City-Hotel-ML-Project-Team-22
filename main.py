"""Run the baseline K-Means clustering workflow."""

from pathlib import Path
from typing import Any

import pandas as pd

from clustering.evaluation import (
    compare_models,
    compute_cluster_metrics,
    summarize_cluster_sizes,
)
from clustering.features import select_clustering_features
from clustering.models import BASELINE_KMEANS_CONFIG, fit_kmeans
from clustering.preprocessing import fit_transform_features
from clustering.profiling import (
    attach_cluster_labels,
    build_cluster_profiles,
    describe_cluster_differences,
)


DATA_PATH = Path("data/cleaned/H2_cleaned3.csv")
SELECTED_K = 8


def run_clustering_baseline(data_path: Path = DATA_PATH) -> dict[str, Any]:
    """Train and evaluate the baseline K-Means model on a cleaned CSV file."""
    # 1. Load the cleaned hotel bookings.
    df = pd.read_csv(data_path)

    # 2. Keep only the 11 features chosen for clustering.
    X_raw = select_clustering_features(df)

    # 3. Scale numeric columns and encode categorical columns.
    preprocessor, X_processed = fit_transform_features(X_raw)

    # 4. Use the selected K=8 configuration and train K-Means.
    config = {**BASELINE_KMEANS_CONFIG, "n_clusters": SELECTED_K}
    model, labels = fit_kmeans(X_processed, config)

    # 5. Measure cluster quality and count bookings in each cluster.
    metrics = compute_cluster_metrics(X_processed, labels, model)
    cluster_sizes = summarize_cluster_sizes(labels)

    # 6. Add labels to readable booking data and describe each cluster.
    labeled_df = attach_cluster_labels(df, labels)
    profiles = build_cluster_profiles(labeled_df)
    descriptions = describe_cluster_differences(profiles)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "labels": labels,
        "metrics": metrics,
        "cluster_sizes": cluster_sizes,
        "profiles": profiles,
        "descriptions": descriptions,
    }


def run_clustering_comparison(
    data_path: Path = DATA_PATH,
    minimum_k: int = 7,
    maximum_k: int = 8,
) -> pd.DataFrame:
    """Train K-Means for every K in a range and return one comparison table."""
    # Load and preprocess once because every experiment uses the same bookings.
    df = pd.read_csv(data_path)
    X_raw = select_clustering_features(df)
    _, X_processed = fit_transform_features(X_raw)

    results = []
    for k in range(minimum_k, maximum_k + 1):
        # Copy the normal settings, changing only the number of requested clusters.
        config = {**BASELINE_KMEANS_CONFIG, "n_clusters": k}
        model, labels = fit_kmeans(X_processed, config)
        metrics = compute_cluster_metrics(X_processed, labels, model)
        sizes = summarize_cluster_sizes(labels)

        # Store one row of information for this K-Means experiment.
        results.append(
            {
                "model": f"K-Means K={k}",
                "K": k,
                **metrics,
                "smallest_cluster_percentage": float(sizes["percentage"].min()),
                "largest_cluster_percentage": float(sizes["percentage"].max()),
            }
        )

    # Turn all result dictionaries into one readable DataFrame.
    return compare_models(results)


def main() -> None:
    """Run K=8, then print its quality metrics and readable profiles."""
    results = run_clustering_baseline()

    print(f"K-Means metrics (K={SELECTED_K}):")
    print(results["metrics"])
    print("\nCluster sizes:")
    print(results["cluster_sizes"].to_string(index=False))
    print("\nMedian booking values by cluster:")
    print(results["profiles"]["numeric_medians"].to_string(index=False))
    print("\nRepeated-guest percentage by cluster:")
    print(results["profiles"]["repeated_guest_percentage"].to_string(index=False))
    print("\nCluster descriptions:")
    for description in results["descriptions"].values():
        print(description)


if __name__ == "__main__":
    main()
