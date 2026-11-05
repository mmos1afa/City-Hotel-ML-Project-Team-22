"""Clustering model creation and training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from sklearn.cluster import KMeans

BASELINE_KMEANS_CONFIG = {
    "n_clusters": 7,  # Our first run will create three booking groups.
    "init": "k-means++",  # Use smart starting cluster centers.
    "n_init": 10,  # Try ten starts and keep the best result.
    "max_iter": 300,  # Allow up to 300 steps for each start.
    "random_state": 42,  # Keep the baseline result repeatable.
}

if TYPE_CHECKING:
    import numpy as np
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.mixture import GaussianMixture


def make_kmeans(
    config: dict[str, Any],
) -> KMeans:  # creates k-means model but does not train
    """Create a K-Means model using the chosen settings, without training it."""
    # config.get(..., default) uses the default value if that setting is missing.
    return KMeans(
        n_clusters=config.get("n_clusters", 3),  # Start with 3 groups.
        init=config.get("init", "k-means++"),  # How starting centers are chosen.
        n_init=config.get("n_init", 10),  # Try 10 different starting positions.
        max_iter=config.get("max_iter", 300),  # Maximum training steps per try.
        random_state=config.get("random_state", 42),  # Same result on reruns.
    )


def fit_kmeans(  # trains k-means model and returns the trained model and the labels for each row in the processed data
    X_processed: Any,
    config: dict[str, Any],
) -> tuple[KMeans, np.ndarray]:
    """Train K-Means on processed bookings and return the model and labels."""
    # Create the K-Means model using the selected settings.
    model = make_kmeans(config)

    # Train on every processed booking. Each returned label is its cluster number for each row.
    labels = model.fit_predict(X_processed)

    # model stores the learned cluster centers; labels map rows to clusters.
    return model, labels


def make_dbscan(config: dict[str, Any]) -> DBSCAN:
    """Return an unfitted DBSCAN model with the supplied settings."""
    pass


def fit_dbscan(X: Any, config: dict[str, Any]) -> tuple[DBSCAN, np.ndarray]:
    """Return fitted DBSCAN and aligned labels, including noise labeled minus one."""
    pass


def make_gmm(config: dict[str, Any]) -> GaussianMixture:
    """Return an unfitted Gaussian mixture model with the supplied settings."""
    pass


def fit_gmm(
    X: Any, config: dict[str, Any]
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Return fitted GMM, aligned labels, and membership probabilities."""
    pass
