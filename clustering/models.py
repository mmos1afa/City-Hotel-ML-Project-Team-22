"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.mixture import GaussianMixture


def make_kmeans(config: dict[str, Any]) -> KMeans:
    """Return an unfitted K-Means model with the supplied settings."""
    pass


def fit_kmeans(X_processed: Any, config: dict[str, Any]) -> tuple[KMeans, np.ndarray]:
    """Return fitted K-Means and labels aligned with the processed rows."""
    pass


def make_dbscan(config: dict[str, Any]) -> DBSCAN:
    """Return an unfitted DBSCAN model with the supplied settings."""
    pass


def fit_dbscan(X: Any, config: dict[str, Any]) -> tuple[DBSCAN, np.ndarray]:
    """Return fitted DBSCAN and aligned labels, including noise labeled minus one."""
    pass


def make_gmm(config: dict[str, Any]) -> GaussianMixture:
    """Return an unfitted Gaussian mixture model with the supplied settings."""
    pass


def fit_gmm(X: Any, config: dict[str, Any]) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Return fitted GMM, aligned labels, and membership probabilities."""
    pass
