"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure
    from sklearn.decomposition import PCA


def fit_pca_projection(X: Any) -> tuple[np.ndarray, PCA]:
    """Return two-dimensional PCA coordinates and the fitted visualization projection."""
    pass


def plot_clusters_2d(coords: np.ndarray, labels: Any) -> Figure:
    """Return a figure coloring supplied two-dimensional coordinates by cluster label."""
    pass


def plot_metric_comparison(results: list[dict[str, Any]]) -> Figure:
    """Return a figure comparing supplied baseline and tuned clustering metrics."""
    pass
