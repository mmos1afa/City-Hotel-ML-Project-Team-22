"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def tune_kmeans(X: Any, search_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return K-Means search settings, metrics, stability, and runtimes."""
    pass


def rank_experiments(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Return experiments ranked by quality, stability, interpretability, and cost."""
    pass


def estimate_eps_candidates(X: Any, config: dict[str, Any]) -> list[float]:
    """Return candidate DBSCAN epsilon values based on k-distance guidance."""
    pass


def tune_dbscan(X: Any, search_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return DBSCAN search results including noise coverage and runtimes."""
    pass


def tune_gmm(X: Any, search_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return GMM search results including BIC, AIC, internal metrics, and runtimes."""
    pass
