"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def compute_cluster_metrics(X: Any, labels: Any, algorithm: Any, sample_size: int | None, seed: int) -> dict[str, Any]:
    """Return internal metrics for the fitted algorithm, recording noise and invalid partitions."""
    pass


def summarize_cluster_sizes(labels: Any) -> pd.DataFrame:
    """Return per-cluster counts and percentages, including noise."""
    pass


def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Return baseline and tuned comparisons on consistent benchmark rows."""
    pass
