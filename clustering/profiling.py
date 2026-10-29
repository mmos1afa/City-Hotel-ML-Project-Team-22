"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def attach_cluster_labels(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return unscaled rows with labels aligned by stable row identifiers."""
    pass


def build_cluster_profiles(labeled_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return numeric summaries and categorical distributions by cluster."""
    pass


def describe_cluster_differences(profiles: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return evidence-based descriptions of differences in observed profiles."""
    pass
