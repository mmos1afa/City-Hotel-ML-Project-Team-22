"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def validate_cleaned_schema(df: pd.DataFrame) -> None:
    """Validate required columns and types in the cleaned input; report violations."""
    pass


def audit_clustering_input(df: pd.DataFrame) -> dict[str, Any]:
    """Return row counts, missingness, categories, and duplicate-policy audit findings."""
    pass


def get_feature_config() -> dict[str, Any]:
    """Return the core feature groups and leakage exclusions."""
    pass


def select_clustering_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return booking-time features and metadata, preserving row identifiers."""
    pass
