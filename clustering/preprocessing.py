"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
    import pandas as pd
    from sklearn.compose import ColumnTransformer


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], config: dict[str, Any]) -> ColumnTransformer:
    """Return an unfitted transformer configured for selected numeric and categorical columns."""
    pass


def fit_transform_features(X_raw: pd.DataFrame, preprocessor: ColumnTransformer) -> tuple[Any, ColumnTransformer]:
    """Return the processed matrix and fitted transformer with stable feature order."""
    pass


def transform_features(X_new: pd.DataFrame, fitted_preprocessor: ColumnTransformer) -> Any:
    """Return new rows transformed using existing fitted preprocessing state."""
    pass


def save_clustering_artifacts(artifacts: dict[str, Any], artifact_dir: str | Path) -> None:
    """Save supplied fitted models, preprocessing state, and metadata locally."""
    pass


def load_clustering_artifacts(artifact_dir: str | Path) -> dict[str, Any]:
    """Return fitted models, preprocessing state, and metadata from local artifacts."""
    pass
