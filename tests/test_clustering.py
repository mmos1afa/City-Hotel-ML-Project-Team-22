"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations


def test_no_leakage() -> None:
    """Future check for exclusion of outcome and post-booking features."""
    pass


def test_feature_matrix() -> None:
    """Future check for finite features and consistent column order."""
    pass


def test_label_alignment() -> None:
    """Future check for one aligned label per retained booking."""
    pass


def test_reproducibility() -> None:
    """Future check for reproducibility under the same settings and seed."""
    pass
