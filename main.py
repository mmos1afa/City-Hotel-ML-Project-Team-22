"""Scaffold interfaces only; every function is unimplemented."""

from __future__ import annotations

from typing import Any


def run_clustering_baseline(config: dict[str, Any]) -> None:
    """Run the planned cleaned-data baseline with the supplied configuration."""
    pass


def run_clustering_comparison(config: dict[str, Any]) -> None:
    """Run the planned baseline and tuned comparison with the supplied configuration."""
    pass


def main() -> None:
    """Application entry point; no execution logic is implemented."""
    pass


if __name__ == "__main__":
    main()
