"""Benchmarking harness and evaluation metrics."""

from .synthetic_corridors import (
    generate_dual_spirals,
    generate_u_arterials,
    trajectories_to_dataframe_format,
)
from .metrics import (
    compute_clustering_metrics,
    compute_interference_contrast,
)

__all__ = [
    "generate_dual_spirals",
    "generate_u_arterials",
    "trajectories_to_dataframe_format",
    "compute_clustering_metrics",
    "compute_interference_contrast",
]
