"""Fast-TRACLUS algorithm (González Delgado et al., 2026)."""

from .vectorized_mdl import (
    vectorized_mdl_partition,
    partition_trajectories_fast,
)
from .distance_matrix import (
    compute_distance_matrix,
)
from .modular_clustering import (
    FastTRACLUS,
    modular_cluster_segments,
)

__all__ = [
    "vectorized_mdl_partition",
    "partition_trajectories_fast",
    "compute_distance_matrix",
    "FastTRACLUS",
    "modular_cluster_segments",
]
