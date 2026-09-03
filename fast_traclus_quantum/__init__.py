"""Fast-TRACLUS with Continuous-Time Quantum Walk (CTQW) Graph Community Grouping.

A high-performance trajectory clustering package replacing classical Graph Laplacian
eigendecomposition (O(N^3) diagonalization + k-means) with sparse Continuous-Time
Quantum Walks on the normalized Laplacian.
"""

from .distance import (
    perpendicular_distance,
    parallel_distance,
    angular_distance,
    segment_distance,
    pairwise_segment_distances,
)
from .segmentation import (
    partition_trajectory,
    partition_trajectories,
    mdl_cost,
)
from .graph import (
    build_affinity_matrix,
    build_normalized_laplacian,
    AffinityGraph,
)
from .ctqw import (
    CTQWClusterer,
)
from .pipeline import (
    FastTRACLUSQuantum,
    FastTRACLUSSpectralBaseline,
)
from .evaluation import (
    davies_bouldin_index,
    silhouette_score,
    interference_contrast_ratio,
    benchmark_clustering,
)

__version__ = "0.1.0"

__all__ = [
    "perpendicular_distance",
    "parallel_distance",
    "angular_distance",
    "segment_distance",
    "pairwise_segment_distances",
    "partition_trajectory",
    "partition_trajectories",
    "mdl_cost",
    "build_affinity_matrix",
    "build_normalized_laplacian",
    "AffinityGraph",
    "CTQWClusterer",
    "FastTRACLUSQuantum",
    "FastTRACLUSSpectralBaseline",
    "davies_bouldin_index",
    "silhouette_score",
    "interference_contrast_ratio",
    "benchmark_clustering",
]
