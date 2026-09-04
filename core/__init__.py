"""Core geometric primitives, distance functions, entropy estimation, and quality metrics."""

from .distance import (
    iterative_segment_distance,
    vectorized_segment_distance,
    pairwise_segment_distances,
)
from .representative import (
    generate_representative_trajectory,
)
from .entropy import (
    compute_entropy_at_eps,
    compute_entropy_sweep,
)
from .qmeasure import (
    compute_qmeasure,
)

__all__ = [
    "iterative_segment_distance",
    "vectorized_segment_distance",
    "pairwise_segment_distances",
    "generate_representative_trajectory",
    "compute_entropy_at_eps",
    "compute_entropy_sweep",
    "compute_qmeasure",
]

