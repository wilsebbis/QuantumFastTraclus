"""Core geometric primitives, distance functions, and trajectory generation utilities."""

from .distance import (
    iterative_segment_distance,
    vectorized_segment_distance,
    pairwise_segment_distances,
)
from .representative import (
    generate_representative_trajectory,
)

__all__ = [
    "iterative_segment_distance",
    "vectorized_segment_distance",
    "pairwise_segment_distances",
    "generate_representative_trajectory",
]
