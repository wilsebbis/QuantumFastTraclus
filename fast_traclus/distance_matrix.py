"""Fast-TRACLUS broadcasted N x N distance tensor computation."""

from typing import Optional, Tuple
import numpy as np
from core.distance import pairwise_segment_distances


def compute_distance_matrix(
    segments: np.ndarray,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    return_components: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the full symmetric N x N pairwise line segment distance tensor.

    Broadcasts operations across multidimensional NumPy arrays to evaluate
    all segment pairs directly without nested Python loops.

    Parameters
    ----------
    segments : ndarray of shape (N, 2, 2) or (N, 4)
        Trajectory segments.
    weights : tuple of (w_perp, w_par, w_theta), default=(1.0, 1.0, 1.0)
        Weights for perpendicular, parallel, and angle distances.
    return_components : bool, default=False
        If True, returns (D_perp, D_par, D_theta).

    Returns
    -------
    dist_matrix : ndarray of shape (N, N)
        Symmetric pairwise distance matrix with zero diagonal.
    """
    return pairwise_segment_distances(
        segments1=segments,
        segments2=None,
        weights=weights,
        return_components=return_components,
    )
