"""Entropy parameter estimation H(X) and MinLns estimation for TRACLUS (Lee et al., 2007)."""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from core.distance import pairwise_segment_distances


def compute_entropy_at_eps(
    distance_matrix: np.ndarray,
    eps: float,
) -> Tuple[float, float, int]:
    """Compute the entropy H(X) and average neighborhood density at threshold eps.

    H(X) = - sum_{i=1}^n p(x_i) * log2(p(x_i))
    where p(x_i) = |N_eps(x_i)| / sum_{j=1}^n |N_eps(x_j)|

    Parameters
    ----------
    distance_matrix : ndarray of shape (N, N)
        Pairwise distance matrix of line segments.
    eps : float
        Neighborhood radius epsilon.

    Returns
    -------
    entropy : float
        Shannon entropy H(X) in bits.
    avg_density : float
        Average neighborhood size avg_{|N_eps(L)|}.
    estimated_min_lines : int
        Estimated MinLns = floor(avg_{|N_eps(L)|}) + 1.
    """
    N = distance_matrix.shape[0]
    if N == 0:
        return 0.0, 0.0, 1

    # In TRACLUS, N_eps(x_i) includes all segments x_j such that dist(x_i, x_j) <= eps (including itself)
    counts = np.sum(distance_matrix <= eps, axis=1).astype(np.float64)
    total_count = float(np.sum(counts))

    if total_count <= 0:
        return 0.0, 0.0, 1

    probs = counts / total_count
    # Filter out zero probabilities for log2
    nonzero_probs = probs[probs > 0]
    entropy = -float(np.sum(nonzero_probs * np.log2(nonzero_probs)))

    avg_density = float(np.mean(counts))
    estimated_min_lines = max(2, int(math.floor(avg_density)) + 1)

    return entropy, avg_density, estimated_min_lines


def compute_entropy_sweep(
    segments: np.ndarray,
    eps_range: Union[np.ndarray, List[float]],
    distance_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """Execute an entropy sweep across epsilon values to identify the optimal parameter eps*.

    Finds the global minimum of H(X) as proposed in Section 5.1 of Lee et al., 2007.

    Parameters
    ----------
    segments : ndarray of shape (N, 2, 2)
        Extracted line segments.
    eps_range : array_like
        Sequence of candidate epsilon values (e.g. range(1, 61)).
    distance_matrix : ndarray of shape (N, N), optional
        Precomputed distance matrix. Computed if None.

    Returns
    -------
    results : dict
        Contains 'eps_values', 'entropy_values', 'avg_densities', 'min_lines_values',
        'optimal_eps', 'optimal_entropy', 'optimal_avg_density', 'optimal_min_lines'.
    """
    if distance_matrix is None:
        distance_matrix = pairwise_segment_distances(segments)

    eps_list = [float(e) for e in eps_range]
    entropy_list = []
    avg_density_list = []
    min_lines_list = []

    for eps in eps_list:
        h, avg_d, min_lns = compute_entropy_at_eps(distance_matrix, eps)
        entropy_list.append(h)
        avg_density_list.append(avg_d)
        min_lines_list.append(min_lns)

    entropy_arr = np.asarray(entropy_list, dtype=np.float64)
    avg_density_arr = np.asarray(avg_density_list, dtype=np.float64)
    min_lines_arr = np.asarray(min_lines_list, dtype=int)

    # Find minimum entropy
    opt_idx = int(np.argmin(entropy_arr))

    return {
        "eps_values": np.asarray(eps_list, dtype=np.float64),
        "entropy_values": entropy_arr,
        "avg_densities": avg_density_arr,
        "min_lines_values": min_lines_arr,
        "optimal_eps": eps_list[opt_idx],
        "optimal_entropy": float(entropy_arr[opt_idx]),
        "optimal_avg_density": float(avg_density_arr[opt_idx]),
        "optimal_min_lines": int(min_lines_arr[opt_idx]),
    }

