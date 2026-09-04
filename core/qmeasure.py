"""Clustering validation metric QMeasure (Lee, Han, & Whang, SIGMOD 2007)."""

from typing import Optional
import numpy as np
from core.distance import pairwise_segment_distances


def compute_qmeasure(
    segments: np.ndarray,
    labels: np.ndarray,
    distance_matrix: Optional[np.ndarray] = None,
) -> float:
    """Compute the QMeasure clustering quality metric defined in Section 5.1 of Lee et al., 2007.

    QMeasure = sum_{i=1}^{num_{clus}} [ 1/(2*|C_i|) * sum_{x in C_i} sum_{y in C_i} dist(x, y)^2 ]
             + 1/(2*|N|) * sum_{w in N} sum_{z in N} dist(w, z)^2

    Parameters
    ----------
    segments : ndarray of shape (M, 2, 2)
        Partitioned line segments.
    labels : ndarray of shape (M,)
        Cluster assignments (-1 denotes noise).
    distance_matrix : ndarray of shape (M, M), optional
        Precomputed line segment distance matrix.

    Returns
    -------
    qmeasure : float
        Total quality score (lower indicates more compact clusters and cohesive noise).
    """
    M = len(labels)
    if M == 0:
        return 0.0

    if distance_matrix is None:
        distance_matrix = pairwise_segment_distances(segments)

    dist_sq = distance_matrix**2
    unique_clusters = np.unique(labels[labels >= 0])
    total_q = 0.0

    # 1. Intra-cluster dispersion across each cluster C_i
    for c in unique_clusters:
        mask = labels == c
        c_size = int(np.sum(mask))
        if c_size > 0:
            c_dist_sq = dist_sq[np.ix_(mask, mask)]
            # 1 / (2 * |C_i|) * sum_{x, y in C_i} dist(x, y)^2
            total_q += float(np.sum(c_dist_sq)) / (2.0 * c_size)

    # 2. Noise dispersion across all noise segments N
    noise_mask = labels == -1
    noise_size = int(np.sum(noise_mask))
    if noise_size > 0:
        noise_dist_sq = dist_sq[np.ix_(noise_mask, noise_mask)]
        total_q += float(np.sum(noise_dist_sq)) / (2.0 * noise_size)

    return float(total_q)

