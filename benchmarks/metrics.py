"""Clustering quality metrics and CTQW interference contrast ratio calculation."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    silhouette_score as sk_silhouette,
    davies_bouldin_score as sk_davies_bouldin,
    calinski_harabasz_score as sk_calinski_harabasz,
)


def extract_segment_features(segments: np.ndarray) -> np.ndarray:
    """Extract 5D geometric feature representation for line segments: [mid_x, mid_y, length, cos, sin]."""
    s = segments[:, 0, :]
    e = segments[:, 1, :]
    mid = 0.5 * (s + e)
    diff = e - s
    length = np.linalg.norm(diff, axis=-1, keepdims=True)
    safe_len = np.maximum(length, 1e-12)
    direction = diff / safe_len
    return np.hstack([mid, length, direction])


def compute_interference_contrast(
    K: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute the Constructive-to-Destructive Quantum Interference Contrast Ratio C.

    C = E[K_jk | j, k in same cluster] / E[K_jk | j, k in different clusters]
    (for j != k and valid clusters j, k >= 0).
    """
    N = len(labels)
    if N < 2:
        return 0.0

    valid = labels >= 0
    if np.sum(valid) < 2:
        return 0.0

    # Intra-cluster mask (same cluster, off-diagonal)
    same_cluster = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
    np.fill_diagonal(same_cluster, False)

    # Inter-cluster mask (different valid clusters)
    diff_cluster = (labels[:, None] != labels[None, :]) & (labels[:, None] >= 0) & (labels[None, :] >= 0)

    intra_vals = K[same_cluster]
    inter_vals = K[diff_cluster]

    if len(intra_vals) == 0 or len(inter_vals) == 0:
        return 0.0

    mean_intra = float(np.mean(intra_vals))
    mean_inter = float(np.mean(inter_vals))

    if mean_inter <= 1e-15:
        return float("inf") if mean_intra > 0.0 else 0.0

    return mean_intra / mean_inter


def compute_clustering_metrics(
    segments: np.ndarray,
    labels: np.ndarray,
    distance_matrix: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int]]:
    """Compute internal cluster quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin).

    Ignores noise points labeled -1 during index calculations.
    """
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    n_valid = len(valid_labels)
    unique_clusters = np.unique(valid_labels)
    n_clusters = len(unique_clusters)

    noise_ratio = float(np.mean(labels == -1)) if len(labels) > 0 else 0.0

    metrics = {
        "n_clusters": int(n_clusters),
        "noise_ratio": float(noise_ratio),
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
        "contrast_ratio": float("nan"),
    }

    if n_clusters < 2 or n_valid < 3:
        return metrics

    # 1. Silhouette Score (on precomputed metric)
    try:
        if distance_matrix is not None:
            D_sub = distance_matrix[np.ix_(valid_mask, valid_mask)]
            metrics["silhouette"] = float(sk_silhouette(D_sub, valid_labels, metric="precomputed"))
        else:
            from core.distance import pairwise_segment_distances
            D = pairwise_segment_distances(segments[valid_mask])
            metrics["silhouette"] = float(sk_silhouette(D, valid_labels, metric="precomputed"))
    except Exception:
        pass

    # 2. Geometric Feature-based metrics (Davies-Bouldin, Calinski-Harabasz)
    try:
        X = extract_segment_features(segments)[valid_mask]
        metrics["davies_bouldin"] = float(sk_davies_bouldin(X, valid_labels))
        metrics["calinski_harabasz"] = float(sk_calinski_harabasz(X, valid_labels))
    except Exception:
        pass

    # 3. Quantum Interference Contrast Ratio (if K is provided)
    if K is not None:
        try:
            metrics["contrast_ratio"] = compute_interference_contrast(K, labels)
        except Exception:
            pass

    return metrics
