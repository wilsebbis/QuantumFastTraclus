"""Validation and comparison metrics for trajectory clustering.

Implements Davies-Bouldin Index, Silhouette Score, and Interference Contrast Ratio
to assess manifold corridor separation, coherence, and runtime.
"""

from typing import Any, Dict, Optional, Tuple, Union
import time
import numpy as np
from sklearn.metrics import davies_bouldin_score as sk_davies_bouldin
from sklearn.metrics import silhouette_score as sk_silhouette


def extract_segment_features(segments: np.ndarray) -> np.ndarray:
    """Extract rotation- and scale-aware feature representation for line segments.

    Features per segment:
    [midpoint_x, midpoint_y, length, cos(theta), sin(theta)]
    """
    s = segments[:, 0, :]
    e = segments[:, 1, :]
    midpoint = 0.5 * (s + e)
    diff = e - s
    length = np.linalg.norm(diff, axis=-1, keepdims=True)
    safe_len = np.maximum(length, 1e-12)
    direction = diff / safe_len

    features = np.hstack([midpoint, length, direction])
    return features


def davies_bouldin_index(
    segments: np.ndarray,
    labels: np.ndarray,
    feature_matrix: Optional[np.ndarray] = None,
) -> float:
    """Calculate Davies-Bouldin Index (DBI) for clustered segments.

    Lower values indicate superior compactness and separation between corridors.
    Ignores noise points labeled -1.

    Parameters
    ----------
    segments : ndarray of shape (N, 2, 2)
        Line segments.
    labels : ndarray of shape (N,)
        Cluster assignments (-1 denotes noise).
    feature_matrix : ndarray of shape (N, D), optional
        Pre-extracted segment features. If None, extracted automatically.

    Returns
    -------
    dbi : float
        Davies-Bouldin index, or float('nan') if fewer than 2 valid clusters.
    """
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]

    unique_clusters = np.unique(valid_labels)
    if len(unique_clusters) < 2:
        return float("nan")

    if feature_matrix is None:
        X = extract_segment_features(segments)[valid_mask]
    else:
        X = feature_matrix[valid_mask]

    try:
        dbi = float(sk_davies_bouldin(X, valid_labels))
        return dbi
    except Exception:
        return float("nan")


def silhouette_score(
    segments: np.ndarray,
    labels: np.ndarray,
    distance_matrix: Optional[np.ndarray] = None,
) -> float:
    """Calculate Silhouette Score across manifold corridors.

    Higher values indicate better-aligned, well-separated trajectory clusters.
    Ignores noise points labeled -1.

    Parameters
    ----------
    segments : ndarray of shape (N, 2, 2)
        Line segments.
    labels : ndarray of shape (N,)
        Cluster assignments (-1 denotes noise).
    distance_matrix : ndarray of shape (N, N), optional
        Precomputed pairwise TRACLUS distance matrix.

    Returns
    -------
    score : float
        Silhouette score in [-1, 1], or float('nan') if fewer than 2 clusters.
    """
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]

    unique_clusters = np.unique(valid_labels)
    if len(unique_clusters) < 2:
        return float("nan")

    try:
        if distance_matrix is not None:
            D_sub = distance_matrix[np.ix_(valid_mask, valid_mask)]
            score = float(sk_silhouette(D_sub, valid_labels, metric="precomputed"))
        else:
            from .distance import pairwise_segment_distances
            D = pairwise_segment_distances(segments[valid_mask])
            score = float(sk_silhouette(D, valid_labels, metric="precomputed"))
        return score
    except Exception:
        return float("nan")


def interference_contrast_ratio(
    P: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Calculate the quantum multipath interference contrast ratio C.

    C = E[P_jk | j, k in same cluster] / E[P_jk | j, k in different clusters]
    (for j != k and valid cluster assignments j, k >= 0).

    High values (~8x to 50x) demonstrate that wave mechanics confinement
    naturally partitions non-convex corridors with minimal inter-corridor leakage.

    Parameters
    ----------
    P : ndarray of shape (N, N)
        CTQW transition probability matrix P_jk = |<k| U(t) |j>|^2.
    labels : ndarray of shape (N,)
        Cluster assignments.

    Returns
    -------
    contrast_ratio : float
        Constructive-to-destructive wave interference contrast.
    """
    N = len(labels)
    if N < 2:
        return 0.0

    valid_mask = labels >= 0
    if np.sum(valid_mask) < 2:
        return 0.0

    # Intra-cluster mask (same cluster, j != k)
    same_cluster = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
    np.fill_diagonal(same_cluster, False)

    # Inter-cluster mask (different valid clusters)
    diff_cluster = (labels[:, None] != labels[None, :]) & (labels[:, None] >= 0) & (labels[None, :] >= 0)

    intra_vals = P[same_cluster]
    inter_vals = P[diff_cluster]

    if len(intra_vals) == 0 or len(inter_vals) == 0:
        return 0.0

    mean_intra = float(np.mean(intra_vals))
    mean_inter = float(np.mean(inter_vals))

    if mean_inter <= 1e-15:
        return float("inf") if mean_intra > 0.0 else 0.0

    return mean_intra / mean_inter


def benchmark_clustering(
    ctqw_pipeline: Any,
    spectral_pipeline: Any,
    trajectories: list,
) -> Dict[str, Any]:
    """Execute head-to-head benchmark between CTQW and classical Spectral Clustering.

    Returns comparison dictionary with runtimes, DBI, Silhouette, and Contrast metrics.
    """
    results = {}

    # 1. Run Fast-TRACLUS CTQW
    t0 = time.perf_counter()
    ctqw_pipeline.fit(trajectories)
    t_ctqw = time.perf_counter() - t0

    segments = ctqw_pipeline.segments_
    labels_ctqw = ctqw_pipeline.labels_
    P_ctqw = ctqw_pipeline.clusterer_.P_

    dbi_ctqw = davies_bouldin_index(segments, labels_ctqw)
    sil_ctqw = silhouette_score(segments, labels_ctqw)
    contrast_ctqw = interference_contrast_ratio(P_ctqw, labels_ctqw) if P_ctqw is not None else 0.0

    n_clusters_ctqw = len(np.unique(labels_ctqw[labels_ctqw >= 0]))
    noise_ratio_ctqw = float(np.mean(labels_ctqw == -1))

    results["ctqw"] = {
        "runtime_sec": t_ctqw,
        "n_clusters": n_clusters_ctqw,
        "noise_ratio": noise_ratio_ctqw,
        "davies_bouldin": dbi_ctqw,
        "silhouette": sil_ctqw,
        "contrast_ratio": contrast_ctqw,
    }

    # 2. Run Classical Spectral Clustering Baseline
    t0 = time.perf_counter()
    spectral_pipeline.fit(trajectories)
    t_spec = time.perf_counter() - t0

    labels_spec = spectral_pipeline.labels_
    dbi_spec = davies_bouldin_index(segments, labels_spec)
    sil_spec = silhouette_score(segments, labels_spec)

    n_clusters_spec = len(np.unique(labels_spec[labels_spec >= 0]))
    noise_ratio_spec = float(np.mean(labels_spec == -1))

    # Evaluate CTQW transition probabilities against spectral cluster assignments
    contrast_spec = interference_contrast_ratio(P_ctqw, labels_spec) if P_ctqw is not None else 0.0

    results["spectral"] = {
        "runtime_sec": t_spec,
        "n_clusters": n_clusters_spec,
        "noise_ratio": noise_ratio_spec,
        "davies_bouldin": dbi_spec,
        "silhouette": sil_spec,
        "contrast_ratio": contrast_spec,
    }

    return results
