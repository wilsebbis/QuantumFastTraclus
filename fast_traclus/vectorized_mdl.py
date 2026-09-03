"""Fast-TRACLUS fully vectorized MDL trajectory partitioning (González Delgado et al., 2026).

Eliminates Python loops using direct scalar vector dot products and vectorized array operations.
"""

from typing import List, Tuple, Union
import numpy as np


def _safe_log2(values: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Vectorized safe log2 calculation avoiding log2(0) or negative bits."""
    return np.log2(np.maximum(values, floor))


def vectorized_point_projections(
    points: np.ndarray,
    s_i: np.ndarray,
    e_i: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orthogonal and parallel projections via direct vector dot products.

    u_1 = ((s_j - s_i) . (e_i - s_i)) / ||e_i - s_i||^2
    p_s = s_i + u_1 * (e_i - s_i)

    Parameters
    ----------
    points : ndarray of shape (K, 2)
        Intermediate points.
    s_i : ndarray of shape (2,)
        Start coordinate of candidate segment.
    e_i : ndarray of shape (2,)
        End coordinate of candidate segment.

    Returns
    -------
    d_perp : ndarray of shape (K,)
        Orthogonal projection distances.
    d_par : ndarray of shape (K,)
        Longitudinal projection distances outside [0, seg_len].
    """
    v = e_i - s_i
    seg_len_sq = float(np.dot(v, v))
    seg_len = np.sqrt(seg_len_sq)

    if seg_len < 1e-12:
        diff = points - s_i
        d_perp = np.linalg.norm(diff, axis=-1)
        d_par = np.zeros_like(d_perp)
        return d_perp, d_par

    # Direct scalar vector dot product: u1 = ((s_j - s_i) . v) / ||v||^2
    diff = points - s_i
    u1 = (diff[:, 0] * v[0] + diff[:, 1] * v[1]) / seg_len_sq
    p_proj = s_i + u1[:, np.newaxis] * v

    # Perpendicular distance ||points - p_proj||
    rejection = points - p_proj
    d_perp = np.linalg.norm(rejection, axis=-1)

    # Parallel distance outside [0, seg_len]
    t = u1 * seg_len
    d_par = np.maximum(0.0, np.maximum(-t, t - seg_len))

    return d_perp, d_par


def vectorized_mdl_costs_fast(sub_pts: np.ndarray) -> Tuple[float, float]:
    """Compute Fast-TRACLUS vectorized MDL costs L(H) + L(D|H) vs L(No-Partition)."""
    k = len(sub_pts)
    if k < 2:
        return 0.0, 0.0

    s_i = sub_pts[0]
    e_i = sub_pts[-1]
    base_len = np.linalg.norm(e_i - s_i)

    # L(H)
    lh = float(_safe_log2(np.array([base_len]))[0])

    # L(D|H)
    if k > 2:
        intermediate = sub_pts[1:-1]
        d_perp, d_par = vectorized_point_projections(intermediate, s_i, e_i)
        ldh = float(np.sum(_safe_log2(d_perp) + _safe_log2(d_par)))
    else:
        ldh = 0.0

    mdl_par = lh + ldh

    # L(No-Partition)
    step_diffs = sub_pts[1:] - sub_pts[:-1]
    step_lens = np.linalg.norm(step_diffs, axis=-1)
    mdl_nopar = float(np.sum(_safe_log2(step_lens)))

    return mdl_par, mdl_nopar


def vectorized_mdl_partition(
    trajectory: Union[np.ndarray, list],
    min_length: float = 1e-4,
) -> np.ndarray:
    """Partition a trajectory into characteristic line segments using Fast-TRACLUS vectorized MDL."""
    pts = np.asarray(trajectory, dtype=np.float64)
    if len(pts) < 2:
        return np.empty((0, 2, 2), dtype=np.float64)

    # Remove stationary points
    diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = diffs >= min_length
    pts = pts[keep]

    n_points = len(pts)
    if n_points < 2:
        return np.empty((0, 2, 2), dtype=np.float64)
    if n_points == 2:
        return np.array([[pts[0], pts[1]]], dtype=np.float64)

    start_idx = 0
    length = 1
    segments = []

    while start_idx + length < n_points:
        curr_idx = start_idx + length
        sub_pts = pts[start_idx : curr_idx + 1]
        cost_par, cost_nopar = vectorized_mdl_costs_fast(sub_pts)

        if cost_par <= cost_nopar:
            length += 1
        else:
            best_end_idx = curr_idx - 1 if length > 1 else curr_idx
            segments.append([pts[start_idx], pts[best_end_idx]])
            start_idx = best_end_idx
            length = 1

    if start_idx < n_points - 1:
        segments.append([pts[start_idx], pts[-1]])

    return np.asarray(segments, dtype=np.float64)


def partition_trajectories_fast(
    trajectories: List[Union[np.ndarray, list]],
    min_length: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition multiple trajectories into a unified segment collection."""
    segments_list = []
    traj_ids_list = []
    seg_ids_list = []

    for t_idx, traj in enumerate(trajectories):
        segs = vectorized_mdl_partition(traj, min_length=min_length)
        if len(segs) > 0:
            segments_list.append(segs)
            traj_ids_list.append(np.full(len(segs), t_idx, dtype=int))
            seg_ids_list.append(np.arange(len(segs), dtype=int))

    if len(segments_list) == 0:
        return (
            np.empty((0, 2, 2), dtype=np.float64),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )

    all_segments = np.concatenate(segments_list, axis=0)
    traj_ids = np.concatenate(traj_ids_list, axis=0)
    seg_ids = np.concatenate(seg_ids_list, axis=0)

    return all_segments, traj_ids, seg_ids
