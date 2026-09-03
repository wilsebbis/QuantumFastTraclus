"""Vectorized Minimum Description Length (MDL) trajectory partitioning for Fast-TRACLUS.

Compresses raw trajectories into representative directional line segments by balancing
conciseness (hypothesis length L(H)) and preciseness (error given hypothesis L(D|H)).
All point-line projections are strictly formulated via vector dot products.
"""

from typing import List, Tuple, Union
import numpy as np


def _safe_log2(values: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Safe log2 calculation, returning 0 for values <= floor to avoid negative lengths."""
    safe_vals = np.maximum(values, floor)
    return np.log2(safe_vals)


def point_segment_projections(
    points: np.ndarray,
    p_start: np.ndarray,
    p_end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orthogonal (d_perp) and parallel (d_par) distances from points to a segment.

    Formulated strictly via vector dot products without coordinate rotation matrices.

    Parameters
    ----------
    points : ndarray of shape (K, 2)
        Intermediate points to project.
    p_start : ndarray of shape (2,)
        Start coordinate of base segment.
    p_end : ndarray of shape (2,)
        End coordinate of base segment.

    Returns
    -------
    d_perp : ndarray of shape (K,)
        Perpendicular distance from each point to the line p_start -> p_end.
    d_par : ndarray of shape (K,)
        Parallel distance from each point's projection outside the base segment [0, seg_len].
        If a point projects inside the segment, d_par is 0.
    """
    v = p_end - p_start
    seg_len = np.linalg.norm(v)

    if seg_len < 1e-12:
        diff = points - p_start
        d_perp = np.linalg.norm(diff, axis=-1)
        d_par = np.zeros_like(d_perp)
        return d_perp, d_par

    u = v / seg_len  # Unit direction vector

    # Vectors from p_start to each point
    w = points - p_start

    # Scalar projection along segment axis: t = w . u
    t = w[:, 0] * u[0] + w[:, 1] * u[1]

    # Orthogonal rejection magnitude: ||w_perp|| = |w_x * u_y - w_y * u_x|
    d_perp = np.abs(w[:, 0] * u[1] - w[:, 1] * u[0])

    # Parallel distance: longitudinal deviation outside the bounds [0, seg_len]
    d_par = np.maximum(0.0, np.maximum(-t, t - seg_len))

    return d_perp, d_par


def mdl_cost(sub_trajectory: np.ndarray) -> Tuple[float, float]:
    """Calculate MDL costs for partitioning vs not partitioning a sub-trajectory.

    Parameters
    ----------
    sub_trajectory : ndarray of shape (K, 2)
        Sub-sequence of trajectory points [p_1, p_2, ..., p_K].

    Returns
    -------
    mdl_par : float
        L(H) + L(D|H) if represented by single segment p_1 -> p_K.
    mdl_nopar : float
        L(No-Partition) representing the original consecutive segments.
    """
    pts = np.asarray(sub_trajectory, dtype=np.float64)
    k = len(pts)
    if k < 2:
        return 0.0, 0.0

    p_start = pts[0]
    p_end = pts[-1]
    v_base = p_end - p_start
    base_len = np.linalg.norm(v_base)

    # 1. Hypothesis length L(H) = log2(||p_start - p_end||)
    lh = float(_safe_log2(np.array([base_len]))[0])

    # 2. Preciseness length L(D|H) = sum of log2(d_perp) + log2(d_par) for intermediate points
    if k > 2:
        intermediate_pts = pts[1:-1]
        d_perp, d_par = point_segment_projections(intermediate_pts, p_start, p_end)
        ldh = float(np.sum(_safe_log2(d_perp) + _safe_log2(d_par)))
    else:
        ldh = 0.0

    mdl_par = lh + ldh

    # 3. Baseline: L(No-Partition) = sum of log2 lengths of consecutive raw segments
    step_diffs = pts[1:] - pts[:-1]
    step_lens = np.linalg.norm(step_diffs, axis=-1)
    mdl_nopar = float(np.sum(_safe_log2(step_lens)))

    return mdl_par, mdl_nopar


def partition_trajectory(
    trajectory: Union[np.ndarray, list],
    min_length: float = 1e-4,
) -> np.ndarray:
    """Partition a single trajectory into representative line segments using greedy MDL.

    Parameters
    ----------
    trajectory : array_like of shape (N, 2)
        Sequence of 2D coordinates [x, y] forming a trajectory.
    min_length : float, default=1e-4
        Minimum length threshold to discard stationary or duplicate points.

    Returns
    -------
    segments : ndarray of shape (M, 2, 2)
        Array of extracted line segments [[start_x, start_y], [end_x, end_y]].
    """
    pts = np.asarray(trajectory, dtype=np.float64)
    if len(pts) < 2:
        return np.empty((0, 2, 2), dtype=np.float64)

    # Remove consecutive duplicate points
    diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = diffs >= min_length
    pts = pts[keep]

    n_points = len(pts)
    if n_points < 2:
        return np.empty((0, 2, 2), dtype=np.float64)
    if n_points == 2:
        return np.array([[pts[0], pts[1]]], dtype=np.float64)

    segments = []
    start_idx = 0
    length = 1

    while start_idx + length < n_points:
        curr_idx = start_idx + length
        sub_pts = pts[start_idx : curr_idx + 1]
        cost_par, cost_nopar = mdl_cost(sub_pts)

        # Condition: continue expanding while partitioned description is more concise than no-partition
        if cost_par <= cost_nopar:
            length += 1
        else:
            # Exceeded error threshold: add characteristic segment up to previous point
            best_end_idx = curr_idx - 1 if length > 1 else curr_idx
            segments.append([pts[start_idx], pts[best_end_idx]])
            start_idx = best_end_idx
            length = 1

    # Add terminal segment if unclosed
    if start_idx < n_points - 1:
        segments.append([pts[start_idx], pts[-1]])

    return np.asarray(segments, dtype=np.float64)


def partition_trajectories(
    trajectories: List[Union[np.ndarray, list]],
    min_length: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition multiple trajectories into a unified segment collection.

    Parameters
    ----------
    trajectories : list of array_like
        List of trajectories, each of shape (N_i, 2).
    min_length : float, default=1e-4
        Minimum length to filter zero-movement jitter.

    Returns
    -------
    all_segments : ndarray of shape (Total_M, 2, 2)
        Unified array of all extracted line segments.
    traj_ids : ndarray of shape (Total_M,)
        Original trajectory index for each segment.
    seg_ids : ndarray of shape (Total_M,)
        Local segment index within its parent trajectory.
    """
    segments_list = []
    traj_ids_list = []
    seg_ids_list = []

    for t_idx, traj in enumerate(trajectories):
        segs = partition_trajectory(traj, min_length=min_length)
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
