"""Original TRACLUS iterative MDL trajectory partitioning (Lee et al., 2007).

Implements sequential point-by-point scanning with candidate point projections
evaluated through explicit 2D coordinate rotation matrices.
"""

from typing import List, Tuple, Union
import math
import numpy as np


def _safe_log2(val: float, floor: float = 1.0) -> float:
    """Return log2(max(val, floor)) to avoid log2(0) or negative bits."""
    return math.log2(max(val, floor))


def rotate_points_2d(points: np.ndarray, origin: np.ndarray, phi: float) -> np.ndarray:
    """Rotate points relative to origin by angle -phi using explicit 2D rotation matrix.

    [x', y']^T = [[cos phi, sin phi], [-sin phi, cos phi]] [x - x0, y - y0]^T
    Aligns line segment with the positive X-axis.
    """
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    # Explicit 2x2 rotation matrix as specified in Lee et al., 2007
    R = np.array([
        [cos_phi, sin_phi],
        [-sin_phi, cos_phi],
    ], dtype=np.float64)

    translated = points - origin
    rotated = translated @ R.T
    return rotated


def iterative_mdl_costs_traclus(
    sub_trajectory: np.ndarray,
    penalty_ratio: float = 0.25,
) -> Tuple[float, float]:
    """Compute MDL_par and MDL_nopar for a candidate segment p_i -> p_k.

    Uses explicit 2D coordinate rotation matrix for projections and
    evaluates perpendicular and angular distances:
        L(H) = log2(||p_i - p_k||)
        L(D|H) = sum_{m=i}^{k-1} [ log2(d_perp(p_i p_k, p_m p_{m+1})) + log2(d_theta(p_i p_k, p_m p_{m+1})) ]
        L(No-Partition) = (1 + penalty_ratio) * sum_{m=i}^{k-1} log2(||p_m - p_{m+1}||)
    """
    pts = np.asarray(sub_trajectory, dtype=np.float64)
    k = len(pts)
    if k < 2:
        return 0.0, 0.0

    p_i = pts[0]
    p_k = pts[-1]
    dx = p_k[0] - p_i[0]
    dy = p_k[1] - p_i[1]
    seg_len = math.hypot(dx, dy)

    # 1. Hypothesis Length: L(H)
    lh = _safe_log2(seg_len)

    # 2. Data error given hypothesis: L(D|H)
    phi = math.atan2(dy, dx)
    # Rotate all points so p_i is at (0, 0) and p_k is at (seg_len, 0)
    pts_rot = rotate_points_2d(pts, origin=p_i, phi=phi)

    ldh = 0.0
    for m in range(k - 1):
        # Original subsegment endpoints rotated
        s_m = pts_rot[m]
        e_m = pts_rot[m + 1]
        v_m_x = e_m[0] - s_m[0]
        v_m_y = e_m[1] - s_m[1]
        len_m = math.hypot(v_m_x, v_m_y)

        # In rotated frame, line p_i p_k lies on the X-axis from 0 to seg_len.
        # Orthogonal projections onto X axis:
        # ps = (s_m[0], 0), pe = (e_m[0], 0)
        # Perpendicular distances:
        l_perp1 = abs(s_m[1])
        l_perp2 = abs(e_m[1])
        sum_p = l_perp1 + l_perp2
        if sum_p > 1e-12:
            d_perp = (l_perp1**2 + l_perp2**2) / sum_p
        else:
            d_perp = 0.0

        # Angular distance d_theta relative to base line (which is along +X direction: (1, 0))
        if len_m < 1e-12:
            d_theta = 0.0
        else:
            # cos(theta) = v_m_x / len_m
            cos_theta = max(-1.0, min(1.0, v_m_x / len_m))
            if cos_theta >= 0.0:
                sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
                d_theta = len_m * sin_theta
            else:
                d_theta = len_m

        ldh += _safe_log2(d_perp) + _safe_log2(d_theta)

    mdl_par = lh + ldh

    # 3. No-Partition Length: L(No-Partition) with penalty offset
    raw_nopar = 0.0
    for m in range(k - 1):
        step_len = math.hypot(pts[m + 1, 0] - pts[m, 0], pts[m + 1, 1] - pts[m, 1])
        raw_nopar += _safe_log2(step_len)

    mdl_nopar = (1.0 + penalty_ratio) * raw_nopar

    return mdl_par, mdl_nopar


def iterative_mdl_partition(
    trajectory: Union[np.ndarray, list],
    min_length: float = 1e-4,
    penalty_ratio: float = 0.25,
) -> np.ndarray:
    """Partition a trajectory into characteristic line segments using TRACLUS iterative MDL.

    Linear scanning: advance candidate partition until MDL_par > MDL_nopar.

    Parameters
    ----------
    trajectory : array_like of shape (N, 2)
        Raw sequence of 2D trajectory coordinates.
    min_length : float, default=1e-4
        Filter out consecutive stationary points.
    penalty_ratio : float, default=0.25
        Constant offset added to cost_nopar to suppress overly short segments.

    Returns
    -------
    segments : ndarray of shape (M, 2, 2)
        Extracted line segments.
    """
    pts = np.asarray(trajectory, dtype=np.float64)
    if len(pts) < 2:
        return np.empty((0, 2, 2), dtype=np.float64)

    # Filter stationary points
    diffs = np.hypot(pts[1:, 0] - pts[:-1, 0], pts[1:, 1] - pts[:-1, 1])
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
        cost_par, cost_nopar = iterative_mdl_costs_traclus(sub_pts, penalty_ratio=penalty_ratio)

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


def partition_trajectories_traclus(
    trajectories: List[Union[np.ndarray, list]],
    min_length: float = 1e-4,
    penalty_ratio: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition multiple trajectories using TRACLUS iterative MDL."""
    segments_list = []
    traj_ids_list = []
    seg_ids_list = []

    for t_idx, traj in enumerate(trajectories):
        segs = iterative_mdl_partition(traj, min_length=min_length, penalty_ratio=penalty_ratio)
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
