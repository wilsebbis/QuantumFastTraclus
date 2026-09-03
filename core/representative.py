"""Representative trajectory generation using the TRACLUS sweep-line algorithm."""

from typing import Dict, List, Optional, Union
import math
import numpy as np


def generate_representative_trajectory(
    cluster_segments: np.ndarray,
    min_lines: int = 3,
    gamma: float = 1.0,
) -> np.ndarray:
    """Generate a representative trajectory for a cluster of directed line segments.

    Implements the sweep-line algorithm (Lee et al., 2007):
    1. Computes the cluster average direction vector V and rotates frame by -alpha to align with +X.
    2. Sweeps across segment X-coordinates, computing average Y for positions with >= min_lines intersections.
    3. Smooths with minimum point separation gamma and rotates back to original space.

    Parameters
    ----------
    cluster_segments : ndarray of shape (N, 2, 2)
        Collection of line segments [[start_x, start_y], [end_x, end_y]] in the cluster.
    min_lines : int, default=3
        Minimum segment density required to generate a representative point.
    gamma : float, default=1.0
        Smoothing threshold to prevent point clustering along the sweep axis.

    Returns
    -------
    rep_trajectory : ndarray of shape (K, 2)
        Smoothed representative 2D trajectory.
    """
    segs = np.asarray(cluster_segments, dtype=np.float64)
    if len(segs) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(segs) == 1:
        return segs[0]

    # Compute cluster average direction vector V
    vectors = segs[:, 1, :] - segs[:, 0, :]
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe_lengths = np.maximum(lengths, 1e-12)
    unit_vectors = vectors / safe_lengths
    V = np.mean(unit_vectors, axis=0)
    norm_V = np.linalg.norm(V)

    if norm_V < 1e-6:
        # Fallback to PCA if direction is isotropic
        pts = segs.reshape(-1, 2)
        cov = np.cov(pts.T)
        _, evecs = np.linalg.eigh(cov)
        u_axis = evecs[:, -1]
        alpha = math.atan2(u_axis[1], u_axis[0])
    else:
        alpha = math.atan2(V[1], V[0])

    # Rotation matrix to align V with positive X axis: R(-alpha)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    R_align = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    R_restore = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rotate all segments: (N, 2, 2) @ (2, 2).T -> (N, 2, 2)
    s_rot = segs[:, 0, :] @ R_align.T
    e_rot = segs[:, 1, :] @ R_align.T

    # Ensure start X <= end X for sweep line evaluation
    swap = s_rot[:, 0] > e_rot[:, 0]
    starts = np.where(swap[:, np.newaxis], e_rot, s_rot)
    ends = np.where(swap[:, np.newaxis], s_rot, e_rot)

    # Event points along rotated X axis
    x_events = np.unique(np.concatenate([starts[:, 0], ends[:, 0]]))
    x_events.sort()

    cand_points = []
    for x_p in x_events:
        # Check which segments intersect vertical line X = x_p
        intersecting = (starts[:, 0] <= x_p) & (x_p <= ends[:, 0])
        num_intersect = np.sum(intersecting)

        if num_intersect >= min_lines:
            # Interpolate Y coordinates
            s_sub = starts[intersecting]
            e_sub = ends[intersecting]
            dx = e_sub[:, 0] - s_sub[:, 0]
            # Handle vertical segments
            safe_dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
            ratio = np.clip((x_p - s_sub[:, 0]) / safe_dx, 0.0, 1.0)
            y_interp = s_sub[:, 1] + ratio * (e_sub[:, 1] - s_sub[:, 1])
            mean_y = float(np.mean(y_interp))
            cand_points.append([x_p, mean_y])

    if len(cand_points) == 0:
        # Fallback: project centroids if sweep density threshold wasn't met
        centroids = 0.5 * (starts + ends)
        order = np.argsort(centroids[:, 0])
        centroids_sorted = centroids[order]
        cand_points = [centroids_sorted[0], centroids_sorted[-1]]

    cand_pts = np.asarray(cand_points, dtype=np.float64)

    # Smoothing step: filter points closer than gamma
    smoothed = [cand_pts[0]]
    for pt in cand_pts[1:]:
        if np.linalg.norm(pt - smoothed[-1]) >= gamma:
            smoothed.append(pt)
    if len(smoothed) < 2 and len(cand_pts) >= 2:
        smoothed.append(cand_pts[-1])

    smoothed_arr = np.asarray(smoothed, dtype=np.float64)

    # Rotate back to original coordinate space
    rep_traj = smoothed_arr @ R_restore.T
    return rep_traj
