"""Vectorized segment distance metrics for Fast-TRACLUS.

Implements TRACLUS line segment distance components (Perpendicular, Parallel,
and Angular) without manual Python loops, utilizing vectorized NumPy broadcasting.
"""

from typing import Optional, Tuple, Union
import numpy as np


def _ensure_segment_array(segments: Union[np.ndarray, list]) -> np.ndarray:
    """Ensure input is a float64 NumPy array of shape (N, 2, 2)."""
    arr = np.asarray(segments, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape == (2, 2):
            arr = arr[np.newaxis, ...]
        elif arr.shape[1] == 4:
            # Format (N, 4): [x1, y1, x2, y2]
            arr = arr.reshape(-1, 2, 2)
        else:
            raise ValueError(f"Invalid segment array shape: {arr.shape}")
    elif arr.ndim != 3 or arr.shape[1:] != (2, 2):
        raise ValueError(f"Expected segments with shape (N, 2, 2), got {arr.shape}")
    return arr


def _compute_components_vectorized(
    s1: np.ndarray,
    e1: np.ndarray,
    v1: np.ndarray,
    len1: np.ndarray,
    s2: np.ndarray,
    e2: np.ndarray,
    v2: np.ndarray,
    len2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (d_perp, d_parallel, d_theta) for aligned arrays of segments."""
    # Determine longer vs shorter segment for each pair
    mask = len1 >= len2  # True where segment 1 is longer or equal

    len_l = np.where(mask, len1, len2)
    len_s = np.where(mask, len2, len1)

    mask_3d = mask[..., np.newaxis]
    s_l = np.where(mask_3d, s1, s2)
    e_l = np.where(mask_3d, e1, e2)
    v_l = np.where(mask_3d, v1, v2)

    s_s = np.where(mask_3d, s2, s1)
    e_s = np.where(mask_3d, e2, e1)
    v_s = np.where(mask_3d, v2, v1)

    # Unit direction vector of longer segment
    safe_len_l = np.maximum(len_l[..., np.newaxis], 1e-15)
    u_l = v_l / safe_len_l

    # --- 1. Perpendicular Distance (d_perp) ---
    # Vectors from start of longer segment to endpoints of shorter segment
    w_s = s_s - s_l
    w_e = e_s - e_l  # or e_s - s_l, cross product with u_l gives same perpendicular distance
    w_e_from_start = e_s - s_l

    # 2D cross product: w x u_l = w_x * u_ly - w_y * u_lx
    l_perp1 = np.abs(w_s[..., 0] * u_l[..., 1] - w_s[..., 1] * u_l[..., 0])
    l_perp2 = np.abs(w_e_from_start[..., 0] * u_l[..., 1] - w_e_from_start[..., 1] * u_l[..., 0])

    sum_perp = l_perp1 + l_perp2
    # Standard TRACLUS formulation: (l1^2 + l2^2) / (l1 + l2)
    d_perp = np.where(
        sum_perp > 1e-12,
        (l_perp1**2 + l_perp2**2) / np.maximum(sum_perp, 1e-15),
        0.0,
    )

    # --- 2. Parallel Distance (d_parallel) ---
    # Scalar projections onto u_l
    t1 = w_s[..., 0] * u_l[..., 0] + w_s[..., 1] * u_l[..., 1]
    t2 = w_e_from_start[..., 0] * u_l[..., 0] + w_e_from_start[..., 1] * u_l[..., 1]

    # Distance to endpoints 0 and len_l along longer segment's axis
    l_par1 = np.minimum(np.abs(t1), np.abs(t1 - len_l))
    l_par2 = np.minimum(np.abs(t2), np.abs(t2 - len_l))
    d_par = np.minimum(l_par1, l_par2)

    # --- 3. Angular Distance (d_theta) ---
    dot_prod = v_l[..., 0] * v_s[..., 0] + v_l[..., 1] * v_s[..., 1]
    denom = np.maximum(len_l * len_s, 1e-15)
    cos_theta = np.clip(dot_prod / denom, -1.0, 1.0)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))

    # d_theta = ||L_shorter|| * sin(theta) if theta <= pi/2 (cos >= 0) else ||L_shorter||
    d_theta = np.where(dot_prod >= 0.0, len_s * sin_theta, len_s)

    # Clean up degenerate segments (length ~ 0)
    degen = (len_l < 1e-12) | (len_s < 1e-12)
    d_perp = np.where(degen, 0.0, d_perp)
    d_par = np.where(degen, 0.0, d_par)
    d_theta = np.where(degen, 0.0, d_theta)

    return d_perp, d_par, d_theta


def perpendicular_distance(seg1: np.ndarray, seg2: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate perpendicular distance between two segments or batches of segments."""
    s1 = _ensure_segment_array(seg1)
    s2 = _ensure_segment_array(seg2)
    dist_matrix = pairwise_segment_distances(s1, s2, return_components=True)[0]
    return float(dist_matrix[0, 0]) if s1.shape[0] == 1 and s2.shape[0] == 1 else dist_matrix


def parallel_distance(seg1: np.ndarray, seg2: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate parallel distance between two segments or batches of segments."""
    s1 = _ensure_segment_array(seg1)
    s2 = _ensure_segment_array(seg2)
    dist_matrix = pairwise_segment_distances(s1, s2, return_components=True)[1]
    return float(dist_matrix[0, 0]) if s1.shape[0] == 1 and s2.shape[0] == 1 else dist_matrix


def angular_distance(seg1: np.ndarray, seg2: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate angular distance between two segments or batches of segments."""
    s1 = _ensure_segment_array(seg1)
    s2 = _ensure_segment_array(seg2)
    dist_matrix = pairwise_segment_distances(s1, s2, return_components=True)[2]
    return float(dist_matrix[0, 0]) if s1.shape[0] == 1 and s2.shape[0] == 1 else dist_matrix


def segment_distance(seg1: np.ndarray, seg2: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate total TRACLUS distance D(L1, L2) = d_perp + d_parallel + d_theta."""
    s1 = _ensure_segment_array(seg1)
    s2 = _ensure_segment_array(seg2)
    dist_matrix = pairwise_segment_distances(s1, s2, return_components=False)
    return float(dist_matrix[0, 0]) if s1.shape[0] == 1 and s2.shape[0] == 1 else dist_matrix


def pairwise_segment_distances(
    segments1: Union[np.ndarray, list],
    segments2: Optional[Union[np.ndarray, list]] = None,
    return_components: bool = False,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute pairwise TRACLUS distances between two collections of line segments.

    Parameters
    ----------
    segments1 : array_like of shape (N, 2, 2) or (N, 4)
        First collection of directed line segments [[start_x, start_y], [end_x, end_y]].
    segments2 : array_like of shape (M, 2, 2) or (M, 4), optional
        Second collection of line segments. If None, computes pairwise distances within segments1.
    return_components : bool, default=False
        If True, returns (d_perp, d_parallel, d_theta). Otherwise returns weighted total distance D.
    weights : tuple of (w_perp, w_par, w_theta), default=(1.0, 1.0, 1.0)
        Weights for perpendicular, parallel, and angular distance components.

    Returns
    -------
    dist_matrix : ndarray of shape (N, M)
        Pairwise distance matrix, or tuple of component matrices if return_components=True.
    """
    arr1 = _ensure_segment_array(segments1)
    is_self = segments2 is None
    arr2 = arr1 if is_self else _ensure_segment_array(segments2)

    N = arr1.shape[0]
    M = arr2.shape[0]

    # Endpoints and vectors for collection 1, broadcasted to shape (N, 1, 2)
    s1 = arr1[:, 0, :][:, np.newaxis, :]
    e1 = arr1[:, 1, :][:, np.newaxis, :]
    v1 = e1 - s1
    len1 = np.linalg.norm(v1, axis=-1)  # shape (N, 1)

    # Endpoints and vectors for collection 2, broadcasted to shape (1, M, 2)
    s2 = arr2[:, 0, :][np.newaxis, :, :]
    e2 = arr2[:, 1, :][np.newaxis, :, :]
    v2 = e2 - s2
    len2 = np.linalg.norm(v2, axis=-1)  # shape (1, M)

    d_perp, d_par, d_theta = _compute_components_vectorized(
        s1=s1, e1=e1, v1=v1, len1=len1,
        s2=s2, e2=e2, v2=v2, len2=len2,
    )

    if is_self:
        # Enforce exact symmetry and zero diagonal
        d_perp = 0.5 * (d_perp + d_perp.T)
        d_par = 0.5 * (d_par + d_par.T)
        d_theta = 0.5 * (d_theta + d_theta.T)
        np.fill_diagonal(d_perp, 0.0)
        np.fill_diagonal(d_par, 0.0)
        np.fill_diagonal(d_theta, 0.0)

    if return_components:
        return d_perp, d_par, d_theta

    total = weights[0] * d_perp + weights[1] * d_par + weights[2] * d_theta
    if is_self:
        np.fill_diagonal(total, 0.0)
    return total
