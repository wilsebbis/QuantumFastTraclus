"""Line segment distance metrics for TRACLUS and Fast-TRACLUS.

Provides both:
1. Iterative scalar implementation (original TRACLUS specification).
2. Fully vectorized multidimensional NumPy implementation (Fast-TRACLUS specification).
"""

from typing import Optional, Tuple, Union
import math
import numpy as np


def iterative_segment_distance(
    seg1: Union[np.ndarray, list],
    seg2: Union[np.ndarray, list],
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    return_components: bool = False,
) -> Union[float, Tuple[float, float, float]]:
    """Compute line segment distance using the iterative scalar formulation (Lee et al., 2007).

    Parameters
    ----------
    seg1 : array_like of shape (2, 2) or (4,)
        First line segment [[x1, y1], [x2, y2]].
    seg2 : array_like of shape (2, 2) or (4,)
        Second line segment [[x3, y3], [x4, y4]].
    weights : tuple of (w_perp, w_par, w_theta), default=(1.0, 1.0, 1.0)
        Weights for perpendicular, parallel, and angle distances.
    return_components : bool, default=False
        Whether to return (d_perp, d_par, d_theta) instead of total distance.

    Returns
    -------
    dist : float or tuple of float
        Total weighted distance or components tuple.
    """
    s1 = np.asarray(seg1, dtype=np.float64).reshape(2, 2)
    s2 = np.asarray(seg2, dtype=np.float64).reshape(2, 2)

    # Identical segment check
    if math.hypot(s1[0, 0] - s2[0, 0], s1[0, 1] - s2[0, 1]) < 1e-12 and \
       math.hypot(s1[1, 0] - s2[1, 0], s1[1, 1] - s2[1, 1]) < 1e-12:
        if return_components:
            return 0.0, 0.0, 0.0
        return 0.0

    len1 = math.hypot(s1[1, 0] - s1[0, 0], s1[1, 1] - s1[0, 1])
    len2 = math.hypot(s2[1, 0] - s2[0, 0], s2[1, 1] - s2[0, 1])

    # Designate the longer segment as Li and the shorter as Lj
    if len1 >= len2:
        Li_s, Li_e = s1[0], s1[1]
        Lj_s, Lj_e = s2[0], s2[1]
        len_i, len_j = len1, len2
    else:
        Li_s, Li_e = s2[0], s2[1]
        Lj_s, Lj_e = s1[0], s1[1]
        len_i, len_j = len2, len1

    if len_i < 1e-12:
        if return_components:
            return 0.0, 0.0, 0.0
        return 0.0

    # Li direction vector and unit vector
    v_i_x = Li_e[0] - Li_s[0]
    v_i_y = Li_e[1] - Li_s[1]
    u_i_x = v_i_x / len_i
    u_i_y = v_i_y / len_i

    # Projections of Lj_s and Lj_e onto the line containing Li
    t_s = (Lj_s[0] - Li_s[0]) * u_i_x + (Lj_s[1] - Li_s[1]) * u_i_y
    t_e = (Lj_e[0] - Li_s[0]) * u_i_x + (Lj_e[1] - Li_s[1]) * u_i_y

    ps_x = Li_s[0] + t_s * u_i_x
    ps_y = Li_s[1] + t_s * u_i_y
    pe_x = Li_s[0] + t_e * u_i_x
    pe_y = Li_s[1] + t_e * u_i_y

    # 1. Perpendicular distance d_perp (Order-2 Lehmer mean)
    l_perp1 = math.hypot(Lj_s[0] - ps_x, Lj_s[1] - ps_y)
    l_perp2 = math.hypot(Lj_e[0] - pe_x, Lj_e[1] - pe_y)
    sum_perp = l_perp1 + l_perp2
    if sum_perp > 1e-12:
        d_perp = (l_perp1**2 + l_perp2**2) / sum_perp
    else:
        d_perp = 0.0

    # 2. Parallel distance d_parallel = min(l_par1, l_par2)
    l_par1 = min(abs(t_s), abs(t_s - len_i))
    l_par2 = min(abs(t_e), abs(t_e - len_i))
    d_par = min(l_par1, l_par2)

    # 3. Angle distance d_theta
    v_j_x = Lj_e[0] - Lj_s[0]
    v_j_y = Lj_e[1] - Lj_s[1]
    if len_j < 1e-12:
        d_theta = 0.0
    else:
        dot_product = v_i_x * v_j_x + v_i_y * v_j_y
        cos_theta = max(-1.0, min(1.0, dot_product / (len_i * len_j)))
        if dot_product >= 0.0:
            if 1.0 - cos_theta < 1e-14:
                d_theta = 0.0
            else:
                sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
                d_theta = len_j * sin_theta
        else:
            d_theta = len_j

    if return_components:
        return float(d_perp), float(d_par), float(d_theta)

    total_dist = weights[0] * d_perp + weights[1] * d_par + weights[2] * d_theta
    return float(total_dist)


def vectorized_segment_distance(
    seg1: np.ndarray,
    seg2: np.ndarray,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    return_components: bool = False,
) -> Union[float, Tuple[float, float, float]]:
    """Compute segment distance for a single pair using the vectorized formulation."""
    arr1 = np.asarray(seg1, dtype=np.float64).reshape(1, 2, 2)
    arr2 = np.asarray(seg2, dtype=np.float64).reshape(1, 2, 2)
    res = pairwise_segment_distances(arr1, arr2, weights=weights, return_components=return_components)
    if return_components:
        return float(res[0][0, 0]), float(res[1][0, 0]), float(res[2][0, 0])
    return float(res[0, 0])


def pairwise_segment_distances(
    segments1: np.ndarray,
    segments2: Optional[np.ndarray] = None,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    return_components: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute pairwise line segment distances between collections via vectorized broadcasting.

    Parameters
    ----------
    segments1 : ndarray of shape (N, 2, 2) or (N, 4)
        First collection of line segments.
    segments2 : ndarray of shape (M, 2, 2) or (M, 4), optional
        Second collection. If None, computes pairwise self-distances within segments1.
    weights : tuple of (w_perp, w_par, w_theta), default=(1.0, 1.0, 1.0)
        Weights for perpendicular, parallel, and angle distances.
    return_components : bool, default=False
        If True, returns (d_perp, d_par, d_theta). Otherwise returns total weighted distance.

    Returns
    -------
    dist_matrix : ndarray of shape (N, M)
        Pairwise distance tensor.
    """
    arr1 = np.asarray(segments1, dtype=np.float64)
    if arr1.ndim == 2 and arr1.shape[1] == 4:
        arr1 = arr1.reshape(-1, 2, 2)
    elif arr1.ndim == 2 and arr1.shape == (2, 2):
        arr1 = arr1[np.newaxis, ...]

    is_self = segments2 is None
    if is_self:
        arr2 = arr1
    else:
        arr2 = np.asarray(segments2, dtype=np.float64)
        if arr2.ndim == 2 and arr2.shape[1] == 4:
            arr2 = arr2.reshape(-1, 2, 2)
        elif arr2.ndim == 2 and arr2.shape == (2, 2):
            arr2 = arr2[np.newaxis, ...]

    # Endpoints: s1, e1 broadcasted to (N, 1, 2); s2, e2 broadcasted to (1, M, 2)
    s1 = arr1[:, 0, :][:, np.newaxis, :]
    e1 = arr1[:, 1, :][:, np.newaxis, :]
    v1 = e1 - s1
    len1 = np.linalg.norm(v1, axis=-1)  # shape (N, 1)

    s2 = arr2[:, 0, :][np.newaxis, :, :]
    e2 = arr2[:, 1, :][np.newaxis, :, :]
    v2 = e2 - s2
    len2 = np.linalg.norm(v2, axis=-1)  # shape (1, M)

    # Boolean mask: True where segment 1 is longer or equal
    mask = len1 >= len2  # shape (N, M)
    mask_3d = mask[..., np.newaxis]

    len_i = np.where(mask, len1, len2)
    len_j = np.where(mask, len2, len1)

    Li_s = np.where(mask_3d, s1, s2)
    Li_e = np.where(mask_3d, e1, e2)
    v_i = np.where(mask_3d, v1, v2)

    Lj_s = np.where(mask_3d, s2, s1)
    Lj_e = np.where(mask_3d, e2, e1)
    v_j = np.where(mask_3d, v2, v1)

    safe_len_i = np.maximum(len_i[..., np.newaxis], 1e-15)
    u_i = v_i / safe_len_i

    # Projections of Lj endpoints onto Li line
    w_s = Lj_s - Li_s
    w_e = Lj_e - Li_s

    # Scalar projections along u_i: t = w . u_i
    t_s = w_s[..., 0] * u_i[..., 0] + w_s[..., 1] * u_i[..., 1]
    t_e = w_e[..., 0] * u_i[..., 0] + w_e[..., 1] * u_i[..., 1]

    # Orthogonal projection distances
    l_perp1 = np.abs(w_s[..., 0] * u_i[..., 1] - w_s[..., 1] * u_i[..., 0])
    l_perp2 = np.abs(w_e[..., 0] * u_i[..., 1] - w_e[..., 1] * u_i[..., 0])

    sum_perp = l_perp1 + l_perp2
    d_perp = np.where(
        sum_perp > 1e-12,
        (l_perp1**2 + l_perp2**2) / np.maximum(sum_perp, 1e-15),
        0.0,
    )

    # Parallel distances
    l_par1 = np.minimum(np.abs(t_s), np.abs(t_s - len_i))
    l_par2 = np.minimum(np.abs(t_e), np.abs(t_e - len_i))
    d_par = np.minimum(l_par1, l_par2)

    # Angle distances
    dot_prod = v_i[..., 0] * v_j[..., 0] + v_i[..., 1] * v_j[..., 1]
    denom = np.maximum(len_i * len_j, 1e-15)
    cos_theta = np.clip(dot_prod / denom, -1.0, 1.0)
    sin_sq = np.maximum(0.0, 1.0 - cos_theta**2)
    # Robust sin_theta avoiding float roundoff near 0 angle
    sin_theta = np.where(1.0 - cos_theta < 1e-14, 0.0, np.sqrt(sin_sq))

    d_theta = np.where(dot_prod >= 0.0, len_j * sin_theta, len_j)

    # Clean degenerate segments
    degen = (len_i < 1e-12) | (len_j < 1e-12)
    d_perp = np.where(degen, 0.0, d_perp)
    d_par = np.where(degen, 0.0, d_par)
    d_theta = np.where(degen, 0.0, d_theta)

    # Identical segment pair mask: endpoints match
    same_seg = (np.linalg.norm(s1 - s2, axis=-1) < 1e-12) & (np.linalg.norm(e1 - e2, axis=-1) < 1e-12)
    d_perp = np.where(same_seg, 0.0, d_perp)
    d_par = np.where(same_seg, 0.0, d_par)
    d_theta = np.where(same_seg, 0.0, d_theta)

    if is_self:
        # Enforce exact numerical symmetry and zero diagonal
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
