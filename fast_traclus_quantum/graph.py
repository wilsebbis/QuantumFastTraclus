"""Sparse affinity graph and normalized Laplacian construction for Fast-TRACLUS.

Constructs sparse affinity matrices and symmetric normalized Laplacians
with zero-degree protection, supporting spatial indexing for scalability.
"""

from typing import Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree

from .distance import pairwise_segment_distances, segment_distance


def build_affinity_matrix(
    segments: np.ndarray,
    eps: float,
    sigma: Optional[float] = None,
    use_spatial_index: bool = True,
    spatial_threshold_n: int = 1000,
) -> sp.csr_matrix:
    """Construct sparse symmetric affinity matrix W with Gaussian radial basis.

    W_jk = exp(-D(L_j, L_k)^2 / (2 * sigma^2))  if D(L_j, L_k) <= eps, else 0.
    W_jj = 0 (no self-loops).

    Parameters
    ----------
    segments : ndarray of shape (N, 2, 2)
        Extracted line segments [[x_start, y_start], [x_end, y_end]].
    eps : float
        Spatial connectivity threshold. Pairs with D(L_j, L_k) > eps have W_jk = 0.
    sigma : float, optional
        Gaussian kernel bandwidth. If None, defaults to eps / 2.0.
    use_spatial_index : bool, default=True
        Whether to prune distant pairs using KD-Tree spatial indexing when N is large.
    spatial_threshold_n : int, default=1000
        Segment count above which spatial indexing is triggered if use_spatial_index=True.

    Returns
    -------
    W : scipy.sparse.csr_matrix of shape (N, N)
        Sparse symmetric affinity matrix.
    """
    N = len(segments)
    if N == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64)
    if N == 1:
        return sp.csr_matrix((1, 1), dtype=np.float64)

    if sigma is None:
        sigma = max(eps / 2.0, 1e-6)

    gamma = 1.0 / (2.0 * sigma**2)

    # Fast path for moderate N: direct vectorized computation
    if (not use_spatial_index) or (N < spatial_threshold_n):
        D = pairwise_segment_distances(segments)
        # Mask where distance <= eps and distance > 0 (strictly excluding diagonal)
        mask = (D <= eps) & (D > 0.0)
        np.fill_diagonal(mask, False)

        weights = np.zeros_like(D)
        weights[mask] = np.exp(-gamma * (D[mask] ** 2))
        return sp.csr_matrix(weights)

    # Scalable path for large N: KD-Tree candidate pre-filtering
    # Segment midpoints and bounding radiuses
    midpoints = 0.5 * (segments[:, 0, :] + segments[:, 1, :])
    seg_vectors = segments[:, 1, :] - segments[:, 0, :]
    half_lengths = 0.5 * np.linalg.norm(seg_vectors, axis=-1)
    max_half_len = float(np.max(half_lengths)) if len(half_lengths) > 0 else 0.0

    # Upper bound on midpoint distance for two segments to be within eps TRACLUS distance:
    # dist(midpoint_i, midpoint_j) <= eps + half_length_i + half_length_j
    search_radius = eps + 2.0 * max_half_len

    tree = KDTree(midpoints)
    candidate_pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

    if len(candidate_pairs) == 0:
        return sp.csr_matrix((N, N), dtype=np.float64)

    i_indices = candidate_pairs[:, 0]
    j_indices = candidate_pairs[:, 1]

    # Evaluate exact TRACLUS distances on candidate pairs
    segs_i = segments[i_indices]
    segs_j = segments[j_indices]

    # Vectorized distance between paired arrays of shape (K, 2, 2)
    # Pairwise segment distances with 1-to-1 matching
    s1 = segs_i[:, 0, :]
    e1 = segs_i[:, 1, :]
    v1 = e1 - s1
    len1 = np.linalg.norm(v1, axis=-1)

    s2 = segs_j[:, 0, :]
    e2 = segs_j[:, 1, :]
    v2 = e2 - s2
    len2 = np.linalg.norm(v2, axis=-1)

    from .distance import _compute_components_vectorized
    d_perp, d_par, d_theta = _compute_components_vectorized(
        s1=s1, e1=e1, v1=v1, len1=len1,
        s2=s2, e2=e2, v2=v2, len2=len2,
    )
    pair_distances = d_perp + d_par + d_theta

    # Filter pairs within eps
    valid = (pair_distances <= eps) & (pair_distances > 0.0)
    valid_i = i_indices[valid]
    valid_j = j_indices[valid]
    valid_d = pair_distances[valid]
    valid_w = np.exp(-gamma * (valid_d ** 2))

    # Symmetrize
    row_indices = np.concatenate([valid_i, valid_j])
    col_indices = np.concatenate([valid_j, valid_i])
    data = np.concatenate([valid_w, valid_w])

    W = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N), dtype=np.float64)
    return W


def build_normalized_laplacian(W: sp.spmatrix) -> sp.csr_matrix:
    """Construct the symmetric normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}.

    Safely handles disconnected components and zero-degree isolated vertices.

    Parameters
    ----------
    W : scipy.sparse matrix of shape (N, N)
        Symmetric affinity matrix.

    Returns
    -------
    L_norm : scipy.sparse.csr_matrix of shape (N, N)
        Symmetric normalized Laplacian with eigenvalues in [0, 2].
    """
    N = W.shape[0]
    if N == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64)
    if N == 1:
        return sp.csr_matrix([[0.0]], dtype=np.float64)

    W_csr = sp.csr_matrix(W, dtype=np.float64)

    # Degree vector: d_j = sum_k W_jk
    degrees = np.array(W_csr.sum(axis=1)).ravel()

    # Numerical protection for zero-degree vertices
    d_inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    nonzero = degrees > 1e-15
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')

    # Symmetric normalized affinity: S = D^{-1/2} W D^{-1/2}
    S = D_inv_sqrt @ W_csr @ D_inv_sqrt

    # Symmetric normalized Laplacian: L_norm = I - S
    I = sp.eye(N, format='csr', dtype=np.float64)
    L_norm = I - S

    return L_norm


class AffinityGraph:
    """Wrapper holding trajectory segments, sparse affinity matrix, and Laplacian."""

    def __init__(
        self,
        segments: np.ndarray,
        eps: float,
        sigma: Optional[float] = None,
        use_spatial_index: bool = True,
    ):
        self.segments = segments
        self.eps = eps
        self.sigma = sigma if sigma is not None else eps / 2.0
        self.n_nodes = len(segments)

        self.W = build_affinity_matrix(
            segments,
            eps=self.eps,
            sigma=self.sigma,
            use_spatial_index=use_spatial_index,
        )
        self.L_norm = build_normalized_laplacian(self.W)

    @property
    def degrees(self) -> np.ndarray:
        return np.array(self.W.sum(axis=1)).ravel()

    @property
    def isolated_nodes(self) -> np.ndarray:
        return np.where(self.degrees < 1e-15)[0]
