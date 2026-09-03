"""Decoupled modular clustering framework for Fast-TRACLUS.

Provides interchangeable clustering backends on top of the precomputed distance matrix:
1. DBSCAN (with trajectory cardinality filtering)
2. OPTICS
3. HDBSCAN
4. AgglomerativeClustering
5. Classical SpectralClustering (normalized Laplacian O(N^3) eigendecomposition + k-means)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
)

from core.representative import generate_representative_trajectory
from .vectorized_mdl import partition_trajectories_fast
from .distance_matrix import compute_distance_matrix


def modular_cluster_segments(
    distance_matrix: np.ndarray,
    traj_ids: np.ndarray,
    backend: str = "dbscan",
    eps: float = 5.0,
    min_lines: int = 3,
    n_clusters: Optional[int] = None,
    sigma: Optional[float] = None,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster line segments using the selected modular backend on the distance matrix.

    Parameters
    ----------
    distance_matrix : ndarray of shape (N, N)
        Precomputed pairwise line segment distance matrix.
    traj_ids : ndarray of shape (N,)
        Trajectory ID for each segment (used for cardinality filtering).
    backend : str, default='dbscan'
        Clustering algorithm: 'dbscan', 'optics', 'hdbscan', 'agglomerative', 'spectral'.
    eps : float
        Neighborhood radius parameter for DBSCAN / distance thresholds.
    min_lines : int
        MinLns parameter (min_samples) and trajectory cardinality threshold.
    n_clusters : int, optional
        Target cluster count for agglomerative or spectral clustering.
    sigma : float, optional
        Bandwidth parameter for spectral affinity matrix W = exp(-D^2 / (2*sigma^2)).
    random_state : int, default=42
        Random seed for k-means initialization in spectral clustering.

    Returns
    -------
    labels : ndarray of shape (N,)
        Cluster assignments (-1 denotes noise).
    """
    N = distance_matrix.shape[0]
    if N == 0:
        return np.empty(0, dtype=int)
    if N < min_lines:
        return np.full(N, -1, dtype=int)

    backend_lower = backend.lower()

    if backend_lower == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_lines, metric="precomputed")
        labels = clusterer.fit_predict(distance_matrix)

    elif backend_lower == "optics":
        clusterer = OPTICS(max_eps=eps, min_samples=min_lines, metric="precomputed")
        labels = clusterer.fit_predict(distance_matrix)

    elif backend_lower == "hdbscan":
        clusterer = HDBSCAN(min_cluster_size=min_lines, metric="precomputed")
        labels = clusterer.fit_predict(distance_matrix)

    elif backend_lower == "agglomerative":
        k = n_clusters if n_clusters is not None else 2
        k = max(1, min(k, N))
        clusterer = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        labels = clusterer.fit_predict(distance_matrix)

    elif backend_lower == "spectral":
        # Classical Spectral Clustering baseline
        # 1. Affinity matrix W with Gaussian radial basis
        s_val = sigma if sigma is not None else max(eps / 2.0, 1e-4)
        gamma = 1.0 / (2.0 * s_val**2)
        W = np.exp(-gamma * (distance_matrix**2))
        W[distance_matrix > eps] = 0.0
        np.fill_diagonal(W, 0.0)

        # 2. Symmetric normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
        degrees = np.sum(W, axis=1)
        d_inv_sqrt = np.zeros_like(degrees)
        nz = degrees > 1e-12
        d_inv_sqrt[nz] = 1.0 / np.sqrt(degrees[nz])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        S = D_inv_sqrt @ W @ D_inv_sqrt
        L_norm = np.eye(N) - S

        # 3. Eigendecomposition: find lowest k eigenvectors (O(N^3))
        k = n_clusters if n_clusters is not None else 2
        k = max(2, min(k, N - 1))
        evals, evecs = np.linalg.eigh(L_norm)
        U_k = evecs[:, :k]

        # Row normalization (Ng-Jordan-Weiss heuristic)
        row_norms = np.linalg.norm(U_k, axis=1, keepdims=True)
        safe_norms = np.maximum(row_norms, 1e-12)
        Y_k = U_k / safe_norms

        # 4. Euclidean k-means on projected subspace
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(Y_k)

        # Mark isolated nodes (degree == 0) as noise
        labels[degrees < 1e-12] = -1

    else:
        raise ValueError(f"Unknown clustering backend: '{backend}'. Choose from 'dbscan', 'optics', 'hdbscan', 'agglomerative', 'spectral'.")

    # --- Apply Trajectory Cardinality Filter across all backends ---
    unique_clusters = np.unique(labels[labels >= 0])
    for c in unique_clusters:
        c_mask = labels == c
        c_trajs = traj_ids[c_mask]
        if len(np.unique(c_trajs)) < min_lines:
            labels[c_mask] = -1

    # Re-index clusters to clean 0, 1, ..., K-1
    surviving = np.unique(labels[labels >= 0])
    final_labels = np.full(N, -1, dtype=int)
    for new_id, old_id in enumerate(surviving):
        final_labels[labels == old_id] = new_id

    return final_labels


class FastTRACLUS:
    """Fast-TRACLUS trajectory clustering pipeline with decoupled modular backends."""

    def __init__(
        self,
        eps: float = 5.0,
        min_lines: int = 3,
        backend: str = "dbscan",
        n_clusters: Optional[int] = None,
        sigma: Optional[float] = None,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        gamma: float = 1.0,
        random_state: int = 42,
    ):
        self.eps = eps
        self.min_lines = min_lines
        self.backend = backend
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.weights = weights
        self.gamma = gamma
        self.random_state = random_state

        self.segments_: Optional[np.ndarray] = None
        self.traj_ids_: Optional[np.ndarray] = None
        self.seg_ids_: Optional[np.ndarray] = None
        self.distance_matrix_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.representative_trajectories_: Dict[int, np.ndarray] = {}

    def fit(self, trajectories: List[Union[np.ndarray, list]]) -> "FastTRACLUS":
        """Execute Fast-TRACLUS partitioning, distance matrix broadcast, and modular clustering."""
        # 1. Vectorized MDL Partitioning
        self.segments_, self.traj_ids_, self.seg_ids_ = partition_trajectories_fast(trajectories)
        N = len(self.segments_)
        if N == 0:
            self.labels_ = np.empty(0, dtype=int)
            return self

        # 2. Broadcasted N x N Distance Matrix
        self.distance_matrix_ = compute_distance_matrix(self.segments_, weights=self.weights)

        # 3. Modular Clustering Backend
        self.labels_ = modular_cluster_segments(
            distance_matrix=self.distance_matrix_,
            traj_ids=self.traj_ids_,
            backend=self.backend,
            eps=self.eps,
            min_lines=self.min_lines,
            n_clusters=self.n_clusters,
            sigma=self.sigma,
            random_state=self.random_state,
        )

        # 4. Sweep-Line Representative Trajectory Extraction
        self._build_representatives()
        return self

    def fit_predict(
        self, trajectories: List[Union[np.ndarray, list]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(trajectories)
        return self.segments_, self.labels_

    def _build_representatives(self) -> None:
        self.representative_trajectories_.clear()
        if self.labels_ is None or self.segments_ is None:
            return

        unique_clusters = np.unique(self.labels_[self.labels_ >= 0])
        for c in unique_clusters:
            c_segs = self.segments_[self.labels_ == c]
            rep = generate_representative_trajectory(c_segs, min_lines=self.min_lines, gamma=self.gamma)
            self.representative_trajectories_[int(c)] = rep

    def get_representative_trajectories(self) -> Dict[int, np.ndarray]:
        return self.representative_trajectories_
