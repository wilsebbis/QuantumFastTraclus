"""End-to-end pipeline orchestrators for Fast-TRACLUS.

Includes:
1. FastTRACLUSQuantum: End-to-end Fast-TRACLUS leveraging Continuous-Time Quantum Walk (CTQW).
2. FastTRACLUSSpectralBaseline: Classical Fast-TRACLUS baseline with Laplacian eigendecomposition + k-means.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans

from .distance import pairwise_segment_distances
from .segmentation import partition_trajectories
from .graph import build_affinity_matrix, build_normalized_laplacian, AffinityGraph
from .ctqw import CTQWClusterer


def _compute_cluster_representative_trajectory(
    segments: np.ndarray,
    min_samples: int = 3,
) -> np.ndarray:
    """Extract a smooth representative trajectory from a cluster of segments.

    Projects segment endpoints onto the cluster's principal flow axis, evaluates
    average transversal positions across intervals, and reconstructs the centerline.
    """
    if len(segments) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(segments) == 1:
        return segments[0]

    # Compute mean directional unit vector
    vectors = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe_lengths = np.maximum(lengths, 1e-12)
    unit_vectors = vectors / safe_lengths
    mean_dir = np.mean(unit_vectors, axis=0)
    norm_mean_dir = np.linalg.norm(mean_dir)

    if norm_mean_dir < 1e-6:
        # Indeterminate direction: fall back to PCA on all points
        all_pts = segments.reshape(-1, 2)
        mean_pt = np.mean(all_pts, axis=0)
        cov = np.cov((all_pts - mean_pt).T)
        evals, evecs = np.linalg.eigh(cov)
        u_axis = evecs[:, -1]
    else:
        u_axis = mean_dir / norm_mean_dir

    # Normal vector
    u_norm = np.array([-u_axis[1], u_axis[0]])

    # Project start and end points onto u_axis and u_norm
    all_starts = segments[:, 0, :]
    all_ends = segments[:, 1, :]

    # Origin at centroid
    origin = np.mean(0.5 * (all_starts + all_ends), axis=0)

    s_rel = all_starts - origin
    e_rel = all_ends - origin

    t_s = s_rel[:, 0] * u_axis[0] + s_rel[:, 1] * u_axis[1]
    t_e = e_rel[:, 0] * u_axis[0] + e_rel[:, 1] * u_axis[1]

    n_s = s_rel[:, 0] * u_norm[0] + s_rel[:, 1] * u_norm[1]
    n_e = e_rel[:, 0] * u_norm[0] + e_rel[:, 1] * u_norm[1]

    # Combine projection samples
    t_vals = np.concatenate([t_s, t_e])
    n_vals = np.concatenate([n_s, n_e])

    sort_idx = np.argsort(t_vals)
    t_sorted = t_vals[sort_idx]
    n_sorted = n_vals[sort_idx]

    # Discretize along axis into representative points
    n_bins = max(3, min(20, len(segments)))
    bin_edges = np.linspace(t_sorted[0], t_sorted[-1], n_bins + 1)

    rep_pts = []
    for b_idx in range(n_bins):
        mask = (t_sorted >= bin_edges[b_idx]) & (t_sorted <= bin_edges[b_idx + 1])
        if np.any(mask):
            mean_t = np.mean(t_sorted[mask])
            mean_n = np.mean(n_sorted[mask])
            pt_orig = origin + mean_t * u_axis + mean_n * u_norm
            rep_pts.append(pt_orig)

    if len(rep_pts) < 2:
        return segments[0]

    return np.asarray(rep_pts, dtype=np.float64)


class FastTRACLUSQuantum:
    """Fast-TRACLUS trajectory clustering with Continuous-Time Quantum Walk (CTQW) grouping.

    Parameters
    ----------
    eps : float, default=5.0
        Affinity distance threshold in spatial metric space.
    min_samples : int, default=3
        Minimum number of segments per cluster.
    t_walk : float, optional
        Quantum walk duration. If None, computed adaptively from the Fiedler value.
    sigma : float, optional
        Bandwidth parameter for Gaussian affinity weights. Defaults to eps / 2.0.
    tau : float, default=0.05
        Coherence threshold ratio for quantum kernel community grouping.
    adaptive_time : bool, default=True
        Whether to calculate optimal tunneling evolution time adaptively.
    use_spatial_index : bool, default=True
        Whether to use KD-Tree candidate pruning for large datasets.
    min_segment_len : float, default=1e-4
        Threshold for trajectory point simplification.
    """

    def __init__(
        self,
        eps: float = 5.0,
        min_samples: int = 3,
        t_walk: Optional[float] = None,
        sigma: Optional[float] = None,
        tau: float = 0.05,
        adaptive_time: bool = True,
        use_spatial_index: bool = True,
        min_segment_len: float = 1e-4,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.t_walk = t_walk
        self.sigma = sigma if sigma is not None else eps / 2.0
        self.tau = tau
        self.adaptive_time = adaptive_time
        self.use_spatial_index = use_spatial_index
        self.min_segment_len = min_segment_len

        # Pipeline state
        self.segments_: Optional[np.ndarray] = None
        self.traj_ids_: Optional[np.ndarray] = None
        self.seg_ids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.affinity_graph_: Optional[AffinityGraph] = None
        self.clusterer_: Optional[CTQWClusterer] = None
        self.representative_trajectories_: Dict[int, np.ndarray] = {}

    def fit(self, trajectories: List[Union[np.ndarray, list]]) -> "FastTRACLUSQuantum":
        """Execute end-to-end Fast-TRACLUS CTQW pipeline on trajectories."""
        # 1. Vectorized MDL Partitioning
        self.segments_, self.traj_ids_, self.seg_ids_ = partition_trajectories(
            trajectories, min_length=self.min_segment_len
        )

        N = len(self.segments_)
        if N == 0:
            self.labels_ = np.empty(0, dtype=int)
            return self

        # 2. Sparse Affinity Graph & Normalized Laplacian Construction
        self.affinity_graph_ = AffinityGraph(
            self.segments_,
            eps=self.eps,
            sigma=self.sigma,
            use_spatial_index=self.use_spatial_index,
        )

        # 3. CTQW Propagation & Community Grouping
        self.clusterer_ = CTQWClusterer(
            t=self.t_walk,
            tau=self.tau,
            min_samples=self.min_samples,
            adaptive_time=self.adaptive_time,
        )
        self.labels_ = self.clusterer_.fit_predict(
            self.affinity_graph_.L_norm,
            W=self.affinity_graph_.W,
        )

        # 4. Generate Representative Trajectories
        self._build_representative_trajectories()

        return self

    def fit_predict(
        self, trajectories: List[Union[np.ndarray, list]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit pipeline and return extracted segments and their cluster assignments."""
        self.fit(trajectories)
        return self.segments_, self.labels_

    def _build_representative_trajectories(self) -> None:
        """Extract representative trajectory for each detected corridor."""
        self.representative_trajectories_.clear()
        if self.labels_ is None or self.segments_ is None:
            return

        unique_clusters = np.unique(self.labels_[self.labels_ >= 0])
        for c_id in unique_clusters:
            c_segs = self.segments_[self.labels_ == c_id]
            rep_traj = _compute_cluster_representative_trajectory(
                c_segs, min_samples=self.min_samples
            )
            self.representative_trajectories_[int(c_id)] = rep_traj

    def get_representative_trajectories(self) -> Dict[int, np.ndarray]:
        """Return dictionary mapping cluster_id to representative 2D trajectory."""
        return self.representative_trajectories_


class FastTRACLUSSpectralBaseline:
    """Classical Fast-TRACLUS baseline with Laplacian eigendecomposition + k-means.

    Serves as the head-to-head comparison model to demonstrate the centroid distortion
    and O(N^3) bottleneck of classical spectral clustering on manifold trajectory corridors.
    """

    def __init__(
        self,
        eps: float = 5.0,
        min_samples: int = 3,
        n_clusters: Optional[int] = None,
        sigma: Optional[float] = None,
        min_segment_len: float = 1e-4,
        random_state: int = 42,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.sigma = sigma if sigma is not None else eps / 2.0
        self.min_segment_len = min_segment_len
        self.random_state = random_state

        self.segments_: Optional[np.ndarray] = None
        self.traj_ids_: Optional[np.ndarray] = None
        self.seg_ids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.affinity_graph_: Optional[AffinityGraph] = None

    def fit(self, trajectories: List[Union[np.ndarray, list]]) -> "FastTRACLUSSpectralBaseline":
        """Execute classical spectral clustering baseline."""
        # 1. Partition trajectories using identical MDL
        self.segments_, self.traj_ids_, self.seg_ids_ = partition_trajectories(
            trajectories, min_length=self.min_segment_len
        )

        N = len(self.segments_)
        if N == 0:
            self.labels_ = np.empty(0, dtype=int)
            return self

        # 2. Build identical normalized Laplacian
        self.affinity_graph_ = AffinityGraph(
            self.segments_,
            eps=self.eps,
            sigma=self.sigma,
            use_spatial_index=False,
        )
        L_norm = self.affinity_graph_.L_norm

        # Determine number of clusters k
        k = self.n_clusters if self.n_clusters is not None else 2
        k = max(2, min(k, N - 1))

        # 3. Classical Spectral Clustering: Eigendecomposition of L_norm
        # Finds k lowest eigenvectors
        try:
            # Note: Classical baseline performs spectral decomposition
            if N <= 1000:
                dense_L = L_norm.toarray()
                evals, evecs = np.linalg.eigh(dense_L)
                U_k = evecs[:, :k]
            else:
                evals, evecs = spla.eigsh(L_norm, k=k, which='SM')
                U_k = evecs
        except Exception:
            self.labels_ = np.full(N, -1, dtype=int)
            return self

        # Row normalization (Ng-Jordan-Weiss heuristic)
        row_norms = np.linalg.norm(U_k, axis=1, keepdims=True)
        safe_norms = np.maximum(row_norms, 1e-12)
        Y_k = U_k / safe_norms

        # 4. Euclidean k-means on projected subspace
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        raw_labels = kmeans.fit_predict(Y_k)

        # Mark isolated / low-degree nodes as noise (-1)
        degrees = np.array(self.affinity_graph_.W.sum(axis=1)).ravel()
        isolated = degrees < 1e-12

        # Discard small clusters (< min_samples)
        unique_labels, counts = np.unique(raw_labels, return_counts=True)
        final_labels = raw_labels.copy()
        for lbl, count in zip(unique_labels, counts):
            if count < self.min_samples:
                final_labels[final_labels == lbl] = -1

        final_labels[isolated] = -1
        self.labels_ = final_labels

        return self

    def fit_predict(
        self, trajectories: List[Union[np.ndarray, list]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(trajectories)
        return self.segments_, self.labels_
