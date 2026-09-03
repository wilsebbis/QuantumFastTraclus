"""Original TRACLUS line-segment DBSCAN with trajectory cardinality filtering."""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from core.distance import iterative_segment_distance
from core.representative import generate_representative_trajectory
from .iterative_mdl import partition_trajectories_traclus


def line_segment_dbscan(
    segments: np.ndarray,
    traj_ids: np.ndarray,
    eps: float,
    min_lines: int,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Execute line-segment DBSCAN clustering with the Trajectory Cardinality Filter (Lee et al., 2007).

    Parameters
    ----------
    segments : ndarray of shape (N, 2, 2)
        Collection of line segments.
    traj_ids : ndarray of shape (N,)
        Parent trajectory ID for each segment.
    eps : float
        Neighborhood radius epsilon.
    min_lines : int
        Minimum segment count (MinLns) and minimum unique trajectory cardinality.
    weights : tuple of (w_perp, w_par, w_theta), default=(1.0, 1.0, 1.0)
        Weights for distance components.

    Returns
    -------
    labels : ndarray of shape (N,)
        Cluster assignments (-1 denotes noise).
    """
    N = len(segments)
    if N == 0:
        return np.empty(0, dtype=int)

    # UNCLASSIFIED = -2, NOISE = -1, CLUSTER >= 0
    labels = np.full(N, -2, dtype=int)
    cluster_id = 0

    def region_query(idx: int) -> List[int]:
        """Sequential loop traversal to compute epsilon-neighborhood."""
        neighbors = []
        target = segments[idx]
        for j in range(N):
            d = iterative_segment_distance(target, segments[j], weights=weights)
            if d <= eps:
                neighbors.append(j)
        return neighbors

    for i in range(N):
        if labels[i] != -2:
            continue

        neighbors = region_query(i)
        if len(neighbors) < min_lines:
            labels[i] = -1  # Mark as noise
            continue

        # Expand cluster
        labels[i] = cluster_id
        seed_set = list(neighbors)
        # Remove target node from seed queue if present
        seed_set = [idx for idx in seed_set if idx != i]

        ptr = 0
        while ptr < len(seed_set):
            curr = seed_set[ptr]
            ptr += 1

            if labels[curr] == -1:
                labels[curr] = cluster_id
            elif labels[curr] == -2:
                labels[curr] = cluster_id
                curr_neighbors = region_query(curr)
                if len(curr_neighbors) >= min_lines:
                    for n_idx in curr_neighbors:
                        if n_idx not in seed_set and labels[n_idx] in (-1, -2):
                            seed_set.append(n_idx)

        cluster_id += 1

    # --- Trajectory Cardinality Filter (Lee et al., 2007) ---
    # Prune any cluster C if |PTR(C)| < MinLns (number of unique trajectories < MinLns)
    unique_clusters = np.unique(labels[labels >= 0])
    for c in unique_clusters:
        c_mask = labels == c
        c_trajs = traj_ids[c_mask]
        num_unique_trajs = len(np.unique(c_trajs))
        if num_unique_trajs < min_lines:
            labels[c_mask] = -1  # Prune cluster to noise

    # Re-index remaining clusters to 0, 1, ..., K-1
    surviving = np.unique(labels[labels >= 0])
    final_labels = np.full(N, -1, dtype=int)
    for new_id, old_id in enumerate(surviving):
        final_labels[labels == old_id] = new_id

    return final_labels


class OriginalTRACLUS:
    """Original TRACLUS trajectory clustering pipeline (Lee, Han, & Whang, 2007)."""

    def __init__(
        self,
        eps: float = 5.0,
        min_lines: int = 3,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        gamma: float = 1.0,
    ):
        self.eps = eps
        self.min_lines = min_lines
        self.weights = weights
        self.gamma = gamma

        self.segments_: Optional[np.ndarray] = None
        self.traj_ids_: Optional[np.ndarray] = None
        self.seg_ids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.representative_trajectories_: Dict[int, np.ndarray] = {}

    def fit(self, trajectories: List[Union[np.ndarray, list]]) -> "OriginalTRACLUS":
        """Execute Original TRACLUS partitioning, line-DBSCAN grouping, and representative extraction."""
        # 1. Iterative MDL Partitioning
        self.segments_, self.traj_ids_, self.seg_ids_ = partition_trajectories_traclus(trajectories)
        if len(self.segments_) == 0:
            self.labels_ = np.empty(0, dtype=int)
            return self

        # 2. Line-Segment DBSCAN with Cardinality Filter
        self.labels_ = line_segment_dbscan(
            self.segments_,
            self.traj_ids_,
            eps=self.eps,
            min_lines=self.min_lines,
            weights=self.weights,
        )

        # 3. Sweep-Line Representative Trajectory Extraction
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
