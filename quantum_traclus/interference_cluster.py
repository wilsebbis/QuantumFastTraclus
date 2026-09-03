"""Quantum wave interference corridor grouping and QuantumFastTRACLUS pipeline."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from core.representative import generate_representative_trajectory
from fast_traclus.vectorized_mdl import partition_trajectories_fast
from fast_traclus.distance_matrix import compute_distance_matrix
from .laplacian_builder import build_laplacian_operators
from .ctqw_evolution import (
    build_ctqw_circuit,
    simulate_ctqw_transitions,
    compute_adaptive_walk_time,
)


def extract_ctqw_corridors(
    P: np.ndarray,
    traj_ids: np.ndarray,
    tau: float = 0.05,
    min_lines: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract non-convex community corridors from the quantum transition kernel.

    Parameters
    ----------
    P : ndarray of shape (N, N)
        Transition probability matrix P_jk(t) = |<k| U(t) |j>|^2.
    traj_ids : ndarray of shape (N,)
        Trajectory ID of each line segment.
    tau : float, default=0.05
        Coherence threshold ratio: retains edges with K_jk >= tau * max(K_offdiag).
    min_lines : int, default=3
        Minimum segment count and trajectory cardinality threshold.

    Returns
    -------
    labels : ndarray of shape (N,)
        Cluster assignments (-1 indicates noise).
    K : ndarray of shape (N, N)
        Symmetric quantum interference kernel K = 0.5 * (P + P^T).
    """
    N = P.shape[0]
    if N == 0:
        return np.empty(0, dtype=int), np.empty((0, 0), dtype=np.float64)

    # 1. Symmetric quantum transition kernel
    K = 0.5 * (P + P.T)

    # 2. Dynamic thresholding on off-diagonal transition intensities
    K_off = K.copy()
    np.fill_diagonal(K_off, 0.0)
    max_transition = float(np.max(K_off)) if N > 1 else 0.0

    if max_transition < 1e-12:
        return np.full(N, -1, dtype=int), K

    threshold = tau * max_transition
    A_thresh = (K_off >= threshold).astype(np.float64)

    # 3. Connected components extraction
    sparse_adj = sp.csr_matrix(A_thresh)
    n_components, raw_labels = connected_components(sparse_adj, directed=False)

    # 4. Filter by min_lines and Trajectory Cardinality Filter |PTR(C)| >= min_lines
    labels = raw_labels.copy()
    unique_c = np.unique(raw_labels)

    for c in unique_c:
        mask = raw_labels == c
        count = np.sum(mask)
        num_unique_trajs = len(np.unique(traj_ids[mask]))
        if count < min_lines or num_unique_trajs < min_lines:
            labels[mask] = -1

    # Re-index surviving clusters to 0, 1, ..., K-1 in decreasing size order
    surviving = [c for c in np.unique(labels) if c >= 0]
    surviving.sort(key=lambda c: np.sum(labels == c), reverse=True)

    final_labels = np.full(N, -1, dtype=int)
    for new_id, old_c in enumerate(surviving):
        final_labels[labels == old_c] = new_id

    return final_labels, K


class QuantumFastTRACLUS:
    """Fast-TRACLUS trajectory clustering with Qiskit Continuous-Time Quantum Walk grouping."""

    def __init__(
        self,
        eps: float = 5.0,
        min_lines: int = 3,
        t_walk: Optional[float] = None,
        tau: float = 0.05,
        sigma: Optional[float] = None,
        trotter: bool = False,
        reps: int = 2,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        gamma: float = 1.0,
    ):
        self.eps = eps
        self.min_lines = min_lines
        self.t_walk = t_walk
        self.tau = tau
        self.sigma = sigma
        self.trotter = trotter
        self.reps = reps
        self.weights = weights
        self.gamma = gamma

        self.segments_: Optional[np.ndarray] = None
        self.traj_ids_: Optional[np.ndarray] = None
        self.seg_ids_: Optional[np.ndarray] = None
        self.distance_matrix_: Optional[np.ndarray] = None
        self.L_norm_: Optional[np.ndarray] = None
        self.t_used_: Optional[float] = None
        self.P_: Optional[np.ndarray] = None
        self.K_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.representative_trajectories_: Dict[int, np.ndarray] = {}

    def fit(self, trajectories: List[Union[np.ndarray, list]]) -> "QuantumFastTRACLUS":
        """Execute Fast-TRACLUS partitioning, Qiskit Hamiltonian construction, and CTQW grouping."""
        # 1. Vectorized MDL Partitioning
        self.segments_, self.traj_ids_, self.seg_ids_ = partition_trajectories_fast(trajectories)
        N = len(self.segments_)
        if N == 0:
            self.labels_ = np.empty(0, dtype=int)
            return self

        # 2. Broadcasted N x N Distance Matrix
        self.distance_matrix_ = compute_distance_matrix(self.segments_, weights=self.weights)

        # 3. Laplacian and Qiskit Operator Construction
        L_norm, L_padded, op, sparse_pauli, n_qubits = build_laplacian_operators(
            self.distance_matrix_, eps=self.eps, sigma=self.sigma
        )
        self.L_norm_ = L_norm

        # 4. Adaptive Walk Time
        if self.t_walk is None:
            self.t_used_ = compute_adaptive_walk_time(L_norm)
        else:
            self.t_used_ = float(self.t_walk)

        # 5. Build Qiskit Quantum Circuit and Simulate
        qc, t_param = build_ctqw_circuit(
            L_padded=L_padded,
            sparse_pauli=sparse_pauli,
            n_qubits=n_qubits,
            time_val=self.t_used_,
            trotter=self.trotter,
            reps=self.reps,
        )

        self.P_ = simulate_ctqw_transitions(
            circuit=qc,
            n_qubits=n_qubits,
            N_nodes=N,
            time_val=self.t_used_,
            t_param=t_param,
        )

        # 6. Quantum Interference Corridor Extraction
        self.labels_, self.K_ = extract_ctqw_corridors(
            P=self.P_,
            traj_ids=self.traj_ids_,
            tau=self.tau,
            min_lines=self.min_lines,
        )

        # 7. Sweep-Line Representative Trajectory Extraction
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
