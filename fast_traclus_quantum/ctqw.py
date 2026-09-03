"""Continuous-Time Quantum Walk (CTQW) propagation and community grouping.

Replaces classical Laplacian eigendecomposition (O(N^3) diagonalization + k-means)
with unitary quantum walk propagation on the normalized Laplacian:
    |psi(t)> = exp(-i * L_norm * t) |psi(0)>
using sparse polynomial/Krylov approximations (scipy.sparse.linalg.expm_multiply).
Quantum wave interference isolates non-convex trajectory corridors without centroid distortion.
"""

from typing import Optional, Tuple, Union, List
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components


class CTQWClusterer:
    """Continuous-Time Quantum Walk (CTQW) community grouping clusterer.

    Parameters
    ----------
    t : float, optional
        Quantum walk evolution time. If None and adaptive_time=True, automatically
        computed from the Fiedler eigenvalue lambda_2: t ~ pi / (2 * sqrt(lambda_2)).
    tau : float, default=0.05
        Coherence threshold ratio for boundary segmentation.
        Edges with transition kernel K_jk >= tau * max(K_offdiag) are retained.
    min_samples : int, default=3
        Minimum number of segments required to form a valid community corridor.
        Components smaller than min_samples are discarded as noise (label -1).
    adaptive_time : bool, default=True
        Whether to calculate optimal tunneling evolution time adaptively.
    time_scale : float, default=1.0
        Multiplier applied to adaptive evolution time.
    batch_size : int, default=1000
        Maximum number of basis vectors processed concurrently during expm_multiply.
    """

    def __init__(
        self,
        t: Optional[float] = None,
        tau: float = 0.05,
        min_samples: int = 3,
        adaptive_time: bool = True,
        time_scale: float = 1.0,
        batch_size: int = 1000,
    ):
        self.t = t
        self.tau = tau
        self.min_samples = min_samples
        self.adaptive_time = adaptive_time
        self.time_scale = time_scale
        self.batch_size = batch_size

        # Fitted attributes
        self.labels_: Optional[np.ndarray] = None
        self.t_: Optional[float] = None
        self.U_: Optional[np.ndarray] = None
        self.P_: Optional[np.ndarray] = None
        self.K_: Optional[np.ndarray] = None
        self.contrast_ratio_: Optional[float] = None

    def compute_adaptive_time(self, L_norm: sp.spmatrix) -> float:
        """Compute characteristic quantum tunneling time: t ~ pi / (2 * sqrt(lambda_2)).

        Uses sparse Arnoldi/Lanczos eigensolver to find the smallest non-zero eigenvalue
        without dense O(N^3) eigendecomposition.
        """
        N = L_norm.shape[0]
        if N <= 2:
            return float(np.pi / 2.0)

        # Number of eigenvalues to query
        k_evals = min(8, N - 1)
        try:
            # Shift-invert or standard smallest algebraic (SA) on symmetric sparse matrix
            vals, _ = spla.eigsh(L_norm, k=k_evals, which='SA', tol=1e-4, maxiter=2000)
            vals = np.sort(np.real(vals))
            # Find the smallest positive eigenvalue strictly above numerical zero
            pos_vals = vals[vals > 1e-5]
            if len(pos_vals) > 0:
                lambda_2 = float(pos_vals[0])
            else:
                lambda_2 = 0.5
        except Exception:
            # Fallback if Lanczos fails to converge on degenerate graph
            lambda_2 = 0.5

        t = self.time_scale * (np.pi / (2.0 * np.sqrt(max(lambda_2, 1e-4))))
        return float(np.clip(t, 0.1, 50.0))

    def evolve(
        self,
        L_norm: sp.spmatrix,
        t: Optional[float] = None,
    ) -> np.ndarray:
        """Compute the unitary evolution operator U(t) = exp(-i * L_norm * t).

        Employs sparse polynomial / Krylov subspace approximation via
        scipy.sparse.linalg.expm_multiply, avoiding full O(N^3) dense diagonalization.

        Parameters
        ----------
        L_norm : scipy.sparse matrix of shape (N, N)
            Symmetric normalized Laplacian.
        t : float, optional
            Evolution time. If None, uses self.t or adaptive time.

        Returns
        -------
        U : ndarray of shape (N, N), complex128
            Unitary matrix U(t).
        """
        N = L_norm.shape[0]
        if N == 0:
            return np.empty((0, 0), dtype=np.complex128)
        if N == 1:
            return np.array([[np.exp(-1j * (t if t is not None else 1.0))]], dtype=np.complex128)

        if t is None:
            t = self.t if self.t is not None else (
                self.compute_adaptive_time(L_norm) if self.adaptive_time else float(np.pi / 2.0)
            )
        self.t_ = t

        # Sparse anti-Hermitian generator: A = -1j * t * L_norm
        A = (-1j * t * L_norm).tocsr()

        # Batch-wise computation over standard basis vectors
        if N <= self.batch_size:
            I_basis = np.eye(N, dtype=np.float64)
            U = spla.expm_multiply(A, I_basis)
        else:
            chunks = []
            for start_idx in range(0, N, self.batch_size):
                end_idx = min(start_idx + self.batch_size, N)
                chunk_len = end_idx - start_idx
                B_chunk = np.zeros((N, chunk_len), dtype=np.float64)
                for col_i in range(chunk_len):
                    B_chunk[start_idx + col_i, col_i] = 1.0
                U_chunk = spla.expm_multiply(A, B_chunk)
                chunks.append(U_chunk)
            U = np.hstack(chunks)

        return U

    def compute_transition_kernel(
        self,
        L_norm: sp.spmatrix,
        t: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the transition probability matrix P and symmetric quantum kernel K.

        P_jk = |<k| U(t) |j>|^2 = |U_kj|^2
        K_jk = 0.5 * (P_jk + P_kj)
        """
        U = self.evolve(L_norm, t=t)
        self.U_ = U

        # Transition probabilities: P_jk = |U_kj|^2
        P = np.abs(U) ** 2
        self.P_ = P

        # Symmetrized quantum walk kernel
        K = 0.5 * (P + P.T)
        self.K_ = K

        return P, K

    def fit_predict(
        self,
        L_norm: sp.spmatrix,
        W: Optional[sp.spmatrix] = None,
    ) -> np.ndarray:
        """Fit CTQW clusterer and return segment cluster assignments.

        Parameters
        ----------
        L_norm : scipy.sparse matrix of shape (N, N)
            Symmetric normalized Laplacian.
        W : scipy.sparse matrix of shape (N, N), optional
            Original affinity matrix (for degree checking).

        Returns
        -------
        labels : ndarray of shape (N,)
            Cluster labels (-1 denotes trajectory noise).
        """
        N = L_norm.shape[0]
        if N == 0:
            self.labels_ = np.empty(0, dtype=int)
            return self.labels_
        if N < self.min_samples:
            self.labels_ = np.full(N, -1, dtype=int)
            return self.labels_

        P, K = self.compute_transition_kernel(L_norm)

        # Off-diagonal transition intensity matrix
        K_off = K.copy()
        np.fill_diagonal(K_off, 0.0)

        max_transition = float(np.max(K_off)) if N > 1 else 0.0
        if max_transition < 1e-12:
            self.labels_ = np.full(N, -1, dtype=int)
            return self.labels_

        # Dynamic threshold based on coherence threshold tau
        threshold = self.tau * max_transition
        A_thresh = (K_off >= threshold).astype(np.float64)

        # Disconnect isolated nodes from original graph if available
        if W is not None:
            degrees = np.array(W.sum(axis=1)).ravel()
            isolated = degrees < 1e-12
            A_thresh[isolated, :] = 0.0
            A_thresh[:, isolated] = 0.0

        sparse_adj = sp.csr_matrix(A_thresh)
        n_components, raw_labels = connected_components(sparse_adj, directed=False)

        # Re-index clusters by size and discard components with size < min_samples as noise (-1)
        unique_labels, counts = np.unique(raw_labels, return_counts=True)
        valid_clusters = [lbl for lbl, cnt in zip(unique_labels, counts) if cnt >= self.min_samples]
        # Sort valid clusters by size descending
        valid_clusters.sort(key=lambda lbl: counts[list(unique_labels).index(lbl)], reverse=True)

        final_labels = np.full(N, -1, dtype=int)
        for new_id, old_lbl in enumerate(valid_clusters):
            final_labels[raw_labels == old_lbl] = new_id

        self.labels_ = final_labels

        # Compute interference contrast ratio if at least 2 clusters exist
        from .evaluation import interference_contrast_ratio
        try:
            self.contrast_ratio_ = interference_contrast_ratio(P, final_labels)
        except Exception:
            self.contrast_ratio_ = 0.0

        return final_labels

    def fit(self, L_norm: sp.spmatrix, W: Optional[sp.spmatrix] = None) -> "CTQWClusterer":
        """Fit CTQW clusterer on normalized Laplacian."""
        self.fit_predict(L_norm, W=W)
        return self

    def to_qiskit_circuit(
        self,
        subgraph_indices: Optional[List[int]] = None,
        time: Optional[float] = None,
    ):
        """Construct a native Qiskit QuantumCircuit representing the CTQW Hamiltonian evolution.

        Translates the normalized Laplacian (or a specified subgraph) into a SparsePauliOp
        observable and appends a PauliEvolutionGate for quantum processor transpilation.

        Parameters
        ----------
        subgraph_indices : list of int, optional
            Node subset to simulate on quantum registers. Defaults to first 4-8 nodes.
        time : float, optional
            Evolution parameter t. Defaults to self.t_.

        Returns
        -------
        qc : qiskit.circuit.QuantumCircuit
            Parameterized circuit ready for execution or statevector simulation.
        op : qiskit.quantum_info.SparsePauliOp
            Pauli operator representation of the graph Laplacian Hamiltonian.
        """
        import qiskit
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp

        if self.K_ is None or self.labels_ is None:
            raise RuntimeError("Clusterer must be fitted before generating Qiskit circuit.")

        # Default subgraph for standard qubit registers
        if subgraph_indices is None:
            # Pick the top corridor nodes
            valid = np.where(self.labels_ >= 0)[0]
            if len(valid) >= 4:
                subgraph_indices = valid[:4].tolist()
            else:
                subgraph_indices = list(range(min(4, len(self.labels_))))

        m = len(subgraph_indices)
        num_qubits = max(1, int(np.ceil(np.log2(m))))
        dim = 2 ** num_qubits

        # Extract subgraph matrix
        sub_L = np.zeros((dim, dim), dtype=np.float64)
        if self.U_ is not None:
            sub_P = self.P_[np.ix_(subgraph_indices, subgraph_indices)]
            # Construct effective local Hamiltonian H = -log(P) or Laplacian
            H_local = -np.log(np.maximum(sub_P, 1e-4))
            np.fill_diagonal(H_local, 0.0)
            H_local += np.diag(H_local.sum(axis=1))
            sub_L[:m, :m] = H_local

        # Symmetrize
        sub_L = 0.5 * (sub_L + sub_L.T)

        op = SparsePauliOp.from_operator(sub_L)
        t_val = time if time is not None else (self.t_ if self.t_ is not None else 1.0)

        qc = QuantumCircuit(num_qubits, name="CTQW_Corridor")
        gate = PauliEvolutionGate(op, time=t_val)
        qc.append(gate, list(range(num_qubits)))

        return qc, op
