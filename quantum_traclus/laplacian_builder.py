"""Graph Laplacian and Qiskit Hamiltonian operator construction."""

from typing import Optional, Tuple
import math
import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import Operator, SparsePauliOp


def build_laplacian_operators(
    distance_matrix: np.ndarray,
    eps: float,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Operator, SparsePauliOp, int]:
    """Construct normalized Laplacian and translate into standard Qiskit operators.

    Parameters
    ----------
    distance_matrix : ndarray of shape (N, N)
        Pairwise segment distance matrix.
    eps : float
        Spatial cutoff threshold epsilon.
    sigma : float, optional
        Bandwidth for Gaussian affinity weights. Defaults to eps / 2.0.

    Returns
    -------
    L_norm : ndarray of shape (N, N)
        Normalized Laplacian matrix.
    L_padded : ndarray of shape (2^n, 2^n)
        Zero-padded Laplacian matrix matching the n-qubit Hilbert space.
    op : qiskit.quantum_info.Operator
        Unitary/Hermitian operator representation.
    sparse_pauli : qiskit.quantum_info.SparsePauliOp
        Decomposed Pauli string expansion.
    n_qubits : int
        Number of qubits required n = ceil(log2(N)).
    """
    N = distance_matrix.shape[0]
    if N == 0:
        raise ValueError("Cannot build Laplacian operators for empty segment collection.")

    s_val = sigma if sigma is not None else max(eps / 2.0, 1e-4)
    gamma = 1.0 / (2.0 * s_val**2)

    # 1. Affinity matrix W
    W = np.exp(-gamma * (distance_matrix**2))
    W[distance_matrix > eps] = 0.0
    np.fill_diagonal(W, 0.0)

    # 2. Symmetric normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
    degrees = np.sum(W, axis=1)
    d_inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    nonzero = degrees > 1e-12
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)

    S = D_inv_sqrt @ W @ D_inv_sqrt
    L_norm = np.eye(N, dtype=np.float64) - S

    # 3. Map into n-qubit Hilbert space: n = ceil(log2(N))
    n_qubits = max(1, int(math.ceil(math.log2(max(N, 2)))))
    dim = 2**n_qubits

    L_padded = np.eye(dim, dtype=np.float64)
    L_padded[:N, :N] = L_norm

    # 4. Standard Qiskit Operator representations
    op = Operator(L_padded)
    sparse_pauli = SparsePauliOp.from_operator(op)

    return L_norm, L_padded, op, sparse_pauli, n_qubits
