"""Continuous-Time Quantum Walk (CTQW) unitary evolution via standardized Qiskit modules."""

from typing import Optional, Tuple, Union
import math
import numpy as np
import scipy.sparse.linalg as spla
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import HamiltonianGate, PauliEvolutionGate
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.primitives import StatevectorSampler


def compute_adaptive_walk_time(L_norm: np.ndarray, time_scale: float = 1.0) -> float:
    """Calculate the adaptive quantum walk evolution time: t ~ pi / (2 * sqrt(lambda_2)).

    Uses sparse eigensolver on the algebraic connectivity (Fiedler value) of the Laplacian.
    """
    N = L_norm.shape[0]
    if N <= 2:
        return float(np.pi / 2.0)

    try:
        # Query lowest eigenvalues
        k_evals = min(6, N - 1)
        vals = spla.eigsh(L_norm, k=k_evals, which="SA", return_eigenvectors=False)
        vals = np.sort(np.real(vals))
        pos_vals = vals[vals > 1e-4]
        lambda_2 = float(pos_vals[0]) if len(pos_vals) > 0 else 0.5
    except Exception:
        lambda_2 = 0.5

    t = time_scale * (np.pi / (2.0 * np.sqrt(max(lambda_2, 1e-4))))
    return float(np.clip(t, 0.2, 50.0))


def build_ctqw_circuit(
    L_padded: np.ndarray,
    sparse_pauli: SparsePauliOp,
    n_qubits: int,
    time_val: Optional[float] = None,
    trotter: bool = False,
    reps: int = 2,
) -> Tuple[QuantumCircuit, Optional[Parameter]]:
    """Construct a native Qiskit QuantumCircuit implementing U(t) = exp(-i * H * t).

    Parameters
    ----------
    L_padded : ndarray of shape (2^n, 2^n)
        Zero-padded normalized Laplacian matrix.
    sparse_pauli : qiskit.quantum_info.SparsePauliOp
        Pauli operator decomposition of the Hamiltonian.
    n_qubits : int
        Number of qubits in the register.
    time_val : float, optional
        Concrete evolution time t. If None, binds a symbolic Qiskit Parameter('t').
    trotter : bool, default=False
        If True, synthesizes via PauliEvolutionGate with LieTrotter product formula.
        If False, creates an exact HamiltonianGate.
    reps : int, default=2
        Number of Trotter synthesis repetitions.

    Returns
    -------
    qc : qiskit.circuit.QuantumCircuit
        Evolution circuit.
    t_param : qiskit.circuit.Parameter or None
        Symbolic parameter if time_val was None.
    """
    qc = QuantumCircuit(n_qubits, name="CTQW_Walk")
    t_param = None

    if time_val is None:
        t_param = Parameter("t")
        t_eval = t_param
    else:
        t_eval = float(time_val)

    if trotter:
        synthesis = LieTrotter(reps=reps)
        gate = PauliEvolutionGate(sparse_pauli, time=t_eval, synthesis=synthesis)
        qc.append(gate, list(range(n_qubits)))
    else:
        # Exact Hamiltonian unitary gate simulation
        gate = HamiltonianGate(data=L_padded, time=t_eval)
        qc.append(gate, list(range(n_qubits)))

    return qc, t_param


def simulate_ctqw_transitions(
    circuit: QuantumCircuit,
    n_qubits: int,
    N_nodes: int,
    time_val: Optional[float] = None,
    t_param: Optional[Parameter] = None,
) -> np.ndarray:
    """Compute the N x N transition probability matrix P_jk(t) = |<k| U(t) |j>|^2 using Statevector.

    Evolves localized basis states |j> across the Qiskit quantum circuit without
    custom NumPy matrix exponentiations.

    Parameters
    ----------
    circuit : qiskit.circuit.QuantumCircuit
        Evolution quantum circuit.
    n_qubits : int
        Number of qubits.
    N_nodes : int
        Original number of graph nodes before zero-padding.
    time_val : float, optional
        Concrete walk time to bind if circuit contains Parameter('t').
    t_param : Parameter, optional
        Symbolic parameter object in circuit.

    Returns
    -------
    P : ndarray of shape (N_nodes, N_nodes)
        Transition probability matrix where column j is the probability distribution
        starting from node j.
    """
    # Bind parameter if required
    if t_param is not None and time_val is not None:
        evol_circuit = circuit.assign_parameters({t_param: time_val})
    else:
        evol_circuit = circuit

    dim = 2**n_qubits
    P = np.zeros((N_nodes, N_nodes), dtype=np.float64)

    for j in range(N_nodes):
        # 1. Localized initial state |j> in computational basis
        psi_0 = Statevector.from_int(j, dims=dim)

        # 2. Unitary time evolution under the circuit
        psi_t = psi_0.evolve(evol_circuit)

        # 3. Transition probability distribution P_kj = |<k|psi_t>|^2
        probs = psi_t.probabilities()
        P[:, j] = probs[:N_nodes]

    # Re-normalize columns to account for any negligible leakage into padded dimensions
    col_sums = np.sum(P, axis=0, keepdims=True)
    P = P / np.maximum(col_sums, 1e-15)

    return P


def sample_ctqw_transitions(
    circuit: QuantumCircuit,
    n_qubits: int,
    N_nodes: int,
    source_node: int,
    shots: int = 1024,
    time_val: Optional[float] = None,
    t_param: Optional[Parameter] = None,
) -> np.ndarray:
    """Sample transition distribution using the standardized Qiskit StatevectorSampler primitive.

    Parameters
    ----------
    circuit : QuantumCircuit
        Evolution circuit.
    n_qubits : int
        Number of qubits.
    N_nodes : int
        Number of graph nodes.
    source_node : int
        Source node j.
    shots : int, default=1024
        Number of measurement shots.

    Returns
    -------
    prob_dist : ndarray of shape (N_nodes,)
        Sampled empirical probability distribution over nodes.
    """
    if t_param is not None and time_val is not None:
        qc = circuit.assign_parameters({t_param: time_val})
    else:
        qc = circuit.copy()

    # Initialize state at source_node
    full_qc = QuantumCircuit(n_qubits)
    binary = format(source_node, f"0{n_qubits}b")[::-1]  # Qiskit little-endian
    for q, bit in enumerate(binary):
        if bit == "1":
            full_qc.x(q)

    full_qc.compose(qc, inplace=True)
    full_qc.measure_all()

    sampler = StatevectorSampler()
    job = sampler.run([(full_qc)], shots=shots)
    pub_res = job.result()[0]
    counts = pub_res.data.meas.get_counts()

    probs = np.zeros(N_nodes, dtype=np.float64)
    for bitstring, count in counts.items():
        # Reverse bitstring for standard integer index
        node_idx = int(bitstring, 2)
        if node_idx < N_nodes:
            probs[node_idx] += count / shots

    total = np.sum(probs)
    if total > 0:
        probs /= total
    return probs
