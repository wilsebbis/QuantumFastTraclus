"""Tests for Qiskit-native Continuous-Time Quantum Walk (CTQW) Fast-TRACLUS."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import HamiltonianGate, PauliEvolutionGate

from quantum_traclus.laplacian_builder import build_laplacian_operators
from quantum_traclus.ctqw_evolution import (
    build_ctqw_circuit,
    simulate_ctqw_transitions,
    sample_ctqw_transitions,
    compute_adaptive_walk_time,
)
from quantum_traclus.interference_cluster import extract_ctqw_corridors, QuantumFastTRACLUS


def test_build_laplacian_operators():
    # 4 segments forming 2 disconnected pairs
    D = np.array([
        [0.0, 1.0, 50.0, 50.0],
        [1.0, 0.0, 50.0, 50.0],
        [50.0, 50.0, 0.0, 1.0],
        [50.0, 50.0, 1.0, 0.0],
    ])
    L_norm, L_padded, op, sparse_pauli, n_qubits = build_laplacian_operators(D, eps=5.0)

    assert n_qubits == 2
    assert L_norm.shape == (4, 4)
    assert L_padded.shape == (4, 4)
    assert isinstance(op, Operator)
    assert isinstance(sparse_pauli, SparsePauliOp)

    # Check eigenvalues of L_norm in [0, 2]
    evals = np.linalg.eigvalsh(L_norm)
    assert np.all(evals >= -1e-12)
    assert np.all(evals <= 2.0 + 1e-12)


def test_build_ctqw_circuit_and_evolution():
    D = np.array([
        [0.0, 1.0, 20.0],
        [1.0, 0.0, 20.0],
        [20.0, 20.0, 0.0],
    ])
    # 3 nodes -> padded to 2^2 = 4 dimensions
    L_norm, L_padded, op, sparse_pauli, n_qubits = build_laplacian_operators(D, eps=5.0)

    # 1. Exact HamiltonianGate circuit
    qc_exact, _ = build_ctqw_circuit(L_padded, sparse_pauli, n_qubits, time_val=1.5, trotter=False)
    assert qc_exact.num_qubits == 2

    # 2. LieTrotter product formula circuit
    qc_trotter, _ = build_ctqw_circuit(L_padded, sparse_pauli, n_qubits, time_val=1.5, trotter=True, reps=2)
    assert qc_trotter.num_qubits == 2

    # 3. Simulate statevector transitions
    P = simulate_ctqw_transitions(qc_exact, n_qubits, N_nodes=3)
    assert P.shape == (3, 3)
    # Conservation of probability: column sums = 1
    assert np.allclose(np.sum(P, axis=0), 1.0)
    # Nodes 0 and 1 are connected, node 2 is disconnected -> P[2, 0] should be 0
    assert np.isclose(P[2, 0], 0.0)


def test_sample_ctqw_transitions():
    D = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    L_norm, L_padded, op, sparse_pauli, n_qubits = build_laplacian_operators(D, eps=5.0)
    qc, _ = build_ctqw_circuit(L_padded, sparse_pauli, n_qubits, time_val=1.0)

    # Sample from node 0
    sampled_probs = sample_ctqw_transitions(qc, n_qubits, N_nodes=2, source_node=0, shots=256)
    assert len(sampled_probs) == 2
    assert np.isclose(np.sum(sampled_probs), 1.0)


def test_quantum_fast_traclus_pipeline():
    # Two distinct corridors
    t1 = [np.column_stack([np.linspace(0, 30, 10), np.random.normal(0.0, 0.1, 10)]) for _ in range(3)]
    t2 = [np.column_stack([np.linspace(0, 30, 10), np.random.normal(25.0, 0.1, 10)]) for _ in range(3)]
    all_trajs = t1 + t2

    model = QuantumFastTRACLUS(eps=6.0, min_lines=3, tau=0.03)
    model.fit(all_trajs)

    assert model.segments_ is not None
    assert model.labels_ is not None
    valid = np.unique(model.labels_[model.labels_ >= 0])
    assert len(valid) == 2
    reps = model.get_representative_trajectories()
    assert len(reps) == 2
