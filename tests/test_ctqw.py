"""Tests for Continuous-Time Quantum Walk propagation and community grouping."""

import numpy as np
import scipy.sparse as sp
import pytest
from fast_traclus_quantum.ctqw import CTQWClusterer
from fast_traclus_quantum.graph import build_normalized_laplacian


def test_unitary_properties_and_probability_conservation():
    # 5-node ring graph
    adj = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=float)
    W = sp.csr_matrix(adj)
    L_norm = build_normalized_laplacian(W)

    clusterer = CTQWClusterer(t=2.0)
    U = clusterer.evolve(L_norm, t=2.0)

    # 1. Unitarity: U^dagger @ U = I
    U_dagger_U = U.conj().T @ U
    assert np.allclose(U_dagger_U, np.eye(5), atol=1e-12)

    # 2. Probability conservation: sum_k P_jk = 1 for each source j
    P, K = clusterer.compute_transition_kernel(L_norm, t=2.0)
    col_sums = np.sum(P, axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-12)

    # 3. Kernel symmetry: K = K.T
    assert np.allclose(K, K.T, atol=1e-12)


def test_adaptive_time():
    # Path graph of 6 nodes
    N = 6
    adj = np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
    L_norm = build_normalized_laplacian(sp.csr_matrix(adj, dtype=float))

    clusterer = CTQWClusterer(adaptive_time=True)
    t = clusterer.compute_adaptive_time(L_norm)
    assert t > 0.0
    assert 0.1 <= t <= 50.0


def test_community_corridor_clustering():
    # Two disconnected corridors, 4 nodes each
    block1 = np.ones((4, 4)) - np.eye(4)
    block2 = np.ones((4, 4)) - np.eye(4)
    adj = np.block([
        [block1, np.zeros((4, 4))],
        [np.zeros((4, 4)), block2],
    ])
    W = sp.csr_matrix(adj, dtype=float)
    L_norm = build_normalized_laplacian(W)

    clusterer = CTQWClusterer(tau=0.05, min_samples=3)
    labels = clusterer.fit_predict(L_norm, W=W)

    assert len(labels) == 8
    # All nodes should be clustered
    assert np.all(labels >= 0)
    # First 4 nodes belong to one cluster, second 4 belong to another
    assert len(np.unique(labels[:4])) == 1
    assert len(np.unique(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_noise_suppression():
    # One cluster of 4 nodes, and 2 isolated noise nodes
    block1 = np.ones((4, 4)) - np.eye(4)
    adj = np.block([
        [block1, np.zeros((4, 2))],
        [np.zeros((2, 4)), np.zeros((2, 2))],
    ])
    W = sp.csr_matrix(adj, dtype=float)
    L_norm = build_normalized_laplacian(W)

    clusterer = CTQWClusterer(min_samples=3)
    labels = clusterer.fit_predict(L_norm, W=W)

    # First 4 nodes should form cluster 0
    assert np.all(labels[:4] == 0)
    # Isolated nodes should be labeled -1 (noise)
    assert labels[4] == -1
    assert labels[5] == -1


def test_qiskit_circuit_bridge():
    # Verify Qiskit Hamiltonian translation
    block1 = np.ones((4, 4)) - np.eye(4)
    W = sp.csr_matrix(block1, dtype=float)
    L_norm = build_normalized_laplacian(W)

    clusterer = CTQWClusterer(min_samples=2)
    clusterer.fit(L_norm, W=W)

    qc, op = clusterer.to_qiskit_circuit(subgraph_indices=[0, 1, 2, 3])
    assert qc.num_qubits == 2
    assert op.num_qubits == 2
    assert len(qc.data) >= 1
