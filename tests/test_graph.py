"""Tests for sparse affinity graph and normalized Laplacian construction."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pytest
from fast_traclus_quantum.graph import (
    build_affinity_matrix,
    build_normalized_laplacian,
    AffinityGraph,
)


def test_affinity_matrix_properties():
    # 4 parallel segments
    segs = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 1.0], [10.0, 1.0]],
        [[0.0, 2.0], [10.0, 2.0]],
        [[0.0, 20.0], [10.0, 20.0]],  # Far away
    ])

    eps = 5.0
    sigma = 2.5
    W = build_affinity_matrix(segs, eps=eps, sigma=sigma, use_spatial_index=False)

    assert sp.issparse(W)
    assert W.shape == (4, 4)

    # Symmetric
    diff = W - W.T
    assert diff.nnz == 0 or np.allclose(diff.toarray(), 0.0)

    # Zero diagonal
    assert np.allclose(W.diagonal(), 0.0)

    # Segment 3 is far away (> eps) from 0, 1, 2
    W_dense = W.toarray()
    assert W_dense[0, 3] == 0.0
    assert W_dense[1, 3] == 0.0
    assert W_dense[2, 3] == 0.0

    # Segments 0 and 1 are close
    assert W_dense[0, 1] > 0.0


def test_normalized_laplacian_spectral_bounds():
    # Connected triangle
    W_dense = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    W = sp.csr_matrix(W_dense)
    L_norm = build_normalized_laplacian(W)

    assert sp.issparse(L_norm)
    # Check eigenvalues lie in [0, 2]
    evals = np.linalg.eigvalsh(L_norm.toarray())
    assert np.all(evals >= -1e-12)
    assert np.all(evals <= 2.0 + 1e-12)
    # Smallest eigenvalue of connected graph is 0
    assert np.isclose(evals[0], 0.0)


def test_isolated_node_handling():
    # 2 connected nodes and 1 completely isolated node
    W_dense = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    W = sp.csr_matrix(W_dense)
    L_norm = build_normalized_laplacian(W)

    L_dense = L_norm.toarray()
    # No NaNs or Infs
    assert not np.any(np.isnan(L_dense))
    assert not np.any(np.isinf(L_dense))

    # Node 2 is isolated -> diagonal entry is 1, off-diagonals 0
    assert np.isclose(L_dense[2, 2], 1.0)
    assert np.isclose(L_dense[2, 0], 0.0)
    assert np.isclose(L_dense[0, 2], 0.0)


def test_spatial_index_equivalence():
    np.random.seed(42)
    # 20 random segments in a cluster
    starts = np.random.uniform(0, 50, (20, 2))
    ends = starts + np.random.uniform(5, 15, (20, 2))
    segs = np.stack([starts, ends], axis=1)

    eps = 10.0
    W_dense_path = build_affinity_matrix(segs, eps=eps, use_spatial_index=False)
    W_spatial_path = build_affinity_matrix(segs, eps=eps, use_spatial_index=True, spatial_threshold_n=0)

    assert np.allclose(W_dense_path.toarray(), W_spatial_path.toarray(), atol=1e-10)
