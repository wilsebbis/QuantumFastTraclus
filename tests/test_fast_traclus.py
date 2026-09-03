"""Tests for Fast-TRACLUS vectorized MDL and modular clustering backends."""

import numpy as np
import pytest
from fast_traclus.vectorized_mdl import (
    vectorized_point_projections,
    vectorized_mdl_partition,
    partition_trajectories_fast,
)
from fast_traclus.distance_matrix import compute_distance_matrix
from fast_traclus.modular_clustering import modular_cluster_segments, FastTRACLUS


def test_vectorized_mdl_partition():
    # Straight line
    x = np.linspace(0, 100, 20)
    traj = np.column_stack([x, np.zeros_like(x)])
    segs = vectorized_mdl_partition(traj)
    assert len(segs) == 1
    assert np.allclose(segs[0, 0], [0.0, 0.0])
    assert np.allclose(segs[0, 1], [100.0, 0.0])


def test_compute_distance_matrix():
    segs = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 1.0], [10.0, 1.0]],
        [[0.0, 20.0], [10.0, 20.0]],
    ])
    D = compute_distance_matrix(segs)
    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert D[0, 1] < D[0, 2]


def test_modular_clustering_backends():
    # 2 clusters of 3 segments each
    c1 = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 0.5], [10.0, 0.5]],
        [[0.0, 1.0], [10.0, 1.0]],
    ])
    c2 = np.array([
        [[0.0, 50.0], [10.0, 50.0]],
        [[0.0, 50.5], [10.0, 50.5]],
        [[0.0, 51.0], [10.0, 51.0]],
    ])
    segs = np.vstack([c1, c2])
    traj_ids = np.array([0, 1, 2, 3, 4, 5])
    D = compute_distance_matrix(segs)

    backends = ["dbscan", "spectral", "optics", "hdbscan", "agglomerative"]
    for backend in backends:
        labels = modular_cluster_segments(
            distance_matrix=D,
            traj_ids=traj_ids,
            backend=backend,
            eps=5.0,
            min_lines=2,
            n_clusters=2,
        )
        assert len(labels) == 6
        valid = np.unique(labels[labels >= 0])
        # Each backend should discover 2 clusters
        assert len(valid) == 2, f"Backend {backend} failed to find 2 clusters: {valid}"
        # Segments 0..2 in one cluster, 3..5 in another
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]


def test_fast_traclus_pipeline():
    t1 = [np.column_stack([np.linspace(0, 30, 10), np.random.normal(0.0, 0.1, 10)]) for _ in range(3)]
    t2 = [np.column_stack([np.linspace(0, 30, 10), np.random.normal(25.0, 0.1, 10)]) for _ in range(3)]
    all_trajs = t1 + t2

    model = FastTRACLUS(eps=5.0, min_lines=3, backend="dbscan")
    model.fit(all_trajs)

    assert model.segments_ is not None
    assert model.labels_ is not None
    valid = np.unique(model.labels_[model.labels_ >= 0])
    assert len(valid) == 2
    reps = model.get_representative_trajectories()
    assert len(reps) == 2
