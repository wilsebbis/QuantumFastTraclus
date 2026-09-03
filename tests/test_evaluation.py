"""Tests for evaluation metrics and benchmark orchestrator."""

import numpy as np
import pytest
from fast_traclus_quantum.evaluation import (
    davies_bouldin_index,
    silhouette_score,
    interference_contrast_ratio,
    benchmark_clustering,
)
from fast_traclus_quantum.pipeline import (
    FastTRACLUSQuantum,
    FastTRACLUSSpectralBaseline,
)


def test_davies_bouldin_index():
    # Two well-separated clusters of segments
    cluster1 = np.array([
        [[0.0, 0.0], [5.0, 0.0]],
        [[0.0, 0.5], [5.0, 0.5]],
        [[0.0, -0.5], [5.0, -0.5]],
    ])
    cluster2 = np.array([
        [[0.0, 100.0], [5.0, 100.0]],
        [[0.0, 100.5], [5.0, 100.5]],
        [[0.0, 99.5], [5.0, 99.5]],
    ])
    segs = np.vstack([cluster1, cluster2])
    labels = np.array([0, 0, 0, 1, 1, 1])

    dbi = davies_bouldin_index(segs, labels)
    assert not np.isnan(dbi)
    # Well-separated clusters should have low DBI (< 0.1)
    assert dbi < 0.1


def test_silhouette_score():
    cluster1 = np.array([
        [[0.0, 0.0], [5.0, 0.0]],
        [[0.0, 0.5], [5.0, 0.5]],
    ])
    cluster2 = np.array([
        [[0.0, 50.0], [5.0, 50.0]],
        [[0.0, 50.5], [5.0, 50.5]],
    ])
    segs = np.vstack([cluster1, cluster2])
    labels = np.array([0, 0, 1, 1])

    sil = silhouette_score(segs, labels)
    assert not np.isnan(sil)
    assert sil > 0.8


def test_interference_contrast_ratio():
    # Synthetic P matrix where intra-cluster transitions are high (0.2) and inter-cluster are low (0.01)
    N = 6
    P = np.full((N, N), 0.01)
    P[:3, :3] = 0.2
    P[3:, 3:] = 0.2
    np.fill_diagonal(P, 0.4)

    labels = np.array([0, 0, 0, 1, 1, 1])
    contrast = interference_contrast_ratio(P, labels)

    # Ratio should be ~0.2 / 0.01 = 20.0
    assert np.isclose(contrast, 20.0, atol=1.0)


def test_benchmark_clustering():
    traj1 = [np.column_stack([np.linspace(0, 30, 10), np.zeros(10)])]
    traj2 = [np.column_stack([np.linspace(0, 30, 10), np.full(10, 20.0)])]
    trajectories = traj1 * 3 + traj2 * 3

    ctqw = FastTRACLUSQuantum(eps=5.0, min_samples=2)
    spectral = FastTRACLUSSpectralBaseline(eps=5.0, min_samples=2, n_clusters=2)

    results = benchmark_clustering(ctqw, spectral, trajectories)

    assert "ctqw" in results
    assert "spectral" in results
    assert "runtime_sec" in results["ctqw"]
    assert "davies_bouldin" in results["ctqw"]
    assert "silhouette" in results["ctqw"]
    assert "contrast_ratio" in results["ctqw"]
