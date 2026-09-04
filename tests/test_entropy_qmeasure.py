"""Tests for entropy parameter estimation and QMeasure clustering validation metric."""

import numpy as np
import pytest

from core.entropy import compute_entropy_at_eps, compute_entropy_sweep
from core.qmeasure import compute_qmeasure
from core.distance import pairwise_segment_distances


def test_entropy_computation():
    # 4 segments: 2 pairs close to each other (dist 1.0), but pairs separated by 50.0
    segs = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 1.0], [10.0, 1.0]],
        [[0.0, 50.0], [10.0, 50.0]],
        [[0.0, 51.0], [10.0, 51.0]],
    ])
    D = pairwise_segment_distances(segs)

    # At eps = 2.0: each segment sees 2 segments (itself + partner)
    h, avg_d, min_lines = compute_entropy_at_eps(D, eps=2.0)
    assert avg_d == 2.0
    assert min_lines == 3  # floor(2.0) + 1
    # Uniform probabilities -> max entropy log2(4) = 2.0
    assert np.isclose(h, 2.0)

    # At eps = 60.0: each segment sees all 4 segments
    h_all, avg_d_all, min_lines_all = compute_entropy_at_eps(D, eps=60.0)
    assert avg_d_all == 4.0
    assert min_lines_all == 5


def test_entropy_sweep():
    segs = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 0.5], [10.0, 0.5]],
        [[0.0, 20.0], [10.0, 20.0]],
    ])
    sweep = compute_entropy_sweep(segs, eps_range=[0.1, 1.0, 5.0, 30.0])
    assert "optimal_eps" in sweep
    assert "entropy_values" in sweep
    assert len(sweep["entropy_values"]) == 4


def test_qmeasure_calculation():
    segs = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 0.5], [10.0, 0.5]],
        [[0.0, 50.0], [10.0, 50.0]],
    ])
    # Case 1: First 2 in cluster 0, 3rd is noise (-1)
    labels = np.array([0, 0, -1])
    q = compute_qmeasure(segs, labels)
    assert q >= 0.0

    # Compact clusters have lower QMeasure than putting everything into noise
    labels_all_noise = np.array([-1, -1, -1])
    q_noise = compute_qmeasure(segs, labels_all_noise)
    assert q < q_noise

