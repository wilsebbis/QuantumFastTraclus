"""Numerical equivalence and geometry tests for core line segment distance functions."""

import numpy as np
import pytest
from core.distance import (
    iterative_segment_distance,
    vectorized_segment_distance,
    pairwise_segment_distances,
)
from core.representative import generate_representative_trajectory


def test_distance_numerical_equivalence_random():
    """Verify that iterative scalar and broadcasted vectorized distance functions are identical."""
    np.random.seed(42)
    N = 25
    starts = np.random.uniform(-50, 50, (N, 2))
    diffs = np.random.uniform(2, 20, (N, 2))
    segments = np.stack([starts, starts + diffs], axis=1)

    # 1. Pairwise test against iterative
    for i in range(N):
        for j in range(N):
            iter_total = iterative_segment_distance(segments[i], segments[j])
            vec_total = vectorized_segment_distance(segments[i], segments[j])
            assert np.isclose(iter_total, vec_total, atol=1e-12), f"Mismatch at ({i}, {j}): {iter_total} != {vec_total}"

            # Check individual components
            d_p1, d_l1, d_t1 = iterative_segment_distance(segments[i], segments[j], return_components=True)
            d_p2, d_l2, d_t2 = vectorized_segment_distance(segments[i], segments[j], return_components=True)
            assert np.isclose(d_p1, d_p2, atol=1e-12)
            assert np.isclose(d_l1, d_l2, atol=1e-12)
            assert np.isclose(d_t1, d_t2, atol=1e-12)


def test_parallel_and_collinear_cases():
    # Parallel segments
    seg1 = np.array([[0.0, 0.0], [10.0, 0.0]])
    seg2 = np.array([[0.0, 4.0], [10.0, 4.0]])
    assert np.isclose(iterative_segment_distance(seg1, seg2), 4.0)
    assert np.isclose(vectorized_segment_distance(seg1, seg2), 4.0)

    # Collinear segments offset by 3.0
    seg3 = np.array([[13.0, 0.0], [23.0, 0.0]])
    assert np.isclose(iterative_segment_distance(seg1, seg3), 3.0)
    assert np.isclose(vectorized_segment_distance(seg1, seg3), 3.0)


def test_orthogonal_and_antiparallel():
    seg1 = np.array([[0.0, 0.0], [10.0, 0.0]])
    seg2 = np.array([[5.0, 0.0], [5.0, 6.0]])  # shorter len 6, orthogonal
    dp, dl, dt = iterative_segment_distance(seg1, seg2, return_components=True)
    assert np.isclose(dt, 6.0)

    # Anti-parallel (180 degrees)
    seg3 = np.array([[10.0, 0.0], [0.0, 0.0]])
    dp, dl, dt = iterative_segment_distance(seg1, seg3, return_components=True)
    assert np.isclose(dt, 10.0)


def test_representative_trajectory_sweepline():
    # Cluster of 4 parallel segments
    segs = np.array([
        [[0.0, 0.0], [20.0, 0.0]],
        [[0.0, 0.5], [20.0, 0.5]],
        [[0.0, -0.5], [20.0, -0.5]],
        [[0.0, 0.2], [20.0, 0.2]],
    ])
    rep = generate_representative_trajectory(segs, min_lines=3, gamma=2.0)
    assert len(rep) >= 2
    # Check that Y coordinates are approximately 0.0
    assert np.allclose(rep[:, 1], 0.0, atol=0.2)
