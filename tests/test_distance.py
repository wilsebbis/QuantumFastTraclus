"""Tests for vectorized segment distance metrics."""

import numpy as np
import pytest
from fast_traclus_quantum.distance import (
    perpendicular_distance,
    parallel_distance,
    angular_distance,
    segment_distance,
    pairwise_segment_distances,
)


def test_identical_segments():
    seg1 = np.array([[0.0, 0.0], [5.0, 0.0]])
    seg2 = np.array([[0.0, 0.0], [5.0, 0.0]])

    d_perp = perpendicular_distance(seg1, seg2)
    d_par = parallel_distance(seg1, seg2)
    d_theta = angular_distance(seg1, seg2)
    total = segment_distance(seg1, seg2)

    assert np.isclose(d_perp, 0.0)
    assert np.isclose(d_par, 0.0)
    assert np.isclose(d_theta, 0.0)
    assert np.isclose(total, 0.0)


def test_parallel_shifted_segments():
    # Two parallel horizontal segments, shifted vertically by 3.0
    seg1 = np.array([[0.0, 0.0], [10.0, 0.0]])
    seg2 = np.array([[0.0, 3.0], [10.0, 3.0]])

    d_perp = perpendicular_distance(seg1, seg2)
    d_par = parallel_distance(seg1, seg2)
    d_theta = angular_distance(seg1, seg2)

    assert np.isclose(d_perp, 3.0)
    assert np.isclose(d_par, 0.0)
    assert np.isclose(d_theta, 0.0)
    assert np.isclose(segment_distance(seg1, seg2), 3.0)


def test_collinear_offset_segments():
    # Seg1: [0, 0] -> [10, 0]
    # Seg2: [15, 0] -> [25, 0] (offset by 5 along axis)
    seg1 = np.array([[0.0, 0.0], [10.0, 0.0]])
    seg2 = np.array([[15.0, 0.0], [25.0, 0.0]])

    d_perp = perpendicular_distance(seg1, seg2)
    d_par = parallel_distance(seg1, seg2)
    d_theta = angular_distance(seg1, seg2)

    assert np.isclose(d_perp, 0.0)
    assert np.isclose(d_par, 5.0)
    assert np.isclose(d_theta, 0.0)
    assert np.isclose(segment_distance(seg1, seg2), 5.0)


def test_orthogonal_segments():
    # Seg1 along X: length 10
    # Seg2 along Y: length 4
    seg1 = np.array([[0.0, 0.0], [10.0, 0.0]])
    seg2 = np.array([[5.0, 0.0], [5.0, 4.0]])

    d_theta = angular_distance(seg1, seg2)
    # Shorter segment length is 4.0, angle is 90 deg -> d_theta = 4.0 * sin(pi/2) = 4.0
    assert np.isclose(d_theta, 4.0)


def test_anti_parallel_segments():
    # Seg1 along +X, Seg2 along -X (angle 180 deg > 90 deg)
    seg1 = np.array([[0.0, 0.0], [10.0, 0.0]])
    seg2 = np.array([[10.0, 0.0], [0.0, 0.0]])

    d_theta = angular_distance(seg1, seg2)
    # For theta > pi/2, d_theta = ||L_shorter|| = 10.0
    assert np.isclose(d_theta, 10.0)


def test_symmetry():
    seg1 = np.array([[1.0, 2.0], [8.0, 6.0]])
    seg2 = np.array([[-2.0, 5.0], [3.0, 11.0]])

    d12 = segment_distance(seg1, seg2)
    d21 = segment_distance(seg2, seg1)

    assert np.isclose(d12, d21, atol=1e-12)


def test_pairwise_broadcasting_consistency():
    segs = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 2.0], [10.0, 2.0]],
        [[5.0, 0.0], [5.0, 5.0]],
        [[12.0, 0.0], [18.0, 0.0]],
    ])

    D_matrix = pairwise_segment_distances(segs)
    assert D_matrix.shape == (4, 4)
    assert np.allclose(D_matrix, D_matrix.T)
    assert np.allclose(np.diag(D_matrix), 0.0)

    # Compare pairwise with individual scalar computations
    for i in range(4):
        for j in range(4):
            if i == j:
                assert np.isclose(D_matrix[i, j], 0.0)
            else:
                expected = segment_distance(segs[i], segs[j])
                assert np.isclose(D_matrix[i, j], expected, atol=1e-12)


def test_degenerate_zero_length_segments():
    seg1 = np.array([[0.0, 0.0], [0.0, 0.0]])
    seg2 = np.array([[5.0, 5.0], [5.0, 5.0]])

    d = segment_distance(seg1, seg2)
    assert np.isclose(d, 0.0)
