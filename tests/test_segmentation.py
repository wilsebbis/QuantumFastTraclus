"""Tests for vectorized MDL trajectory partitioning."""

import numpy as np
import pytest
from fast_traclus_quantum.segmentation import (
    point_segment_projections,
    mdl_cost,
    partition_trajectory,
    partition_trajectories,
)


def test_point_segment_projections():
    p_start = np.array([0.0, 0.0])
    p_end = np.array([10.0, 0.0])
    # Intermediate points:
    # [5.0, 0.0]: on the segment -> d_perp = 0, d_par = 0
    # [3.0, 4.0]: inside X range [0, 10], shifted in Y -> d_perp = 4, d_par = 0
    # [-2.0, 0.0]: outside left of segment -> d_perp = 0, d_par = 2
    # [14.0, 3.0]: outside right of segment -> d_perp = 3, d_par = 4
    points = np.array([
        [5.0, 0.0],
        [3.0, 4.0],
        [-2.0, 0.0],
        [14.0, 3.0],
    ])

    d_perp, d_par = point_segment_projections(points, p_start, p_end)

    assert np.allclose(d_perp, [0.0, 4.0, 0.0, 3.0])
    assert np.allclose(d_par, [0.0, 0.0, 2.0, 4.0])


def test_straight_line_partition():
    # 20 collinear points along X axis
    x = np.linspace(0, 100, 20)
    traj = np.column_stack([x, np.zeros_like(x)])

    segments = partition_trajectory(traj)
    # A perfectly straight line should compress into a single segment
    assert len(segments) == 1
    assert np.allclose(segments[0, 0], [0.0, 0.0])
    assert np.allclose(segments[0, 1], [100.0, 0.0])


def test_l_shape_partition():
    # 10 points along +X, followed by 10 points along +Y
    pts1 = np.column_stack([np.linspace(0, 50, 10), np.zeros(10)])
    pts2 = np.column_stack([np.full(10, 50.0), np.linspace(5, 50, 10)])
    traj = np.vstack([pts1, pts2])

    segments = partition_trajectory(traj)
    # Should detect the corner and partition into exactly 2 segments
    assert len(segments) == 2
    assert np.allclose(segments[0, 0], [0.0, 0.0])
    assert np.allclose(segments[1, 1], [50.0, 50.0])
    # Intermediate corner point
    assert np.allclose(segments[0, 1], segments[1, 0])
    assert np.isclose(segments[0, 1][0], 50.0)


def test_partition_trajectories_batch():
    traj1 = np.column_stack([np.linspace(0, 20, 5), np.zeros(5)])
    traj2 = np.column_stack([np.zeros(5), np.linspace(0, 20, 5)])

    all_segs, traj_ids, seg_ids = partition_trajectories([traj1, traj2])
    assert len(all_segs) == 2
    assert np.array_equal(traj_ids, [0, 1])
    assert np.array_equal(seg_ids, [0, 0])


def test_short_trajectory():
    # Single point
    segs = partition_trajectory(np.array([[1.0, 2.0]]))
    assert len(segs) == 0

    # Two points
    segs2 = partition_trajectory(np.array([[0.0, 0.0], [10.0, 10.0]]))
    assert len(segs2) == 1
