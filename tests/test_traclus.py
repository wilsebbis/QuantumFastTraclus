"""Tests for Original TRACLUS (iterative MDL and line DBSCAN)."""

import numpy as np
import pytest
from traclus.iterative_mdl import (
    rotate_points_2d,
    iterative_mdl_costs_traclus,
    iterative_mdl_partition,
    partition_trajectories_traclus,
)
from traclus.line_dbscan import line_segment_dbscan, OriginalTRACLUS


def test_rotate_points_2d():
    origin = np.array([10.0, 10.0])
    pts = np.array([
        [10.0, 10.0],
        [20.0, 10.0],  # along +X
        [10.0, 20.0],  # along +Y
    ])
    # Rotate by 90 degrees (pi/2)
    rotated = rotate_points_2d(pts, origin=origin, phi=np.pi / 2.0)
    # Origin stays at 0, 0
    assert np.allclose(rotated[0], [0.0, 0.0])
    # (20, 10) rotated by -pi/2 becomes (0, -10)
    assert np.allclose(rotated[1], [0.0, -10.0])
    # (10, 20) rotated by -pi/2 becomes (10, 0)
    assert np.allclose(rotated[2], [10.0, 0.0])


def test_traclus_iterative_mdl():
    # Straight line of 20 points
    x = np.linspace(0, 50, 20)
    straight = np.column_stack([x, np.zeros_like(x)])
    segs = iterative_mdl_partition(straight)
    assert len(segs) == 1
    assert np.allclose(segs[0, 0], [0.0, 0.0])
    assert np.allclose(segs[0, 1], [50.0, 0.0])


def test_trajectory_cardinality_filter():
    # 4 segments clustered close together
    segs = np.array([
        [[0.0, 0.0], [10.0, 0.0]],
        [[0.0, 0.2], [10.0, 0.2]],
        [[0.0, 0.4], [10.0, 0.4]],
        [[0.0, 0.6], [10.0, 0.6]],
    ])

    # Case 1: Trajectory IDs [0, 0, 1, 1] -> only 2 unique trajectories
    # With min_lines=3, cardinality filter should prune to noise (-1)
    traj_ids_fail = np.array([0, 0, 1, 1])
    labels_fail = line_segment_dbscan(segs, traj_ids_fail, eps=2.0, min_lines=3)
    assert np.all(labels_fail == -1)

    # Case 2: Trajectory IDs [0, 1, 2, 3] -> 4 unique trajectories >= 3
    # Should successfully form a cluster
    traj_ids_pass = np.array([0, 1, 2, 3])
    labels_pass = line_segment_dbscan(segs, traj_ids_pass, eps=2.0, min_lines=3)
    assert np.all(labels_pass == 0)


def test_original_traclus_pipeline():
    # Two distinct corridors with 3 trajectories each
    t1 = [np.column_stack([np.linspace(0, 30, 10), np.random.normal(0.0, 0.1, 10)]) for _ in range(3)]
    t2 = [np.column_stack([np.linspace(0, 30, 10), np.random.normal(20.0, 0.1, 10)]) for _ in range(3)]
    all_trajs = t1 + t2

    model = OriginalTRACLUS(eps=5.0, min_lines=3)
    model.fit(all_trajs)

    assert model.segments_ is not None
    assert model.labels_ is not None
    valid = np.unique(model.labels_[model.labels_ >= 0])
    assert len(valid) == 2
    reps = model.get_representative_trajectories()
    assert len(reps) == 2
