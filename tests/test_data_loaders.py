"""Tests for dataset loaders, parsers, and sample fixtures."""

import os
from pathlib import Path
import numpy as np
import pytest

from data.loaders import (
    parse_tra_file,
    write_tra_file,
    load_hurricane_data,
    load_elk_data,
    load_deer_data,
    load_synthetic_noise_data,
    load_fast_traclus_dataset,
)


def test_tra_file_roundtrip(tmp_path):
    # Create sample trajectories
    t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    t2 = np.array([[10.0, 10.0], [12.0, 15.0]])
    trajs = [t1, t2]

    out_file = tmp_path / "test.tra"
    write_tra_file(out_file, trajs)
    assert out_file.exists()

    loaded = parse_tra_file(out_file)
    assert len(loaded) == 2
    assert np.allclose(loaded[0], t1, atol=1e-4)
    assert np.allclose(loaded[1], t2, atol=1e-4)


def test_load_hurricane_dataset():
    # Tests loading either from real file or sample fixture
    trajs, is_real = load_hurricane_data()
    assert len(trajs) == 570
    total_pts = sum(len(t) for t in trajs)
    assert total_pts == 17736, f"Expected 17,736 points, got {total_pts}"


def test_load_elk_dataset():
    trajs, is_real = load_elk_data()
    assert len(trajs) == 33
    total_pts = sum(len(t) for t in trajs)
    assert total_pts == 47204, f"Expected 47,204 points, got {total_pts}"


def test_load_deer_dataset():
    trajs, is_real = load_deer_data()
    assert len(trajs) == 32
    total_pts = sum(len(t) for t in trajs)
    assert total_pts == 20065, f"Expected 20,065 points, got {total_pts}"


def test_synthetic_noise_dataset():
    trajs, labels = load_synthetic_noise_data(noise_ratio=0.25)
    assert len(trajs) == len(labels)
    # Check noise proportion ~25%
    noise_count = np.sum(labels == -1)
    noise_frac = noise_count / len(labels)
    assert 0.20 <= noise_frac <= 0.30


def test_fast_traclus_dataset_subsets():
    for ds in ["taxi", "movebank", "geolife"]:
        trajs, _ = load_fast_traclus_dataset(ds, n_trajectories=100)
        assert len(trajs) == 100

