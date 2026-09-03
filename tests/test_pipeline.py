"""Tests for end-to-end Fast-TRACLUS pipelines."""

import numpy as np
import pytest
from fast_traclus_quantum.pipeline import (
    FastTRACLUSQuantum,
    FastTRACLUSSpectralBaseline,
)


@pytest.fixture
def sample_corridors():
    """Generate two clear horizontal corridors of trajectories."""
    # Corridor 1: y ~ 0, x in [0, 50]
    # Corridor 2: y ~ 30, x in [0, 50]
    corridor1 = [
        np.column_stack([np.linspace(0, 50, 15), np.random.normal(0.0, 0.2, 15)])
        for _ in range(4)
    ]
    corridor2 = [
        np.column_stack([np.linspace(0, 50, 15), np.random.normal(30.0, 0.2, 15)])
        for _ in range(4)
    ]
    return corridor1 + corridor2


def test_fast_traclus_quantum_pipeline(sample_corridors):
    model = FastTRACLUSQuantum(eps=8.0, min_samples=2, tau=0.05)
    model.fit(sample_corridors)

    assert model.segments_ is not None
    assert len(model.segments_) > 0
    assert model.labels_ is not None
    assert len(model.labels_) == len(model.segments_)

    # Should detect 2 distinct corridors
    valid_clusters = np.unique(model.labels_[model.labels_ >= 0])
    assert len(valid_clusters) == 2

    # Representative trajectories should be computed for each cluster
    rep_trajs = model.get_representative_trajectories()
    assert len(rep_trajs) == 2
    for c_id, traj in rep_trajs.items():
        assert len(traj) >= 2
        assert traj.ndim == 2


def test_spectral_baseline_pipeline(sample_corridors):
    model = FastTRACLUSSpectralBaseline(eps=8.0, min_samples=2, n_clusters=2)
    model.fit(sample_corridors)

    assert model.segments_ is not None
    assert model.labels_ is not None
    valid_clusters = np.unique(model.labels_[model.labels_ >= 0])
    assert len(valid_clusters) == 2


def test_empty_trajectories():
    model = FastTRACLUSQuantum()
    model.fit([])
    assert len(model.segments_) == 0
    assert len(model.labels_) == 0
    assert len(model.get_representative_trajectories()) == 0
