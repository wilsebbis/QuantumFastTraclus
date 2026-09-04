"""Sample and synthetic fixtures matching real benchmark dataset volumes and spatial patterns.

Enables immediate end-to-end testing, parameter sweeps, and benchmark replication
before or in absence of full raw dataset downloads.
"""

from typing import Dict, List, Tuple
import numpy as np


def generate_sample_hurricane_dataset(
    n_trajectories: int = 570,
    target_points: int = 17736,
    random_state: int = 42,
) -> List[np.ndarray]:
    """Generate high-fidelity sample Atlantic hurricane trajectories matching SIGMOD 2007 specs.

    570 trajectories, ~17,736 points representing:
    1. East-to-West Cape Verde tracks (curving westward into Caribbean/Gulf)
    2. Recurving tracks (turning northward, then accelerating northeast into the Atlantic)
    3. Gulf of Mexico origin tracks
    """
    rng = np.random.RandomState(random_state)
    trajectories = []

    # Distribute 17,736 points across 570 trajectories (mean ~31 points per storm)
    pts_per_traj = rng.poisson(lam=target_points / n_trajectories, size=n_trajectories)
    pts_per_traj = np.maximum(8, pts_per_traj)
    diff = target_points - int(np.sum(pts_per_traj))
    for i in range(abs(diff)):
        idx = i % n_trajectories
        pts_per_traj[idx] += 1 if diff > 0 else -1

    for i in range(n_trajectories):
        n_pts = pts_per_traj[i]
        track_type = rng.choice(["cape_verde", "recurving", "gulf", "straight_west"])

        t = np.linspace(0, 1, n_pts)
        noise_x = rng.normal(0, 0.4, n_pts)
        noise_y = rng.normal(0, 0.4, n_pts)

        if track_type == "cape_verde":
            # Start at Cape Verde (lon -25, lat 12), move west-northwest to (-75, 25)
            x = -25.0 - 50.0 * t + noise_x
            y = 12.0 + 12.0 * t + 3.0 * np.sin(np.pi * t) + noise_y
        elif track_type == "recurving":
            # Parabolic recurving: (-50, 15) -> (-75, 30) -> (-40, 48)
            x = -50.0 - 35.0 * np.sin(np.pi * t * 0.7) + 20.0 * (t**2) + noise_x
            y = 15.0 + 35.0 * t + noise_y
        elif track_type == "gulf":
            # Gulf storm: (-85, 20) -> (-92, 28)
            x = -85.0 - 8.0 * np.sin(np.pi * t) + noise_x
            y = 20.0 + 10.0 * t + noise_y
        else:
            # Straight westward into Central America: (-30, 11) -> (-85, 14)
            x = -30.0 - 55.0 * t + noise_x
            y = 11.0 + 4.0 * t + noise_y

        trajectories.append(np.column_stack([x, y]))

    return trajectories


def generate_sample_elk1993_dataset(
    n_trajectories: int = 33,
    target_points: int = 47204,
    random_state: int = 42,
) -> List[np.ndarray]:
    """Generate sample Starkey Elk1993 dataset matching SIGMOD 2007 volume: 33 trajectories, 47,204 points."""
    rng = np.random.RandomState(random_state)
    trajectories = []

    pts_per_traj = rng.poisson(lam=target_points / n_trajectories, size=n_trajectories)
    pts_per_traj = np.maximum(50, pts_per_traj)
    diff = target_points - int(np.sum(pts_per_traj))
    for i in range(abs(diff)):
        idx = i % n_trajectories
        pts_per_traj[idx] += 1 if diff > 0 else -1

    # Starkey Project UTM coordinates roughly in range [800..1100] x [100..400]
    for i in range(n_trajectories):
        n_pts = pts_per_traj[i]
        t = np.linspace(0, 1, n_pts)
        corridor = i % 4
        if corridor == 0:
            x = np.linspace(850, 1020, n_pts) + rng.normal(0, 4.0, n_pts)
            y = np.linspace(120, 260, n_pts) + rng.normal(0, 4.0, n_pts)
        elif corridor == 1:
            x = np.linspace(1010, 930, n_pts) + rng.normal(0, 3.5, n_pts)
            y = np.linspace(250, 380, n_pts) + rng.normal(0, 3.5, n_pts)
        elif corridor == 2:
            x = np.linspace(870, 990, n_pts) + rng.normal(0, 3.0, n_pts)
            y = np.linspace(340, 160, n_pts) + rng.normal(0, 3.0, n_pts)
        else:
            x = np.linspace(920, 860, n_pts) + rng.normal(0, 3.0, n_pts)
            y = np.linspace(150, 320, n_pts) + rng.normal(0, 3.0, n_pts)
        trajectories.append(np.column_stack([x, y]))

    return trajectories


def generate_sample_deer1995_dataset(
    n_trajectories: int = 32,
    target_points: int = 20065,
    random_state: int = 42,
) -> List[np.ndarray]:
    """Generate sample Starkey Deer1995 dataset matching SIGMOD 2007 volume: 32 trajectories, 20,065 points."""
    rng = np.random.RandomState(random_state)
    trajectories = []

    pts_per_traj = rng.poisson(lam=target_points / n_trajectories, size=n_trajectories)
    pts_per_traj = np.maximum(40, pts_per_traj)
    diff = target_points - int(np.sum(pts_per_traj))
    for i in range(abs(diff)):
        idx = i % n_trajectories
        pts_per_traj[idx] += 1 if diff > 0 else -1

    # 2 major corridor clusters as discovered in SIGMOD 2007 (Figure 22)
    for i in range(n_trajectories):
        n_pts = pts_per_traj[i]
        corridor = i % 2
        t = np.linspace(0, 1, n_pts)
        if corridor == 0:
            # Corridor A: northwest-to-southeast
            x = np.linspace(870, 1015, n_pts) + rng.normal(0, 3.0, n_pts)
            y = np.linspace(350, 180, n_pts) + rng.normal(0, 3.0, n_pts)
        else:
            # Corridor B: southwest-to-northeast
            x = np.linspace(890, 1010, n_pts) + rng.normal(0, 3.0, n_pts)
            y = np.linspace(150, 320, n_pts) + rng.normal(0, 3.0, n_pts)
        trajectories.append(np.column_stack([x, y]))

    return trajectories


def generate_sample_synthetic_noise_dataset(
    n_corridor_trajectories: int = 30,
    noise_ratio: float = 0.25,
    n_points: int = 30,
    random_state: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate 4 linear corridors with 25% random noise trajectories (SIGMOD 2007 Figure 23)."""
    rng = np.random.RandomState(random_state)
    trajectories = []
    labels = []

    # 4 distinct linear corridors
    corridors = [
        ([0, 100], [0, 0]),
        ([0, 100], [50, 50]),
        ([0, 0], [0, 100]),
        ([50, 50], [0, 100]),
    ]

    for c_id, (x_span, y_span) in enumerate(corridors):
        for _ in range(n_corridor_trajectories // len(corridors)):
            t = np.linspace(0, 1, n_points)
            x = np.linspace(x_span[0], x_span[1], n_points) + rng.normal(0, 1.2, n_points)
            y = np.linspace(y_span[0], y_span[1], n_points) + rng.normal(0, 1.2, n_points)
            trajectories.append(np.column_stack([x, y]))
            labels.append(c_id)

    # Injected uniform random noise trajectories (25% of total)
    n_noise = int(round(len(trajectories) * noise_ratio / (1.0 - noise_ratio)))
    for _ in range(n_noise):
        start = rng.uniform(-10, 110, 2)
        end = rng.uniform(-10, 110, 2)
        t = np.linspace(0, 1, n_points)[:, np.newaxis]
        line = start + t * (end - start) + rng.normal(0, 3.0, (n_points, 2))
        trajectories.append(line)
        labels.append(-1)

    return trajectories, np.array(labels)


def generate_sample_fast_traclus_subsets(
    dataset_name: str,
    n_trajectories: int,
    random_state: int = 42,
) -> List[np.ndarray]:
    """Generate realistic progressive subsets for Taxi, Movebank, or Geolife benchmark reproduction."""
    rng = np.random.RandomState(random_state)
    trajectories = []

    if dataset_name.lower() == "taxi":
        # Urban taxi routes with road-network turns
        for _ in range(n_trajectories):
            n_p = rng.randint(20, 60)
            t = np.linspace(0, 1, n_p)
            route = rng.choice([0, 1, 2, 3])
            noise = rng.normal(0, 0.002, (n_p, 2))
            base_x = -8.62 + 0.05 * t + (0.02 if route % 2 == 0 else -0.02) * np.sin(2 * np.pi * t)
            base_y = 41.15 + 0.04 * t + (0.02 if route >= 2 else -0.01) * np.cos(2 * np.pi * t)
            trajectories.append(np.column_stack([base_x, base_y]) + noise)

    elif dataset_name.lower() == "movebank":
        # Sparse wildlife collar trajectories
        for _ in range(n_trajectories):
            n_p = rng.randint(25, 80)
            t = np.linspace(0, 1, n_p)
            base_x = 10.0 + 15.0 * t + rng.normal(0, 0.8, n_p)
            base_y = 45.0 + 8.0 * np.sin(np.pi * t) + rng.normal(0, 0.8, n_p)
            trajectories.append(np.column_stack([base_x, base_y]))

    else:  # geolife
        # High-density pedestrian GPS tracks
        for _ in range(n_trajectories):
            n_p = rng.randint(60, 150)
            t = np.linspace(0, 1, n_p)
            base_x = 116.3 + 0.08 * t + rng.normal(0, 0.001, n_p)
            base_y = 39.9 + 0.06 * (t**1.5) + rng.normal(0, 0.001, n_p)
            trajectories.append(np.column_stack([base_x, base_y]))

    return trajectories

