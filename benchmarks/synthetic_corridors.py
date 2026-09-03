"""Synthetic non-convex trajectory corridor generators."""

from typing import List, Tuple
import numpy as np


def generate_dual_spirals(
    n_trajectories_per_corridor: int = 8,
    n_points: int = 35,
    noise: float = 0.12,
    random_state: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate dual interlocking Archimedean spiral trajectory corridors.

    Parameters
    ----------
    n_trajectories_per_corridor : int, default=8
        Number of trajectories simulated per spiral arm.
    n_points : int, default=35
        Number of GPS coordinates sampled along each trajectory.
    noise : float, default=0.12
        Standard deviation of transverse trajectory jitter.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    trajectories : list of ndarray of shape (n_points, 2)
        List of 2D trajectories.
    labels : ndarray of shape (2 * n_trajectories_per_corridor,)
        Ground-truth corridor assignments (0 or 1).
    """
    np.random.seed(random_state)
    trajectories = []
    ground_truth = []

    # Spiral A
    for i in range(n_trajectories_per_corridor):
        lane_offset = (i - n_trajectories_per_corridor / 2.0) * 0.45
        theta = np.linspace(0.6 * np.pi, 2.2 * np.pi, n_points)
        r = 10.0 + 3.5 * theta + lane_offset + np.random.normal(0, noise, n_points)
        th = theta + np.random.normal(0, 0.015, n_points)
        x = r * np.cos(th)
        y = r * np.sin(th)
        trajectories.append(np.column_stack([x, y]))
        ground_truth.append(0)

    # Spiral B (interleaving, rotated 180 degrees)
    for i in range(n_trajectories_per_corridor):
        lane_offset = (i - n_trajectories_per_corridor / 2.0) * 0.45
        theta = np.linspace(0.6 * np.pi, 2.2 * np.pi, n_points)
        r = 10.0 + 3.5 * theta + lane_offset + np.random.normal(0, noise, n_points)
        th = theta + np.pi + np.random.normal(0, 0.015, n_points)
        x = r * np.cos(th)
        y = r * np.sin(th)
        trajectories.append(np.column_stack([x, y]))
        ground_truth.append(1)

    return trajectories, np.array(ground_truth)


def generate_u_arterials(
    n_trajectories_per_corridor: int = 8,
    n_points: int = 36,
    noise: float = 0.15,
    random_state: int = 123,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate nested non-convex U-shaped arterial corridors."""
    np.random.seed(random_state)
    trajectories = []
    ground_truth = []

    leg = n_points // 3

    # Inner U-turn arterial: (0, 15) -> (25, 15) -> (25, 0) -> (0, 0)
    for i in range(n_trajectories_per_corridor):
        lane = (i - n_trajectories_per_corridor / 2.0) * 0.4
        leg1 = np.column_stack([np.linspace(0, 25, leg), np.full(leg, 15.0 + lane)])
        leg2 = np.column_stack([np.full(leg, 25.0 + lane), np.linspace(15.0, 0.0, leg)])
        leg3 = np.column_stack([np.linspace(25, 0, leg), np.full(leg, 0.0 + lane)])
        traj = np.vstack([leg1, leg2, leg3]) + np.random.normal(0, noise, (3 * leg, 2))
        trajectories.append(traj)
        ground_truth.append(0)

    # Outer bypass arterial: (0, 26) -> (38, 26) -> (38, -12) -> (0, -12)
    for i in range(n_trajectories_per_corridor):
        lane = (i - n_trajectories_per_corridor / 2.0) * 0.4
        leg1 = np.column_stack([np.linspace(0, 38, leg), np.full(leg, 26.0 + lane)])
        leg2 = np.column_stack([np.full(leg, 38.0 + lane), np.linspace(26.0, -12.0, leg)])
        leg3 = np.column_stack([np.linspace(38, 0, leg), np.full(leg, -12.0 + lane)])
        traj = np.vstack([leg1, leg2, leg3]) + np.random.normal(0, noise, (3 * leg, 2))
        trajectories.append(traj)
        ground_truth.append(1)

    return trajectories, np.array(ground_truth)


def trajectories_to_dataframe_format(trajectories: List[np.ndarray]) -> np.ndarray:
    """Convert trajectory list into standard tabular format: [trajectory_id, timestamp, x, y]."""
    rows = []
    for traj_id, traj in enumerate(trajectories):
        for t_step, pt in enumerate(traj):
            rows.append([float(traj_id), float(t_step), float(pt[0]), float(pt[1])])
    return np.asarray(rows, dtype=np.float64)


def dataframe_format_to_trajectories(tabular_data: np.ndarray) -> List[np.ndarray]:
    """Convert standard tabular format [trajectory_id, timestamp, x, y] to trajectory list."""
    traj_ids = np.unique(tabular_data[:, 0].astype(int))
    traj_ids.sort()
    trajectories = []
    for tid in traj_ids:
        mask = tabular_data[:, 0].astype(int) == tid
        sub = tabular_data[mask]
        # Sort by timestamp
        order = np.argsort(sub[:, 1])
        coords = sub[order, 2:4]
        trajectories.append(coords)
    return trajectories
