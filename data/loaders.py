"""Data parsers, loaders, and schema converters for real and benchmark trajectory datasets."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

from data.sample_fixtures import (
    generate_sample_hurricane_dataset,
    generate_sample_elk1993_dataset,
    generate_sample_deer1995_dataset,
    generate_sample_synthetic_noise_dataset,
    generate_sample_fast_traclus_subsets,
)

DEFAULT_RAW_DIR = Path(__file__).parent / "raw"


def parse_tra_file(filepath: Union[str, Path]) -> List[np.ndarray]:
    """Parse standard TRACLUS format file (.tra).

    Format:
    Line 0: Dimension (e.g. 2)
    Line 1: Total number of trajectories
    Line 2..N+1: <traj_id> <num_points> <x1> <y1> <x2> <y2> ...
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        return []

    dim = int(lines[0])
    n_trajs = int(lines[1])
    trajectories = []

    for line in lines[2 : 2 + n_trajs]:
        tokens = line.split()
        if len(tokens) < 2:
            continue
        traj_id = int(tokens[0])
        n_pts = int(tokens[1])
        coords = np.array(tokens[2 : 2 + n_pts * dim], dtype=np.float64)
        if len(coords) == n_pts * dim:
            coords = coords.reshape(n_pts, dim)
            trajectories.append(coords)

    return trajectories


def write_tra_file(filepath: Union[str, Path], trajectories: List[np.ndarray]) -> None:
    """Export trajectories to standard TRACLUS .tra file format."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        dim = trajectories[0].shape[1] if len(trajectories) > 0 else 2
        f.write(f"{dim}\n")
        f.write(f"{len(trajectories)}\n")
        for tid, traj in enumerate(trajectories):
            coords_flat = " ".join(f"{x:.4f} {y:.4f}" for x, y in traj)
            f.write(f"{tid} {len(traj)} {coords_flat}\n")


def load_hurricane_data(
    filepath: Optional[Union[str, Path]] = None,
    use_sample: bool = False,
    n_trajectories: int = 570,
) -> Tuple[List[np.ndarray], bool]:
    """Load Hurricane Track Data (Best Track: 570 trajectories, 17,736 GPS points).

    Returns (trajectories, is_real_file).
    """
    raw_path = Path(filepath) if filepath else DEFAULT_RAW_DIR / "hurricane1950_2006.tra"
    if not use_sample and raw_path.exists():
        trajectories = parse_tra_file(raw_path)
        # Limit to the 1950-2004 hurricane scope (first 570 trajectories)
        return trajectories[:n_trajectories], True

    # Fallback to high-fidelity fixture matching volume and points
    return generate_sample_hurricane_dataset(n_trajectories=n_trajectories, target_points=17736), False


def load_elk_data(
    filepath: Optional[Union[str, Path]] = None,
    use_sample: bool = False,
) -> Tuple[List[np.ndarray], bool]:
    """Load Starkey Project Elk1993 Data (33 trajectories, 47,204 points)."""
    raw_path = Path(filepath) if filepath else DEFAULT_RAW_DIR / "elk_1993.tra"
    if not use_sample and raw_path.exists():
        trajectories = parse_tra_file(raw_path)
        return trajectories[:33], True

    return generate_sample_elk1993_dataset(n_trajectories=33, target_points=47204), False


def load_deer_data(
    filepath: Optional[Union[str, Path]] = None,
    use_sample: bool = False,
) -> Tuple[List[np.ndarray], bool]:
    """Load Starkey Project Deer1995 Data (32 trajectories, 20,065 points)."""
    raw_path = Path(filepath) if filepath else DEFAULT_RAW_DIR / "deer1995.tra"
    if not use_sample and raw_path.exists():
        trajectories = parse_tra_file(raw_path)
        return trajectories[:32], True

    return generate_sample_deer1995_dataset(n_trajectories=32, target_points=20065), False


def load_synthetic_noise_data(
    noise_ratio: float = 0.25,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load Synthetic Linear Corridor Data injected with 25% background noise."""
    return generate_sample_synthetic_noise_dataset(noise_ratio=noise_ratio)


def load_fast_traclus_dataset(
    dataset_name: str,
    n_trajectories: int = 100,
    filepath: Optional[Union[str, Path]] = None,
    use_sample: bool = False,
) -> Tuple[List[np.ndarray], bool]:
    """Load subsets for Fast-TRACLUS benchmarks: 'taxi', 'movebank', or 'geolife'."""
    raw_path = Path(filepath) if filepath else DEFAULT_RAW_DIR / f"{dataset_name.lower()}_{n_trajectories}.tra"
    if not use_sample and raw_path.exists():
        return parse_tra_file(raw_path)[:n_trajectories], True

    return generate_sample_fast_traclus_subsets(dataset_name=dataset_name, n_trajectories=n_trajectories), False

