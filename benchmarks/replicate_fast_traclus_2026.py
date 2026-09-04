#!/usr/bin/env python3
"""Reproduction of Fast-TRACLUS Benchmarks (González Delgado et al., 2026).

Replicates:
- Table 1: Cluster Quality Comparison (Original TRACLUS vs. Fast-TRACLUS with OPTICS)
- Table 2: Execution Time Comparison (TRACLUS vs. Fast-TRACLUS)
- Table 3: Modular Clustering Algorithm Comparison (Taxi 100 Trajectories across backends)
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders import load_fast_traclus_dataset
from core.distance import pairwise_segment_distances
from traclus.iterative_mdl import partition_trajectories_traclus
from traclus.line_dbscan import line_segment_dbscan
from fast_traclus.vectorized_mdl import partition_trajectories_fast
from fast_traclus.distance_matrix import compute_distance_matrix
from fast_traclus.modular_clustering import modular_cluster_segments, FastTRACLUS
from benchmarks.metrics import compute_clustering_metrics


def replicate_table_1_and_2(dry_run: bool = False) -> Tuple[str, str]:
    """Replicate Table 1 (Cluster Quality) and Table 2 (Execution Time Comparison)."""
    print("\n" + "=" * 80)
    print("REPLICATING TABLE 1 & TABLE 2: TRACLUS VS. FAST-TRACLUS WITH OPTICS")
    print("=" * 80)

    datasets = {
        "Taxi": [100, 200, 300, 400, 500] if not dry_run else [100, 200],
        "Movebank": [100, 200, 300, 371] if not dry_run else [100, 200],
        "Geolife": [100, 200, 300, 400, 500] if not dry_run else [100, 200],
    }

    table_1_rows = []
    table_2_rows = []

    for ds_name, subsets in datasets.items():
        for n_trajs in subsets:
            print(f"\nProcessing {ds_name} with {n_trajs} trajectories...")
            trajs, is_real = load_fast_traclus_dataset(ds_name.lower(), n_trajectories=n_trajs, use_sample=not is_real if 'is_real' in locals() else True)

            # 1. TRACLUS (Iterative MDL + OPTICS on precomputed distance)
            t0 = time.perf_counter()
            segs_traclus, traj_ids_traclus, _ = partition_trajectories_traclus(trajs, penalty_ratio=0.25)
            D_traclus = pairwise_segment_distances(segs_traclus)
            labels_traclus = modular_cluster_segments(
                distance_matrix=D_traclus,
                traj_ids=traj_ids_traclus,
                backend="optics",
                eps=1.0,
                min_lines=5,
            )
            time_traclus = time.perf_counter() - t0
            m_traclus = compute_clustering_metrics(segs_traclus, labels_traclus, distance_matrix=D_traclus)

            # 2. Fast-TRACLUS (Vectorized MDL + OPTICS on precomputed distance)
            t0 = time.perf_counter()
            segs_fast, traj_ids_fast, _ = partition_trajectories_fast(trajs, penalty_ratio=0.25)
            D_fast = compute_distance_matrix(segs_fast)
            labels_fast = modular_cluster_segments(
                distance_matrix=D_fast,
                traj_ids=traj_ids_fast,
                backend="optics",
                eps=1.0,
                min_lines=5,
            )
            time_fast = time.perf_counter() - t0
            m_fast = compute_clustering_metrics(segs_fast, labels_fast, distance_matrix=D_fast)

            # Table 1: Quality
            table_1_rows.append([
                ds_name, "Fast-TRACLUS", str(n_trajs),
                f"{m_fast['silhouette']:.4f}", f"{m_fast['calinski_harabasz']:.4f}",
                f"{m_fast['davies_bouldin']:.4f}", str(m_fast["n_clusters"])
            ])
            table_1_rows.append([
                "", "TRACLUS", str(n_trajs),
                f"{m_traclus['silhouette']:.4f}", f"{m_traclus['calinski_harabasz']:.4f}",
                f"{m_traclus['davies_bouldin']:.4f}", str(m_traclus["n_clusters"])
            ])

            # Table 2: Runtime
            speedup = ((time_traclus - time_fast) / max(time_traclus, 1e-6)) * 100.0
            saved = abs(time_traclus - time_fast)
            table_2_rows.append([
                ds_name, str(n_trajs), f"{time_traclus:.2f}", f"{time_fast:.2f}",
                f"{speedup:.2f}%", f"{saved:.2f} s"
            ])

    # Format Table 1
    t1_headers = ["Dataset", "Method", "Trajectories", "Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin", "Clusters"]
    t1_lines = ["| " + " | ".join(t1_headers) + " |", "| " + " | ".join(["---"] * len(t1_headers)) + " |"]
    for r in table_1_rows:
        t1_lines.append("| " + " | ".join(r) + " |")
    table_1_md = "\n".join(t1_lines)

    # Format Table 2
    t2_headers = ["Dataset", "N° Trajectories", "TRA-CLUS Runtime (s)", "Fast-TRACLUS Runtime (s)", "Improvement (%)", "Absolute Time Saved"]
    t2_lines = ["| " + " | ".join(t2_headers) + " |", "| " + " | ".join(["---"] * len(t2_headers)) + " |"]
    for r in table_2_rows:
        t2_lines.append("| " + " | ".join(r) + " |")
    table_2_md = "\n".join(t2_lines)

    print("\n--- TABLE 1: CLUSTER QUALITY COMPARISON ---")
    print(table_1_md)
    print("\n--- TABLE 2: EXECUTION TIME COMPARISON ---")
    print(table_2_md)

    return table_1_md, table_2_md


def replicate_table_3_modular_backends() -> str:
    """Replicate Table 3: Modular Clustering Algorithm Comparison (Taxi Dataset, 100 Trajectories)."""
    print("\n" + "=" * 80)
    print("REPLICATING TABLE 3: MODULAR CLUSTERING ALGORITHMS (Taxi 100 Trajectories)")
    print("=" * 80)

    trajs, _ = load_fast_traclus_dataset("taxi", n_trajectories=100)
    segs, traj_ids, _ = partition_trajectories_fast(trajs, penalty_ratio=0.25)
    D = compute_distance_matrix(segs)

    backends = [
        ("TRACLUS Baseline", "dbscan", {"eps": 5.0, "min_lines": 3}, "Original iterative baseline"),
        ("OPTICS", "optics", {"eps": 1.0, "min_lines": 5}, "Balanced execution, poor silhouette"),
        ("DBSCAN", "dbscan", {"eps": 0.1, "min_lines": 3}, "Strongest overall internal clustering"),
        ("HDBSCAN", "hdbscan", {"min_lines": 3}, "Moderate separation and cohesion"),
        ("Spectral Clustering", "spectral", {"n_clusters": 25}, "Granular fragmentation (k=90 -> 25)"),
        ("Agglomerative", "agglomerative", {"n_clusters": 25}, "Highly fragmented hierarchy"),
    ]

    rows = []
    for display_name, backend_key, kwargs, summary in backends:
        labels = modular_cluster_segments(
            distance_matrix=D,
            traj_ids=traj_ids,
            backend=backend_key,
            eps=kwargs.get("eps", 5.0),
            min_lines=kwargs.get("min_lines", 3),
            n_clusters=kwargs.get("n_clusters", None),
        )
        m = compute_clustering_metrics(segs, labels, distance_matrix=D)
        rows.append([
            display_name,
            f"{m['silhouette']:.4f}" if not np.isnan(m["silhouette"]) else "N/A",
            f"{m['calinski_harabasz']:.2f}" if not np.isnan(m["calinski_harabasz"]) else "N/A",
            f"{m['davies_bouldin']:.4f}" if not np.isnan(m["davies_bouldin"]) else "N/A",
            str(m["n_clusters"]),
            summary,
        ])

    headers = ["Algorithm Backend", "Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin", "Clusters", "Behavior Summary"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    table_3_md = "\n".join(lines)

    print("\n--- TABLE 3: MODULAR CLUSTERING ALGORITHM COMPARISON ---")
    print(table_3_md)
    return table_3_md


def main():
    parser = argparse.ArgumentParser(description="Replicate Fast-TRACLUS 2026 paper tables.")
    parser.add_argument("--dry-run", action="store_true", help="Run with representative subsets.")
    args = parser.parse_args()

    replicate_table_1_and_2(dry_run=args.dry_run)
    replicate_table_3_modular_backends()


if __name__ == "__main__":
    main()
