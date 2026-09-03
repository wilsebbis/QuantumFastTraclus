#!/usr/bin/env python3
"""End-to-End Comparative Benchmark Harness for Trajectory Clustering.

Benchmarks:
1. Original TRACLUS (Lee et al., 2007)
2. Fast-TRACLUS (DBSCAN backend, 2026)
3. Fast-TRACLUS (Classical Spectral Clustering backend)
4. Fast-TRACLUS (Qiskit CTQW Quantum Walk backend)

Outputs:
- Comprehensive performance and internal cluster quality metrics table in Markdown.
- Visual comparative plots (benchmarks_comparison.png).
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.distance import pairwise_segment_distances
from traclus.iterative_mdl import partition_trajectories_traclus
from traclus.line_dbscan import line_segment_dbscan, OriginalTRACLUS
from fast_traclus.vectorized_mdl import partition_trajectories_fast
from fast_traclus.distance_matrix import compute_distance_matrix
from fast_traclus.modular_clustering import modular_cluster_segments, FastTRACLUS
from quantum_traclus.interference_cluster import QuantumFastTRACLUS
from benchmarks.synthetic_corridors import (
    generate_dual_spirals,
    trajectories_to_dataframe_format,
)
from benchmarks.metrics import compute_clustering_metrics, compute_interference_contrast


def benchmark_single_model(
    model_name: str,
    trajectories: List[np.ndarray],
    eps: float,
    min_lines: int,
    **kwargs,
) -> Dict[str, Any]:
    """Profile partitioning time, grouping time, and compute clustering quality metrics."""
    t_part_start = time.perf_counter()

    if model_name == "Original TRACLUS (2007)":
        # Original iterative MDL with 2D rotation matrices
        segments, traj_ids, seg_ids = partition_trajectories_traclus(trajectories)
        t_part = time.perf_counter() - t_part_start

        t_group_start = time.perf_counter()
        labels = line_segment_dbscan(segments, traj_ids, eps=eps, min_lines=min_lines)
        t_group = time.perf_counter() - t_group_start

        D = pairwise_segment_distances(segments)
        K = None

    elif model_name == "Fast-TRACLUS (DBSCAN, 2026)":
        # Fast vectorized MDL with dot products
        segments, traj_ids, seg_ids = partition_trajectories_fast(trajectories)
        t_part = time.perf_counter() - t_part_start

        t_group_start = time.perf_counter()
        D = compute_distance_matrix(segments)
        labels = modular_cluster_segments(
            distance_matrix=D,
            traj_ids=traj_ids,
            backend="dbscan",
            eps=eps,
            min_lines=min_lines,
        )
        t_group = time.perf_counter() - t_group_start
        K = None

    elif model_name == "Fast-TRACLUS (Spectral)":
        # Fast vectorized MDL with classical spectral clustering (O(N^3) Laplacian diagonalization)
        segments, traj_ids, seg_ids = partition_trajectories_fast(trajectories)
        t_part = time.perf_counter() - t_part_start

        t_group_start = time.perf_counter()
        D = compute_distance_matrix(segments)
        labels = modular_cluster_segments(
            distance_matrix=D,
            traj_ids=traj_ids,
            backend="spectral",
            eps=eps,
            min_lines=min_lines,
            n_clusters=kwargs.get("n_clusters", 2),
        )
        t_group = time.perf_counter() - t_group_start
        K = None

    elif model_name == "Fast-TRACLUS (Qiskit CTQW)":
        # Fast vectorized MDL with Qiskit CTQW Hamiltonian evolution
        segments, traj_ids, seg_ids = partition_trajectories_fast(trajectories)
        t_part = time.perf_counter() - t_part_start

        t_group_start = time.perf_counter()
        q_model = QuantumFastTRACLUS(
            eps=eps,
            min_lines=min_lines,
            tau=kwargs.get("tau", 0.02),
            trotter=kwargs.get("trotter", False),
        )
        # Internal fit on trajectories
        q_model.segments_ = segments
        q_model.traj_ids_ = traj_ids
        q_model.seg_ids_ = seg_ids
        D = compute_distance_matrix(segments)
        q_model.distance_matrix_ = D

        from quantum_traclus.laplacian_builder import build_laplacian_operators
        from quantum_traclus.ctqw_evolution import (
            build_ctqw_circuit,
            simulate_ctqw_transitions,
            compute_adaptive_walk_time,
        )
        from quantum_traclus.interference_cluster import extract_ctqw_corridors

        L_norm, L_padded, op, sparse_pauli, n_qubits = build_laplacian_operators(D, eps=eps)
        t_used = compute_adaptive_walk_time(L_norm)
        qc, t_param = build_ctqw_circuit(L_padded, sparse_pauli, n_qubits, time_val=t_used)
        P = simulate_ctqw_transitions(qc, n_qubits, len(segments), time_val=t_used, t_param=t_param)
        labels, K = extract_ctqw_corridors(P, traj_ids, tau=kwargs.get("tau", 0.02), min_lines=min_lines)
        t_group = time.perf_counter() - t_group_start

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    t_total = t_part + t_group
    metrics = compute_clustering_metrics(segments, labels, distance_matrix=D, K=K)

    return {
        "model_name": model_name,
        "segments": segments,
        "labels": labels,
        "D": D,
        "K": K,
        "t_part": t_part,
        "t_group": t_group,
        "t_total": t_total,
        "metrics": metrics,
    }


def print_markdown_comparison_table(results: List[Dict[str, Any]]) -> str:
    """Format benchmark results into a standardized publication-ready Markdown table."""
    headers = [
        "Algorithm",
        "Partition (s)",
        "Grouping (s)",
        "Total (s)",
        "Silhouette ↑",
        "Calinski-H ↑",
        "Davies-B ↓",
        "N_clusters",
        "Noise (%)",
        "Contrast C",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in results:
        name = r["model_name"]
        m = r["metrics"]
        t_p = f"{r['t_part']:.4f}"
        t_g = f"{r['t_group']:.4f}"
        t_tot = f"{r['t_total']:.4f}"
        sil = f"{m['silhouette']:.4f}" if not np.isnan(m['silhouette']) else "N/A"
        ch = f"{m['calinski_harabasz']:.1f}" if not np.isnan(m['calinski_harabasz']) else "N/A"
        dbi = f"{m['davies_bouldin']:.4f}" if not np.isnan(m['davies_bouldin']) else "N/A"
        n_c = str(m["n_clusters"])
        noise = f"{m['noise_ratio']*100:.1f}%"
        if not np.isnan(m["contrast_ratio"]):
            contrast = f"{m['contrast_ratio']:.1f}x" if m["contrast_ratio"] < 1e5 else ">10^5x"
        else:
            contrast = "N/A"

        row = [name, t_p, t_g, t_tot, sil, ch, dbi, n_c, noise, contrast]
        lines.append("| " + " | ".join(row) + " |")

    table_md = "\n".join(lines)
    print("\n" + table_md + "\n")
    return table_md


def plot_benchmark_comparison(
    trajectories: List[np.ndarray],
    results: List[Dict[str, Any]],
    save_path: str = "benchmarks_comparison.png",
):
    """Generate visual comparison artifact."""
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(5 * (len(results) + 1), 4.5))

    # Panel 0: Raw trajectories
    ax0 = axes[0]
    for traj in trajectories:
        ax0.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1.5)
    ax0.set_title("Ground-Truth Trajectories\n(Interlocking Spirals)", fontsize=10, fontweight="bold")
    ax0.set_aspect("equal", "datalim")
    ax0.grid(True, linestyle="--", alpha=0.4)

    # Panels 1..N: Model clusterings
    for idx, r in enumerate(results):
        ax = axes[idx + 1]
        segments = r["segments"]
        labels = r["labels"]
        unique_l = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_l))))

        for s_i, seg in enumerate(segments):
            lbl = labels[s_i]
            color = "lightgray" if lbl == -1 else colors[list(unique_l).index(lbl)]
            alpha = 0.3 if lbl == -1 else 0.85
            ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], color=color, alpha=alpha, linewidth=1.8)

        m = r["metrics"]
        sil_str = f"Sil: {m['silhouette']:.3f}" if not np.isnan(m['silhouette']) else "Sil: N/A"
        dbi_str = f"DBI: {m['davies_bouldin']:.3f}" if not np.isnan(m['davies_bouldin']) else "DBI: N/A"
        ax.set_title(f"{r['model_name']}\n{sil_str} | {dbi_str}", fontsize=9, fontweight="bold")
        ax.set_aspect("equal", "datalim")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[Artifact Saved] Visual comparison plot saved to: {save_path}")


def main():
    print("=" * 80)
    print("TRAJECTORY CLUSTERING SUITE: BENCHMARK RUNNER")
    print("Evaluating TRACLUS, Fast-TRACLUS, and Qiskit Quantum CTQW Fast-TRACLUS")
    print("=" * 80)

    # 1. Synthesize non-convex interlocking spiral trajectory dataset
    print("\n[Step 1] Synthesizing non-convex interlocking spiral trajectory dataset...")
    trajectories, gt_labels = generate_dual_spirals(
        n_trajectories_per_corridor=8,
        n_points=35,
        noise=0.10,
        random_state=42,
    )
    print(f"Generated {len(trajectories)} trajectories with {len(trajectories)*35} total GPS points.")

    # Export to tabular format
    tabular_df = trajectories_to_dataframe_format(trajectories)
    print(f"Tabular format verified: shape {tabular_df.shape} [trajectory_id, timestamp, x, y]")

    # Benchmark parameters
    eps = 6.0
    min_lines = 3
    tau = 0.02

    models = [
        "Original TRACLUS (2007)",
        "Fast-TRACLUS (DBSCAN, 2026)",
        "Fast-TRACLUS (Spectral)",
        "Fast-TRACLUS (Qiskit CTQW)",
    ]

    results = []
    for model_name in models:
        print(f"\n[Running Benchmark] -> {model_name} ...")
        res = benchmark_single_model(
            model_name=model_name,
            trajectories=trajectories,
            eps=eps,
            min_lines=min_lines,
            tau=tau,
            n_clusters=2,
        )
        results.append(res)
        print(f"  Finished in {res['t_total']:.4f}s (Part: {res['t_part']:.4f}s, Group: {res['t_group']:.4f}s)")

    # 2. Print Markdown Table
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    table_md = print_markdown_comparison_table(results)

    # 3. Plot visual artifact
    plot_benchmark_comparison(trajectories, results, save_path="benchmarks_comparison.png")

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
