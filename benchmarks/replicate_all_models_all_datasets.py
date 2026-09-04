#!/usr/bin/env python3
"""Comprehensive Multi-Model Trajectory Clustering Benchmark Suite.

Evaluates 3 Trajectory Clustering Engines:
1. Original TRACLUS (Lee, Han, & Whang, SIGMOD 2007)
2. Fast-TRACLUS (González Delgado et al., 2026)
3. Quantum Fast-TRACLUS with Continuous-Time Quantum Walks (Qiskit CTQW)

Across 8 Real and Synthetic Datasets:
1. Hurricane Track Data (Best Track 1950-2004)
2. Starkey Project Elk1993 Movement Data
3. Starkey Project Deer1995 Movement Data
4. Synthetic Linear Corridors with 25% Uniform Noise
5. Porto Taxi (100 Trajectories)
6. Movebank Wildlife (100 Trajectories)
7. GeoLife Pedestrian Mobility (100 Trajectories)
8. Dual Concentric Interlocking Spirals (Non-Convex Manifolds)

Generates:
- 4-panel visual comparison figures for EACH of the 8 datasets.
- CTQW probability interference transition kernel insets.
- Master summary comparison dashboard across all datasets and models.
- Publication-ready comparative Markdown metric tables.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.distance import pairwise_segment_distances
from core.representative import generate_representative_trajectory
from traclus.iterative_mdl import partition_trajectories_traclus
from traclus.line_dbscan import line_segment_dbscan
from fast_traclus.vectorized_mdl import partition_trajectories_fast
from fast_traclus.distance_matrix import compute_distance_matrix
from fast_traclus.modular_clustering import modular_cluster_segments
from quantum_traclus.laplacian_builder import build_laplacian_operators
from quantum_traclus.ctqw_evolution import (
    build_ctqw_circuit,
    simulate_ctqw_transitions,
    compute_adaptive_walk_time,
)
from quantum_traclus.interference_cluster import extract_ctqw_corridors
from benchmarks.synthetic_corridors import generate_dual_spirals
from benchmarks.metrics import compute_clustering_metrics
from data.loaders import (
    load_hurricane_data,
    load_elk_data,
    load_deer_data,
    load_synthetic_noise_data,
    load_fast_traclus_dataset,
)

FIGURES_DIR = PROJECT_ROOT / "figures"
ARTIFACTS_DIR = Path("/Users/wilsebbis/.gemini/antigravity/brain/5eb890ef-1bb1-422d-90e2-2328cabbbdd1")


def run_traclus(
    trajectories: List[np.ndarray],
    eps: float,
    min_lines: int,
) -> Dict[str, Any]:
    """Execute Original TRACLUS (iterative MDL + line DBSCAN)."""
    t_start = time.perf_counter()
    segs, traj_ids, seg_ids = partition_trajectories_traclus(trajectories, penalty_ratio=0.25)
    t_part = time.perf_counter() - t_start

    t_g_start = time.perf_counter()
    D = pairwise_segment_distances(segs)
    labels = line_segment_dbscan(segs, traj_ids, eps=eps, min_lines=min_lines, distance_matrix=D)
    t_group = time.perf_counter() - t_g_start

    # Build representative trajectories
    reps = {}
    for c in np.unique(labels[labels >= 0]):
        c_segs = segs[labels == c]
        rep = generate_representative_trajectory(c_segs, min_lines=max(2, min_lines // 2))
        if len(rep) >= 2:
            reps[c] = rep

    metrics = compute_clustering_metrics(segs, labels, distance_matrix=D)

    return {
        "model_name": "Original TRACLUS (2007)",
        "segments": segs,
        "labels": labels,
        "reps": reps,
        "D": D,
        "P": None,
        "t_part": t_part,
        "t_group": t_group,
        "t_total": t_part + t_group,
        "metrics": metrics,
    }


def run_fast_traclus(
    trajectories: List[np.ndarray],
    eps: float,
    min_lines: int,
) -> Dict[str, Any]:
    """Execute Fast-TRACLUS (vectorized MDL + broadcasted distance + DBSCAN)."""
    t_start = time.perf_counter()
    segs, traj_ids, seg_ids = partition_trajectories_fast(trajectories, penalty_ratio=0.25)
    t_part = time.perf_counter() - t_start

    t_g_start = time.perf_counter()
    D = compute_distance_matrix(segs)
    labels = modular_cluster_segments(
        distance_matrix=D,
        traj_ids=traj_ids,
        backend="dbscan",
        eps=eps,
        min_lines=min_lines,
    )
    t_group = time.perf_counter() - t_g_start

    # Build representative trajectories
    reps = {}
    for c in np.unique(labels[labels >= 0]):
        c_segs = segs[labels == c]
        rep = generate_representative_trajectory(c_segs, min_lines=max(2, min_lines // 2))
        if len(rep) >= 2:
            reps[c] = rep

    metrics = compute_clustering_metrics(segs, labels, distance_matrix=D)

    return {
        "model_name": "Fast-TRACLUS (2026)",
        "segments": segs,
        "labels": labels,
        "reps": reps,
        "D": D,
        "P": None,
        "t_part": t_part,
        "t_group": t_group,
        "t_total": t_part + t_group,
        "metrics": metrics,
    }


def run_quantum_fast_traclus(
    trajectories: List[np.ndarray],
    eps: float,
    min_lines: int,
    tau: float = 0.04,
) -> Dict[str, Any]:
    """Execute Quantum Fast-TRACLUS (vectorized MDL + Qiskit CTQW interference grouping)."""
    t_start = time.perf_counter()
    segs, traj_ids, seg_ids = partition_trajectories_fast(trajectories, penalty_ratio=0.25)
    t_part = time.perf_counter() - t_start

    t_g_start = time.perf_counter()
    D = compute_distance_matrix(segs)
    N = len(segs)

    if N > 0:
        L_norm, L_padded, op, sparse_pauli, n_qubits = build_laplacian_operators(D, eps=eps)
        t_walk = compute_adaptive_walk_time(L_norm)
        qc, t_param = build_ctqw_circuit(L_padded, sparse_pauli, n_qubits, time_val=t_walk, trotter=False)
        P = simulate_ctqw_transitions(qc, n_qubits, N, time_val=t_walk, t_param=t_param)
        labels, K = extract_ctqw_corridors(P, traj_ids, tau=tau, min_lines=min_lines)
    else:
        labels = np.empty(0, dtype=int)
        P = np.empty((0, 0))
        K = None

    t_group = time.perf_counter() - t_g_start

    # Build representative trajectories
    reps = {}
    if len(labels) > 0:
        for c in np.unique(labels[labels >= 0]):
            c_segs = segs[labels == c]
            rep = generate_representative_trajectory(c_segs, min_lines=max(2, min_lines // 2))
            if len(rep) >= 2:
                reps[c] = rep

    metrics = compute_clustering_metrics(segs, labels, distance_matrix=D, K=K)

    return {
        "model_name": "Quantum Fast-TRACLUS (CTQW)",
        "segments": segs,
        "labels": labels,
        "reps": reps,
        "D": D,
        "P": P,
        "t_part": t_part,
        "t_group": t_group,
        "t_total": t_part + t_group,
        "metrics": metrics,
    }


def plot_4panel_dataset_comparison(
    dataset_title: str,
    trajectories: List[np.ndarray],
    results: List[Dict[str, Any]],
    output_filename: str,
):
    """Render and save a 4-panel visual comparison: Raw | TRACLUS | Fast-TRACLUS | Quantum CTQW."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))

    # Panel 0: Raw trajectories
    ax0 = axes[0]
    palette = plt.cm.viridis(np.linspace(0, 1, max(1, len(trajectories))))
    for idx, traj in enumerate(trajectories):
        ax0.plot(traj[:, 0], traj[:, 1], color=palette[idx % len(palette)], alpha=0.55, linewidth=1.2)
        ax0.scatter(traj[0, 0], traj[0, 1], color=palette[idx % len(palette)], s=12, alpha=0.7)
    ax0.set_title(f"Raw Input Trajectories\n{dataset_title}\n(N={len(trajectories)} paths)", fontsize=10, fontweight="bold")
    ax0.set_xlabel("X coordinate")
    ax0.set_ylabel("Y coordinate")
    ax0.grid(True, linestyle="--", alpha=0.35)

    # Panels 1, 2, 3: Models
    model_styles = {
        "Original TRACLUS (2007)": {"rep_color": "crimson", "alpha_seg": 0.65},
        "Fast-TRACLUS (2026)": {"rep_color": "royalblue", "alpha_seg": 0.65},
        "Quantum Fast-TRACLUS (CTQW)": {"rep_color": "darkorange", "alpha_seg": 0.70},
    }

    for p_idx, res in enumerate(results):
        ax = axes[p_idx + 1]
        name = res["model_name"]
        style = model_styles.get(name, {"rep_color": "red", "alpha_seg": 0.6})
        segs = res["segments"]
        labels = res["labels"]
        reps = res["reps"]
        m = res["metrics"]

        unique_l = [c for c in np.unique(labels) if c >= 0]
        n_clusters = len(unique_l)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(1, n_clusters)))

        # Plot segments
        for s_i, seg in enumerate(segs):
            lbl = labels[s_i]
            if lbl == -1:
                ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], color="lightgray", alpha=0.25, linewidth=0.9)
            else:
                c_idx = unique_l.index(lbl)
                ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], color=cmap[c_idx], alpha=style["alpha_seg"], linewidth=1.6)

        # Plot representative trajectories
        for c, rep in reps.items():
            ax.plot(rep[:, 0], rep[:, 1], color=style["rep_color"], linewidth=3.2, zorder=10)

        # Subtitle metrics
        sil_str = f"Sil: {m['silhouette']:.3f}" if not np.isnan(m["silhouette"]) else "Sil: N/A"
        dbi_str = f"DBI: {m['davies_bouldin']:.3f}" if not np.isnan(m["davies_bouldin"]) else "DBI: N/A"
        time_str = f"t_tot: {res['t_total']:.3f}s"
        ax.set_title(
            f"{name}\nClusters: {n_clusters} (Noise: {m['noise_ratio']*100:.1f}%)\n{sil_str} | {dbi_str} | {time_str}",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.set_xlabel("X coordinate")
        ax.grid(True, linestyle="--", alpha=0.35)

        # Inset for Quantum model showing CTQW probability interference transition matrix P_ij
        if "Quantum" in name and res["P"] is not None and res["P"].shape[0] > 0:
            P_mat = res["P"]
            sub_n = min(60, P_mat.shape[0])
            ax_inset = inset_axes(ax, width="32%", height="32%", loc="upper right", borderpad=0.6)
            im = ax_inset.imshow(P_mat[:sub_n, :sub_n], cmap="magma", interpolation="nearest", aspect="auto")
            ax_inset.set_title(f"CTQW P(t) ({sub_n}x{sub_n})", fontsize=7.5, color="black", pad=2)
            ax_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()

    # Save to figures/ and artifacts/
    fig_path = FIGURES_DIR / output_filename
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200)

    if ARTIFACTS_DIR.exists():
        art_path = ARTIFACTS_DIR / output_filename
        plt.savefig(art_path, dpi=200)

    plt.close()
    print(f"  -> Saved figure: {fig_path}")


def plot_master_summary_dashboard(all_dataset_results: Dict[str, List[Dict[str, Any]]]):
    """Render comprehensive multi-dataset comparative dashboard."""
    datasets = list(all_dataset_results.keys())
    model_names = ["Original TRACLUS (2007)", "Fast-TRACLUS (2026)", "Quantum Fast-TRACLUS (CTQW)"]
    colors = ["#e74c3c", "#3498db", "#f39c12"]

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    x = np.arange(len(datasets))
    bar_width = 0.26

    # 1. Silhouette Scores (Higher is better)
    ax1 = axes[0, 0]
    for m_i, m_name in enumerate(model_names):
        vals = []
        for d in datasets:
            res = next(r for r in all_dataset_results[d] if r["model_name"] == m_name)
            v = res["metrics"]["silhouette"]
            vals.append(v if not np.isnan(v) else 0.0)
        ax1.bar(x + (m_i - 1) * bar_width, vals, width=bar_width, label=m_name, color=colors[m_i], alpha=0.85)

    ax1.set_title("Internal Cluster Quality: Silhouette Score (Higher ↑)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Silhouette Score")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper right")

    # 2. Davies-Bouldin Index (Lower is better)
    ax2 = axes[0, 1]
    for m_i, m_name in enumerate(model_names):
        vals = []
        for d in datasets:
            res = next(r for r in all_dataset_results[d] if r["model_name"] == m_name)
            v = res["metrics"]["davies_bouldin"]
            vals.append(min(v, 6.0) if not np.isnan(v) else 6.0)
        ax2.bar(x + (m_i - 1) * bar_width, vals, width=bar_width, label=m_name, color=colors[m_i], alpha=0.85)

    ax2.set_title("Cluster Separation: Davies-Bouldin Index (Lower ↓)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Davies-Bouldin Index")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="upper right")

    # 3. Total Execution Time (Log Scale)
    ax3 = axes[1, 0]
    for m_i, m_name in enumerate(model_names):
        vals = []
        for d in datasets:
            res = next(r for r in all_dataset_results[d] if r["model_name"] == m_name)
            vals.append(res["t_total"])
        ax3.bar(x + (m_i - 1) * bar_width, vals, width=bar_width, label=m_name, color=colors[m_i], alpha=0.85)

    ax3.set_yscale("log")
    ax3.set_title("Total Runtime Efficiency (Partition + Grouping, Log Scale ↓)", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ax3.set_ylabel("Execution Time (seconds, log scale)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9, loc="upper right")

    # 4. Cluster Count & Noise Ratio
    ax4 = axes[1, 1]
    for m_i, m_name in enumerate(model_names):
        c_vals = []
        for d in datasets:
            res = next(r for r in all_dataset_results[d] if r["model_name"] == m_name)
            c_vals.append(res["metrics"]["n_clusters"])
        ax4.plot(x, c_vals, marker="o", linewidth=2.0, label=f"{m_name} (Clusters)", color=colors[m_i])

    ax4.set_title("Discovered Clusters Across Benchmark Manifolds", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ax4.set_ylabel("Number of Clusters")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    master_path = FIGURES_DIR / "master_all_models_summary.png"
    plt.savefig(master_path, dpi=200)

    if ARTIFACTS_DIR.exists():
        plt.savefig(ARTIFACTS_DIR / "master_all_models_summary.png", dpi=200)

    plt.close()
    print(f"  -> Saved master summary dashboard: {master_path}")


def print_master_results_table(all_dataset_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Format all results into a single comprehensive Markdown table."""
    headers = [
        "Dataset",
        "Algorithm",
        "Partition (s)",
        "Grouping (s)",
        "Total (s)",
        "Silhouette ↑",
        "Davies-B ↓",
        "Clusters",
        "Noise (%)",
        "Quantum Contrast C",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for d_name, results in all_dataset_results.items():
        first = True
        for r in results:
            d_col = d_name if first else ""
            first = False
            name = r["model_name"].replace("Original ", "").replace(" (CTQW)", "")
            t_p = f"{r['t_part']:.3f}"
            t_g = f"{r['t_group']:.3f}"
            t_tot = f"{r['t_total']:.3f}"
            m = r["metrics"]
            sil = f"{m['silhouette']:.3f}" if not np.isnan(m["silhouette"]) else "N/A"
            dbi = f"{m['davies_bouldin']:.3f}" if not np.isnan(m["davies_bouldin"]) else "N/A"
            n_c = str(m["n_clusters"])
            noise = f"{m['noise_ratio']*100:.1f}%"
            if not np.isnan(m["contrast_ratio"]):
                c_str = f"{m['contrast_ratio']:.1f}x" if m["contrast_ratio"] < 1e5 else ">10^5x"
            else:
                c_str = "—"

            row = [d_col, name, t_p, t_g, t_tot, sil, dbi, n_c, noise, c_str]
            lines.append("| " + " | ".join(row) + " |")

    table_md = "\n".join(lines)
    print("\n" + "=" * 90)
    print("MASTER CROSS-MODEL BENCHMARK TABLE")
    print("=" * 90)
    print(table_md)
    print("=" * 90 + "\n")
    return table_md


def main():
    parser = argparse.ArgumentParser(description="Run all combined benchmarks on TRACLUS, Fast-TRACLUS, and Quantum Fast-TRACLUS.")
    parser.add_argument("--dry-run", action="store_true", help="Run with lighter downsampling for fast testing.")
    args = parser.parse_args()

    print("=" * 90)
    print("EXECUTING FULL CROSS-MODEL BENCHMARK SUITE")
    print("Models: Original TRACLUS | Fast-TRACLUS | Quantum Fast-TRACLUS (CTQW)")
    print("Datasets: 8 Real and Synthetic Trajectory Datasets")
    print("=" * 90)

    # Define test suite configs
    configs = [
        {
            "id": "1_hurricane",
            "title": "Hurricane Best Track (1950-2004)",
            "loader": lambda: load_hurricane_data(n_trajectories=120 if args.dry_run else 570)[0],
            "eps": 30.0,
            "min_lines": 6,
            "tau": 0.04,
        },
        {
            "id": "2_elk1993",
            "title": "Starkey Elk1993 Movement",
            "loader": lambda: [t[::5] for t in load_elk_data()[0]],
            "eps": 27.0,
            "min_lines": 8,
            "tau": 0.03,
        },
        {
            "id": "3_deer1995",
            "title": "Starkey Deer1995 Movement",
            "loader": lambda: [t[::4] for t in load_deer_data()[0]],
            "eps": 29.0,
            "min_lines": 8,
            "tau": 0.03,
        },
        {
            "id": "4_synthetic_noise",
            "title": "Synthetic Corridors (25% Noise)",
            "loader": lambda: load_synthetic_noise_data(noise_ratio=0.25)[0],
            "eps": 8.0,
            "min_lines": 4,
            "tau": 0.05,
        },
        {
            "id": "5_taxi",
            "title": "Porto Urban Taxi (100 Trajectories)",
            "loader": lambda: load_fast_traclus_dataset("taxi", n_trajectories=100)[0],
            "eps": 1.5,
            "min_lines": 4,
            "tau": 0.05,
        },
        {
            "id": "6_movebank",
            "title": "Movebank Wildlife (100 Trajectories)",
            "loader": lambda: load_fast_traclus_dataset("movebank", n_trajectories=100)[0],
            "eps": 2.0,
            "min_lines": 4,
            "tau": 0.04,
        },
        {
            "id": "7_geolife",
            "title": "GeoLife Pedestrians (100 Trajectories)",
            "loader": lambda: load_fast_traclus_dataset("geolife", n_trajectories=100)[0],
            "eps": 1.5,
            "min_lines": 4,
            "tau": 0.05,
        },
        {
            "id": "8_dual_spirals",
            "title": "Dual Interlocking Spirals (Non-Convex)",
            "loader": lambda: generate_dual_spirals(n_trajectories_per_corridor=8, n_points=35, noise=0.10, random_state=42)[0],
            "eps": 6.0,
            "min_lines": 3,
            "tau": 0.02,
        },
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for cfg in configs:
        ds_id = cfg["id"]
        ds_title = cfg["title"]
        eps = cfg["eps"]
        min_lines = cfg["min_lines"]
        tau = cfg["tau"]

        print(f"\n[{ds_id.upper()}] Loading {ds_title}...")
        trajs = cfg["loader"]()
        total_pts = sum(len(t) for t in trajs)
        print(f"  Loaded {len(trajs)} trajectories ({total_pts} total GPS points).")

        ds_res = []

        # Model 1: Original TRACLUS
        print("  -> Running Model 1: Original TRACLUS (2007)...")
        res_traclus = run_traclus(trajs, eps=eps, min_lines=min_lines)
        ds_res.append(res_traclus)
        print(f"     Done: Part={res_traclus['t_part']:.3f}s, Group={res_traclus['t_group']:.3f}s, Clusters={res_traclus['metrics']['n_clusters']}")

        # Model 2: Fast-TRACLUS
        print("  -> Running Model 2: Fast-TRACLUS (2026)...")
        res_fast = run_fast_traclus(trajs, eps=eps, min_lines=min_lines)
        ds_res.append(res_fast)
        print(f"     Done: Part={res_fast['t_part']:.3f}s, Group={res_fast['t_group']:.3f}s, Clusters={res_fast['metrics']['n_clusters']}")

        # Model 3: Quantum Fast-TRACLUS (CTQW)
        print("  -> Running Model 3: Quantum Fast-TRACLUS (Qiskit CTQW)...")
        res_quantum = run_quantum_fast_traclus(trajs, eps=eps, min_lines=min_lines, tau=tau)
        ds_res.append(res_quantum)
        print(f"     Done: Part={res_quantum['t_part']:.3f}s, Group={res_quantum['t_group']:.3f}s, Clusters={res_quantum['metrics']['n_clusters']}")

        # Plot 4-Panel comparison
        out_fig = f"comparison_{ds_id}.png"
        plot_4panel_dataset_comparison(ds_title, trajs, ds_res, out_fig)

        all_results[ds_title.split("(")[0].strip()] = ds_res

    # Plot master summary dashboard
    print("\n[Generating Master Summary Dashboard]...")
    plot_master_summary_dashboard(all_results)

    # Print master markdown table
    table_md = print_master_results_table(all_results)

    # Save table to reports
    report_file = FIGURES_DIR / "benchmark_table.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(table_md)

    print("ALL BENCHMARK FIGURES AND METRIC TABLES GENERATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
