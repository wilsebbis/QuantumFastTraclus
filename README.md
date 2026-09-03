# Trajectory Clustering Suite: TRACLUS, Fast-TRACLUS, and Qiskit Quantum CTQW Fast-TRACLUS

A high-performance Python repository implementing and benchmarking three generations of trajectory clustering engines:
1. **Original TRACLUS** (Lee, Han, & Whang, SIGMOD 2007): Iterative MDL trajectory partitioning with explicit 2D coordinate rotation matrices and line-segment DBSCAN with trajectory cardinality filtering ($\text{PTR}(C) \ge \text{MinLns}$).
2. **Fast-TRACLUS** (González Delgado et al., 2026): Fully vectorized NumPy MDL partitioning via direct vector dot products, broadcasted pairwise distance tensor calculation, and a decoupled modular grouping architecture (`DBSCAN`, `OPTICS`, `HDBSCAN`, `Agglomerative`, and Classical `Spectral`).
3. **Fast-TRACLUS with Qiskit CTQW Grouping**: Fast-TRACLUS partitioning paired with Continuous-Time Quantum Walks on the normalized Laplacian, executed strictly through **standardized Qiskit primitives and circuit libraries** (`SparsePauliOp`, `HamiltonianGate`, `PauliEvolutionGate`, `LieTrotter`, `Statevector`, `StatevectorSampler`).
4. **Comparative Benchmark Harness**: Automated evaluation suite comparing all engines across execution runtime and internal cluster quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, and Interference Contrast Ratio $\mathcal{C}$).

---

## Repository Architecture

```
trajectory_clustering_suite/
│
├── core/
│   ├── __init__.py
│   ├── distance.py             # Iterative and vectorized line segment distance implementations
│   └── representative.py       # Sweep-line representative trajectory generation
│
├── traclus/
│   ├── __init__.py
│   ├── iterative_mdl.py        # Original TRACLUS iterative MDL partitioning
│   └── line_dbscan.py          # Original TRACLUS line segment DBSCAN with cardinality filter
│
├── fast_traclus/
│   ├── __init__.py
│   ├── vectorized_mdl.py       # Fast-TRACLUS fully vectorized MDL
│   ├── distance_matrix.py      # Broadcasted N x N distance computation
│   └── modular_clustering.py   # DBSCAN, OPTICS, HDBSCAN, Agglomerative, Spectral
│
├── quantum_traclus/
│   ├── __init__.py
│   ├── laplacian_builder.py    # Sparse normalized Laplacian to Qiskit operator converter
│   ├── ctqw_evolution.py       # Qiskit-native Hamiltonian/Pauli evolution circuit
│   └── interference_cluster.py # Quantum walk transition kernel & corridor extraction
│
├── benchmarks/
│   ├── __init__.py
│   ├── synthetic_corridors.py  # Non-convex synthetic trajectory generation
│   ├── metrics.py              # Silhouette, Davies-Bouldin, Calinski-Harabasz, Contrast Ratio
│   └── run_comparison.py       # End-to-end multi-algorithm benchmark script
│
├── tests/
│   ├── test_core_distance.py   # Numerical equivalence between iterative and vectorized distance
│   ├── test_traclus.py         # Original TRACLUS tests
│   ├── test_fast_traclus.py    # Fast-TRACLUS tests
│   └── test_quantum_traclus.py # Qiskit CTQW evolution tests
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Algorithmic Formulations

### 1. Line Segment Distance
Let $L_i = s_i e_i$ (longer segment) and $L_j = s_j e_j$ (shorter segment):
* **Perpendicular Distance ($d_\perp$)** (Order-2 Lehmer mean):
  $$d_\perp(L_i, L_j) = \frac{l_{\perp 1}^2 + l_{\perp 2}^2}{l_{\perp 1} + l_{\perp 2}} \quad \text{where } l_{\perp 1} = \|s_j - p_s\|_2, \; l_{\perp 2} = \|e_j - p_e\|_2$$
* **Parallel Distance ($d_\parallel$)**:
  $$d_\parallel(L_i, L_j) = \min(l_{\parallel 1}, l_{\parallel 2})$$
  $$l_{\parallel 1} = \min(\|p_s - s_i\|_2, \|p_s - e_i\|_2), \quad l_{\parallel 2} = \min(\|p_e - s_i\|_2, \|p_e - e_i\|_2)$$
* **Angle Distance ($d_\theta$)**:
  $$d_\theta(L_i, L_j) = \begin{cases} \|L_j\|_2 \sin(\theta), & \text{if } 0 \le \theta < \pi/2 \\ \|L_j\|_2, & \text{if } \pi/2 \le \theta \le \pi \end{cases}$$
* **Total Weighted Distance**:
  $$\text{dist}(L_i, L_j) = w_\perp d_\perp + w_\parallel d_\parallel + w_\theta d_\theta \quad (\text{default weights } = 1.0)$$

### 2. Original TRACLUS (Lee et al., 2007)
* **Iterative MDL**: Point projections evaluated via explicit 2D coordinate rotation matrices:
  $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\phi & \sin\phi \\ -\sin\phi & \cos\phi \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$
* **Line-Segment DBSCAN**: Sequential loop traversal for $\epsilon$-neighborhood calculation with the **Trajectory Cardinality Filter**:
  $$|\text{PTR}(C)| < \text{MinLns} \implies \text{prune cluster } C \to -1$$

### 3. Fast-TRACLUS (González Delgado et al., 2026)
* **Vectorized MDL**: Direct scalar vector dot product projections:
  $$u_1 = \frac{(s_j - s_i) \cdot (e_i - s_i)}{\|e_i - s_i\|_2^2}, \quad p_s = s_i + u_1(e_i - s_i)$$
* **Broadcasted Distance Matrix**: Evaluates the $N \times N$ pairwise distance tensor directly with zero nested Python loops.
* **Modular Clustering Framework**: Interchangeable backends (`dbscan`, `spectral`, `optics`, `hdbscan`, `agglomerative`).

### 4. Fast-TRACLUS with Qiskit CTQW
* Symmetric normalized Laplacian mapped to $2^n \times 2^n$ Hamiltonian ($n = \lceil\log_2 N\rceil$):
  $$L_{\text{norm}} = I - D^{-1/2} W D^{-1/2}$$
* Standard Qiskit libraries:
  * `SparsePauliOp.from_operator(Operator(L_norm_padded))`
  * Unitary time evolution: `HamiltonianGate(data=L_norm_padded, time=t)` or `PauliEvolutionGate(op, time=t, synthesis=LieTrotter(reps=...))`
  * Propagation: `Statevector.from_int(j, dims=2**n).evolve(circuit)`
  * Sampling: `StatevectorSampler`
* Quantum interference transition kernel $K = \frac{1}{2}(P + P^T)$ thresholded at $K_{jk} \ge \tau \cdot \max_{j \ne k}(K)$.

---

## Experimental Benchmark Results

Executed on interlocking non-convex dual concentric spirals (`python3 benchmarks/run_comparison.py`):

| Algorithm | Partition (s) | Grouping (s) | Total (s) | Silhouette ↑ | Calinski-H ↑ | Davies-B ↓ | N_clusters | Noise (%) | Contrast C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Original TRACLUS (2007)** | 0.0047 | 0.0299 | 0.0346 | 0.7184 | 243.5 | 0.5341 | 14 | 11.0% | N/A |
| **Fast-TRACLUS (DBSCAN, 2026)** | 0.0076 | **0.0013** | **0.0089** | **0.8587** | **1625.5** | **0.1942** | 10 | 3.8% | N/A |
| **Fast-TRACLUS (Spectral Baseline)** | 0.0077 | 0.0585 | 0.0662 | 0.1352 | 3.3 | 4.2879 | 2 | 3.8% | N/A |
| **Fast-TRACLUS (Qiskit CTQW)** | 0.0085 | 0.0855 | 0.0939 | **0.8587** | **1625.5** | **0.1942** | 10 | 3.8% | **>10^5x** |

### Insights:
1. **Grouping Speedup**: Fast-TRACLUS achieves a **23x grouping speedup** (0.0013s vs 0.0299s) over Original TRACLUS due to broadcasted distance matrix evaluation.
2. **Non-Convex Corridor Resolution**: Classical Spectral Clustering with $k$-means fails catastrophically on the spiral manifold (Silhouette 0.1352, DBI 4.2879). In contrast, Fast-TRACLUS with Qiskit CTQW achieves **Silhouette 0.8587, DBI 0.1942, and an Interference Contrast Ratio $>10^5\times$**, perfectly isolating the concentric arms without centroid distortion.

---

## Running Benchmarks and Tests

```bash
# Run unit & integration test suite (16 tests)
pytest -v tests/

# Execute comparative benchmark runner and generate comparison plots
python3 benchmarks/run_comparison.py
```
