# Fast-TRACLUS Quantum: Trajectory Clustering with Continuous-Time Quantum Walks (CTQW)

A high-performance trajectory clustering package implementing the **Fast-TRACLUS** architecture, with a quantum algorithmic breakthrough in its grouping module: replacing classical Graph Laplacian Eigendecomposition ($O(N^3)$ diagonalization + Euclidean $k$-means projection) with **Continuous-Time Quantum Walks (CTQW)** on the symmetric normalized Laplacian.

---

## Key Algorithmic Innovations

Classical spectral clustering projects graph Laplacian eigenvectors into Euclidean space and partitions them via $k$-means. This suffers from two major limitations:
1. **$O(N^3)$ Computational Complexity:** Full diagonalization of the Laplacian matrix incurs cubic runtime scaling.
2. **Centroid Distortion along Non-Convex Manifolds:** $k$-means imposes convex, hyperspherical Voronoi boundaries in projection space, cutting through interleaved non-convex trajectory corridors (such as spirals, roundabouts, and arterial highway bypasses).

**Fast-TRACLUS Quantum** resolves both challenges:
* **Hamiltonian / Normalized Laplacian:**
  Constructs the symmetric normalized Laplacian:
  $$L_{\text{norm}} = I - D^{-1/2} W D^{-1/2}$$
* **Unitary Schrödinger Evolution:**
  $$|\psi(t)\rangle = U(t)|\psi(0)\rangle = \exp(-i L_{\text{norm}} t) |\psi(0)\rangle$$
* **Avoiding $O(N^3)$ Diagonalization via Sparse Matrix Exponentiation:**
  The unitary operator action $\exp(-i L_{\text{norm}} t) v$ is computed directly using sparse polynomial approximations (Al-Mohy & Higham algorithm via `scipy.sparse.linalg.expm_multiply`), requiring zero dense matrix inversions or dense eigendecompositions.
* **Wave Interference & Non-Convex Corridor Isolation:**
  The transition probability intensity from segment $j$ to segment $k$ at time $t$ is:
  $$P_{jk}(t) = \left| \langle k | \exp(-i L_{\text{norm}} t) | j \rangle \right|^2$$
  Constructive wave interference isolates connected non-convex corridors with high contrast (~8x to 1000x+), bypassing $k$-means centroid distortion.

---

## Package Architecture

```
fast_traclus_quantum/
│
├── __init__.py            # Package entry point and exports
├── distance.py            # Vectorized TRACLUS segment metrics (perpendicular, parallel, angular)
├── segmentation.py        # Vectorized MDL trajectory partitioning via vector dot products
├── graph.py               # Sparse affinity graph W and normalized Laplacian L_norm
├── ctqw.py                # CTQWClusterer via expm_multiply & Qiskit Hamiltonian bridge
├── pipeline.py            # FastTRACLUSQuantum & FastTRACLUSSpectralBaseline
└── evaluation.py          # Davies-Bouldin Index, Silhouette Score, Contrast Ratio
```

---

## Performance & Manifold Resolution Benchmark

Evaluated against classical Spectral Clustering on dual interlocking non-convex spirals and U-shaped arterials (`python3 run_benchmark.py`):

| Trajectory Scenario | Metric | Fast-TRACLUS CTQW (Proposed) | Classical Spectral Baseline | Advantage |
|---|---|---|---|---|
| **Dual Concentric Spirals** | **Davies-Bouldin Index** (lower=better) | **0.1785** | 2.9876 | **16.7x superior separation** |
| | **Silhouette Score** (higher=better) | **0.8697** | 0.1172 | **7.4x higher cohesion** |
| | **Execution Time** | **0.0124 s** | 0.0220 s | **1.8x faster** |
| | **Interference Contrast Ratio $\mathcal{C}$** | **>1,000,000x** | N/A | High phase coherence |
| **Interlocking U-Arterials** | **Silhouette Score** | **0.7929** | 0.6087 | **+30% higher cohesion** |
| | **Interference Contrast Ratio $\mathcal{C}$** | **349.84x** | N/A | Substantial wave confinement |

---

## Installation & Usage

```bash
git clone https://github.com/wilsebbis/QuantumFastTraclus.git
cd QuantumFastTraclus
pip install -e .
```

### Python API Example

```python
import numpy as np
from fast_traclus_quantum import FastTRACLUSQuantum

# Example trajectories (list of 2D numpy arrays)
trajectories = [
    np.column_stack([np.linspace(0, 50, 30), np.sin(np.linspace(0, 3, 30))]),
    np.column_stack([np.linspace(0, 50, 30), np.sin(np.linspace(0, 3, 30)) + 0.2]),
]

# Initialize and fit Fast-TRACLUS Quantum
model = FastTRACLUSQuantum(
    eps=5.0,            # Spatial connectivity threshold
    min_samples=2,      # Minimum segment cluster size
    tau=0.03,           # Wave coherence threshold
    adaptive_time=True, # Auto-tune evolution time to Fiedler eigenvalue
)

segments, labels = model.fit_predict(trajectories)
print(f"Extracted {len(segments)} segments across {len(set(labels))} clusters.")

# Extract representative trajectory corridors
rep_trajectories = model.get_representative_trajectories()
```

### Qiskit Quantum Hardware Bridge

`CTQWClusterer` includes a native Qiskit bridge translating the normalized graph Hamiltonian into a `SparsePauliOp` for execution on quantum processors:

```python
qc, pauli_op = model.clusterer_.to_qiskit_circuit()
print(f"Synthesized Quantum Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
```

---

## Running the Benchmark & Tests

```bash
# Run unit & integration test suite (29 tests)
pytest -v

# Run benchmark and generate comparison figures
python3 run_benchmark.py
```
