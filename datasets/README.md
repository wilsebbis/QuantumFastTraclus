# TRACLUS & Fast-TRACLUS Benchmark Datasets

This directory contains the experimental datasets, raw archives, and progressive evaluation subsets for replicating:
1. **Original TRACLUS** (*Lee, Han, & Whang*, SIGMOD 2007: *Trajectory Clustering: A Partition-and-Group Framework*)
2. **Fast-TRACLUS** (*González Delgado, Porras Alfonso, Baruque Zanón, & Cogollos Adrian*, 2026: *Fast-TRACLUS: An Optimized Trajectory Clustering Algorithm for Large-Scale Datasets*)

---

## 1. Original TRACLUS Datasets (`original_traclus/`)

All files are stored in the canonical `.tra` format:
- Line 1: Dimensionality $d$ (2 for 2D spatial coordinates).
- Line 2: Number of trajectories $N$.
- Lines $3 \dots N+2$: `<trajectory_id> <num_points> <x_1> <y_1> <x_2> <y_2> ... <x_n> <y_n>`

| File | Source | Trajectories | Total Points | Attributes | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `hurricane1950_2004.tra` | National Hurricane Center / Unisys Weather Atlantic Best Track (1950–2004) | **570** | **17,736** | 6-hourly Lat/Lon (scaled) | **Exact Match** (Paper Section 5.1: 570 trajectories, 17,736 points) |
| `hurricane1950_2006.tra` | Unisys Atlantic Hurricane Database extended (1950–2006) | **608** | **18,951** | 6-hourly Lat/Lon | **Complete Archive** |
| `elk_1993.tra` | Starkey Project Radio-Telemetry Database (1993) | **33** | **47,204** | Projected planar X/Y (UTM) | **Exact Match** (Paper Section 5.1: 33 trajectories, 47,204 points) |
| `deer1995.tra` | Starkey Project Radio-Telemetry Database (1995) | **32** | **20,065** | Projected planar X/Y (UTM) | **Exact Match** (Paper Section 5.1: 32 trajectories, 20,065 points) |
| `synthetic_noise.tra` | Synthetic Corridor & Outlier Benchmark (Section 5.5, Figure 23) | **200** | **3,724** | 2D planar X/Y | **Exact Match** (150 corridor paths + 50 random paths = 25% noise) |

---

## 2. Fast-TRACLUS Datasets (`fast_traclus/`)

### A. Porto Taxi Dataset (ECML PKDD 2015)
- **Source**: [UCI Machine Learning Repository #339](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015)
- **Raw Archive**: `taxi_service_ecml_pkdd_2015.zip` (509 MB)
- **Extracted Content**: `raw_taxi/train.csv` (1.8 GB, 1.7 million trajectories across 442 taxis)
- **Subsets Generated**:
  - `taxi_100.tra` (100 trajectories, 4,996 points)
  - `taxi_200.tra` (200 trajectories, 10,794 points)
  - `taxi_300.tra` (300 trajectories, 16,633 points)
  - `taxi_400.tra` (400 trajectories, 22,544 points)
  - `taxi_500.tra` (500 trajectories, 28,499 points)

### B. Microsoft GeoLife GPS Trajectories v1.3
- **Source**: [Microsoft Research GeoLife](https://www.microsoft.com/en-us/download/details.aspx?id=52367) / [Figshare](https://figshare.com/articles/dataset/Geolife_Trajectories_1_3_zip/25577268)
- **Raw Archive**: `Geolife_Trajectories_1.3.zip` (336 MB)
- **Extracted Content**: `raw_geolife/Data/` (18,670 raw `.plt` trajectories across 182 users)
- **Subsets Generated**:
  - `geolife_100.tra` (100 trajectories, 91,629 points)
  - `geolife_200.tra` (200 trajectories, 228,142 points)
  - `geolife_300.tra` (300 trajectories, 399,113 points)
  - `geolife_400.tra` (400 trajectories, 518,928 points)
  - `geolife_500.tra` (500 trajectories, 683,237 points)

### C. Wildlife Tracking Data (Movebank)
- **Source**: [Kaggle Animal Tracking Dataset](https://www.kaggle.com/datasets/frasonfrancis/animal-tracking-data-set)
- **Subsets Targeted**: 100, 200, 300, and 371 trajectories.
- **Manual Acquisition**: Download directly via browser from Kaggle or via Kaggle CLI:
  ```bash
  kaggle datasets download -d frasonfrancis/animal-tracking-data-set -p /Users/wilsebbis/Developer/New_QRLSTC_Paper/datasets/fast_traclus/
  ```

---

## 3. Automation Scripts

- `download_and_extract_fast_traclus.py`: Full end-to-end download, unzipping, and subset extraction script for Taxi and GeoLife.
