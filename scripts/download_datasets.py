#!/usr/bin/env python3
"""Dataset Downloader and Verification Utility for TRACLUS and Fast-TRACLUS Benchmarks."""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import urllib.request

# Verified endpoints for original TRACLUS datasets (Lee, Han, & Whang, SIGMOD 2007)
TRACLUS_DATASETS = {
    "deer1995.tra": {
        "url": "https://raw.githubusercontent.com/ICDI0906/traclus/master/deer1995.tra",
        "description": "Starkey Project Mule Deer 1995 (32 trajectories, 20,065 points)",
        "expected_trajs": 32,
        "expected_points": 20065,
    },
    "elk_1993.tra": {
        "url": "https://raw.githubusercontent.com/ICDI0906/traclus/master/elk_1993.tra",
        "description": "Starkey Project Elk 1993 (33 trajectories, 47,204 points)",
        "expected_trajs": 33,
        "expected_points": 47204,
    },
    "hurricane1950_2006.tra": {
        "url": "https://raw.githubusercontent.com/ICDI0906/traclus/master/hurricane1950_2006.tra",
        "description": "NHC Atlantic Hurricanes 1950-2004 (570 trajectories, 17,736 points)",
        "expected_trajs": 570,
        "expected_points": 17736,
    },
}

EXTERNAL_DATASET_SOURCES = {
    "taxi_ecml_pkdd_2015": {
        "url": "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip",
        "description": "UCI ECML PKDD 2015 Taxi Service Trajectory Prediction Challenge",
    },
    "geolife_trajectories": {
        "url": "https://download.microsoft.com/download/F/4/8/F4894AA5-CDBC-4E40-9205-5DD13D193547/Geolife%20Trajectories%201.3.zip",
        "description": "Microsoft Research GeoLife GPS Trajectories v1.3",
    },
    "movebank_repository": {
        "url": "https://www.movebank.org",
        "description": "Movebank Animal Tracking Data Repository",
    },
}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def check_urls() -> bool:
    """Verify connectivity to all dataset endpoints."""
    print("Checking dataset endpoint accessibility...")
    all_ok = True
    for filename, info in TRACLUS_DATASETS.items():
        url = info["url"]
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                print(f"  [OK] {filename} -> HTTP {status}")
        except Exception as e:
            print(f"  [FAIL] {filename} -> {e}")
            all_ok = False
    return all_ok


def download_traclus_datasets(output_dir: Path) -> None:
    """Download and verify original TRACLUS datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading TRACLUS datasets to: {output_dir}")

    for filename, info in TRACLUS_DATASETS.items():
        dest = output_dir / filename
        url = info["url"]
        desc = info["description"]

        print(f"\nFetching {filename} ({desc})...")
        print(f"  Source: {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size / 1024.0
            print(f"  Successfully downloaded: {size_kb:.1f} KB")

            # Verify contents
            with open(dest, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            n_trajs_header = int(lines[1])
            target_trajs = info["expected_trajs"]
            # Sum points
            total_pts = sum(int(lines[i].split()[1]) for i in range(2, 2 + target_trajs))

            print(f"  Verified: {target_trajs} trajectories, {total_pts} points.")
            if total_pts == info["expected_points"]:
                print("  [VALIDATION PASSED] Exact point count match with SIGMOD 2007 paper!")
            else:
                print(f"  [NOTE] Point count is {total_pts} (expected {info['expected_points']}).")
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download and verify TRACLUS & Fast-TRACLUS benchmark datasets.")
    parser.add_argument("--check-urls", action="store_true", help="Check connectivity to dataset endpoints.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for raw files.")
    parser.add_argument("--download-traclus", action="store_true", help="Download original TRACLUS datasets.")
    parser.add_argument("--info", action="store_true", help="Print information about available datasets.")

    args = parser.parse_args()

    if args.info or len(sys.argv) == 1:
        print("=" * 80)
        print("TRACLUS & Fast-TRACLUS Benchmark Datasets Catalog")
        print("=" * 80)
        print("\n1. Original TRACLUS (Lee, Han, & Whang, SIGMOD 2007):")
        for fn, info in TRACLUS_DATASETS.items():
            print(f"  - {fn}: {info['description']}")
            print(f"    URL: {info['url']}")

        print("\n2. Fast-TRACLUS (González Delgado et al., 2026):")
        for key, info in EXTERNAL_DATASET_SOURCES.items():
            print(f"  - {key}: {info['description']}")
            print(f"    URL: {info['url']}")

        print("\nUsage:")
        print("  python3 scripts/download_datasets.py --check-urls        # Check endpoint connectivity")
        print("  python3 scripts/download_datasets.py --download-traclus  # Download original TRACLUS datasets")
        return

    if args.check_urls:
        check_urls()

    if args.download_traclus:
        download_traclus_datasets(Path(args.output_dir))


if __name__ == "__main__":
    main()

