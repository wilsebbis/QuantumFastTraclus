"""
Fast-TRACLUS Dataset Acquisition & Subset Generator
===================================================
Automates acquisition and progressive subset extraction (100, 200, 300, 400, 500 trajectories)
for the three benchmark datasets used in Fast-TRACLUS (González Delgado et al., 2026):
  1. Taxi Movement Data (ECML PKDD 2015 - UCI / Kaggle)
  2. Wildlife Tracking Data (Movebank animal tracking - 371 trajectories)
  3. Pedestrian Movement Data (Microsoft Research GeoLife v1.3)
"""

import os
import sys
import json
import zipfile
import urllib.request
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAST_TRACLUS_DIR = os.path.join(BASE_DIR, "fast_traclus")
os.makedirs(FAST_TRACLUS_DIR, exist_ok=True)

UCI_TAXI_URL = "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip"
FIGSHARE_GEOLIFE_URL = "https://ndownloader.figshare.com/articles/25577268/versions/1"

def download_file(url, target_path, desc="file"):
    if os.path.exists(target_path):
        print(f"[{desc}] Already exists: {target_path}")
        return True
    print(f"[{desc}] Downloading from {url}...")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(100, blocknum * blocksize * 100 // totalsize)
                sys.stdout.write(f"\r  -> Progress: {percent}% ({blocknum*blocksize // (1024*1024)} MB)")
                sys.stdout.flush()
        urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
        print(f"\n[{desc}] Download complete: {target_path}")
        return True
    except Exception as e:
        print(f"\n[{desc}] Download failed: {e}")
        return False

def extract_taxi_subsets(csv_path, out_dir):
    """
    Extracts 100, 200, 300, 400, 500 trajectory subsets from Porto Taxi train.csv.
    Format: POLYLINE column contains [[lon, lat], [lon, lat], ...]
    """
    subsets = [100, 200, 300, 400, 500]
    max_needed = max(subsets)
    trajs = []
    
    print("[Taxi] Parsing trajectories from CSV...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('MISSING_DATA') == 'True':
                continue
            poly_str = row.get('POLYLINE', '').strip()
            if not poly_str or poly_str == '[]':
                continue
            try:
                coords = json.loads(poly_str)
                if len(coords) >= 10:  # Valid trajectory with minimum length
                    trajs.append(coords)
                    if len(trajs) >= max_needed:
                        break
            except Exception:
                continue

    print(f"[Taxi] Extracted {len(trajs)} valid trajectories.")
    for k in subsets:
        subset_file = os.path.join(out_dir, f"taxi_{k}.tra")
        with open(subset_file, 'w') as f:
            f.write("2\n")
            f.write(f"{min(k, len(trajs))}\n")
            for tid, pts in enumerate(trajs[:k]):
                pts_flat = ' '.join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts)
                f.write(f"{tid} {len(pts)} {pts_flat}\n")
        print(f"  -> Saved {subset_file} ({min(k, len(trajs))} trajectories)")

if __name__ == "__main__":
    print(f"Fast-TRACLUS Data Directory: {FAST_TRACLUS_DIR}")
    print("Run this script to retrieve and format the progressive subsets for Fast-TRACLUS.")
