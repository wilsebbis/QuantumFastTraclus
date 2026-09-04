"""
Download and Extraction Runner for Fast-TRACLUS Benchmark Datasets
==================================================================
1. Porto Taxi (ECML PKDD 2015 - UCI Machine Learning Repository)
   - Downloads full archive: taxi service trajectory prediction challenge ecml pkdd 2015.zip
   - Extracts train.csv.zip -> train.csv
   - Generates progressive subsets: taxi_100.tra, taxi_200.tra, taxi_300.tra, taxi_400.tra, taxi_500.tra

2. Microsoft GeoLife GPS Trajectories v1.3 (Figshare Verified Archive)
   - Downloads full archive: Geolife Trajectories 1.3.zip (352 MB)
   - Extracts all user PLT trajectory logs (182 users)
   - Generates progressive subsets: geolife_100.tra, geolife_200.tra, geolife_300.tra, geolife_400.tra, geolife_500.tra
"""

import os
import sys
import json
import zipfile
import urllib.request
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "fast_traclus")
os.makedirs(OUT_DIR, exist_ok=True)

UCI_TAXI_URL = "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip"
GEOLIFE_URL = "https://ndownloader.figshare.com/files/45571758"

def download_with_progress(url, dest_path, label):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        print(f"[{label}] File already downloaded: {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")
        return True

    print(f"[{label}] Starting download from: {url}")
    print(f"[{label}] Saving to: {dest_path}")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    )
    
    try:
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)
                    sys.stdout.write(f"\r  -> [{label}] {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
                else:
                    sys.stdout.write(f"\r  -> [{label}] {downloaded / (1024*1024):.1f} MB downloaded")
                sys.stdout.flush()
        print(f"\n[{label}] Download finished successfully: {dest_path}")
        return True
    except Exception as e:
        print(f"\n[{label}] Download error: {e}")
        return False

def process_taxi():
    zip_path = os.path.join(OUT_DIR, "taxi_service_ecml_pkdd_2015.zip")
    extract_dir = os.path.join(OUT_DIR, "raw_taxi")
    os.makedirs(extract_dir, exist_ok=True)
    
    if not download_with_progress(UCI_TAXI_URL, zip_path, "Taxi"):
        return
    
    # Extract outer archive
    print("[Taxi] Extracting main zip archive...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
        
    # Look for train.csv or train.csv.zip
    train_csv = os.path.join(extract_dir, "train.csv")
    train_zip = os.path.join(extract_dir, "train.csv.zip")
    
    if not os.path.exists(train_csv) and os.path.exists(train_zip):
        print("[Taxi] Extracting train.csv.zip...")
        with zipfile.ZipFile(train_zip, 'r') as z:
            z.extractall(extract_dir)
            
    if not os.path.exists(train_csv):
        # Scan for train.csv in subfolders
        for root, _, files in os.walk(extract_dir):
            if "train.csv" in files:
                train_csv = os.path.join(root, "train.csv")
                break
                
    if not os.path.exists(train_csv):
        print(f"[Taxi] Error: train.csv not found in {extract_dir}")
        return

    print(f"[Taxi] Found train.csv: {train_csv} ({os.path.getsize(train_csv) / (1024*1024):.1f} MB)")
    print("[Taxi] Extracting progressive subsets (100, 200, 300, 400, 500 trajectories)...")
    
    subsets = [100, 200, 300, 400, 500]
    max_needed = max(subsets)
    trajs = []
    
    with open(train_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('MISSING_DATA') == 'True':
                continue
            poly_str = row.get('POLYLINE', '').strip()
            if not poly_str or poly_str == '[]':
                continue
            try:
                pts = json.loads(poly_str)
                # Keep valid trips with >= 10 points
                if len(pts) >= 10:
                    trajs.append(pts)
                    if len(trajs) >= max_needed:
                        break
            except Exception:
                continue

    for k in subsets:
        out_file = os.path.join(OUT_DIR, f"taxi_{k}.tra")
        actual_k = min(k, len(trajs))
        with open(out_file, 'w') as f:
            f.write("2\n")
            f.write(f"{actual_k}\n")
            for tid, pts in enumerate(trajs[:actual_k]):
                pts_flat = ' '.join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts)
                f.write(f"{tid} {len(pts)} {pts_flat}\n")
        total_pts = sum(len(p) for p in trajs[:actual_k])
        print(f"  -> Generated {out_file}: {actual_k} trajectories, {total_pts} points")

def process_geolife():
    zip_path = os.path.join(OUT_DIR, "Geolife_Trajectories_1.3.zip")
    extract_dir = os.path.join(OUT_DIR, "raw_geolife")
    os.makedirs(extract_dir, exist_ok=True)
    
    if not download_with_progress(GEOLIFE_URL, zip_path, "GeoLife"):
        return
        
    print("[GeoLife] Extracting zip archive...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
        
    print("[GeoLife] Discovering PLT trajectory files...")
    plt_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".plt"):
                plt_files.append(os.path.join(root, f))
                
    plt_files.sort()
    print(f"[GeoLife] Discovered {len(plt_files)} raw trajectory files across users.")
    
    subsets = [100, 200, 300, 400, 500]
    max_needed = max(subsets)
    trajs = []
    
    for plt_path in plt_files:
        try:
            with open(plt_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            # First 6 lines in GeoLife PLT are headers
            pts = []
            for line in lines[6:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    lat, lon = float(parts[0]), float(parts[1])
                    pts.append((lon, lat))  # x=lon, y=lat
            if len(pts) >= 10:
                trajs.append(pts)
                if len(trajs) >= max_needed:
                    break
        except Exception:
            continue
            
    print(f"[GeoLife] Extracted {len(trajs)} trajectories.")
    for k in subsets:
        out_file = os.path.join(OUT_DIR, f"geolife_{k}.tra")
        actual_k = min(k, len(trajs))
        with open(out_file, 'w') as f:
            f.write("2\n")
            f.write(f"{actual_k}\n")
            for tid, pts in enumerate(trajs[:actual_k]):
                pts_flat = ' '.join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts)
                f.write(f"{tid} {len(pts)} {pts_flat}\n")
        total_pts = sum(len(p) for p in trajs[:actual_k])
        print(f"  -> Generated {out_file}: {actual_k} trajectories, {total_pts} points")

if __name__ == "__main__":
    print("=== Fast-TRACLUS Raw Dataset Downloader & Extractor ===")
    print("Step 1: Processing Porto Taxi (ECML PKDD 2015)...")
    process_taxi()
    print("\nStep 2: Processing Microsoft GeoLife v1.3...")
    process_geolife()
    print("\n=== All Fast-TRACLUS downloads and extractions complete ===")
