"""Generate notebooks/pipeline_end_to_end_gpu.ipynb with clean cells."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "pipeline_end_to_end_gpu.ipynb"

cells = []

def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# ----------------- CELLS -----------------

cells.append(md("""# Floor Plan Scanner — End-to-End GPU Pipeline

This notebook runs the complete **floorplan-scanner** pipeline with GPU acceleration:
- Multi-tier 3D reconstruction: **Video**, **LiDAR**, and **Photo**.
- Structure-from-Motion (SfM) via COLMAP on CUDA GPU.
- Open-vocabulary damage detection and segmentation using **MobileSAM + NanoOWL**.
- 2D dimensioned floor plan rendering and JSON report generation.
"""))

cells.append(md("""## Step 0: Clone Repository (dev branch)
If running on **Google Colab** or **Kaggle**, clone the repository to get the full codebase (`run.py`, `src/`, etc.) and switch into the project directory.
*(If already running locally inside the repository, this step automatically detects it and continues).*
"""))

cells.append(code("""import os, sys
from pathlib import Path

REPO_URL = "https://github.com/Prashanf/floorplan-scanner.git"
BRANCH = "dev"
REPO_NAME = "floorplan-scanner"

# If we are not already in the repository root (run.py is missing):
if not Path("run.py").is_file():
    if not Path(REPO_NAME).is_dir():
        print(f"Cloning {REPO_URL} (branch: {BRANCH})...")
        ret = os.system(f"git clone -b {BRANCH} {REPO_URL}")
        if ret != 0:
            print(f"Branch '{BRANCH}' not found yet; falling back to default branch...")
            os.system(f"git clone {REPO_URL}")
    if Path(REPO_NAME).is_dir():
        %cd {REPO_NAME}
        print("Working directory updated to:", os.getcwd())

# Ensure repository root is on sys.path
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Prepare capture and output directories
!mkdir -p benchmark/captures/video/video2 benchmark/results/video/video2

print("Current Directory:", os.getcwd())
print("Repository Ready (run.py found):", Path("run.py").is_file())
"""))

cells.append(md("""## Step 1: GPU Acceleration Check
Verify that a CUDA GPU is present and configured in this environment (e.g. Kaggle T4/P100, Colab, or local workstation).
"""))

cells.append(code("""!nvidia-smi
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
"""))

cells.append(md("""## Step 2: Install Dependencies (with GPU & MobileSAM support)
Installs:
1. Base geometric libraries (`open3d`, `shapely`, `pydantic`, `scikit-learn`, `matplotlib`, `pandas`, etc.)
2. `pycolmap-cuda12` (GPU-accelerated SIFT extraction & matching for COLMAP)
3. `transformers` (NanoOWL / OWL-ViT open-vocabulary detector)
4. `MobileSAM` (Lightweight prompt-guided segmenter, ~40MB weights)
"""))

cells.append(code("""# 1. Base packages
!pip install -q open3d shapely pydantic pillow-heif rawpy trimesh click pyyaml tqdm scikit-learn matplotlib pandas

# 2. PyColmap with CUDA (GPU acceleration)
!pip install -q pycolmap-cuda12 || pip install -q pycolmap

# 3. Model dependencies for MobileSAM + NanoOWL
!pip install -q transformers timm
!pip install -q git+https://github.com/ChaoningZhang/MobileSAM.git

# 4. Download lightweight MobileSAM weights (~40MB)
!mkdir -p weights
!wget -q -O weights/mobile_sam.pt https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt || true
"""))

cells.append(md("""## Step 3: Configure GPU Environment Variables
Configure COLMAP to automatically use the CUDA GPU device.
"""))

cells.append(code("""import os, sys
import torch

os.environ["FLOORPLAN_COLMAP_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["FLOORPLAN_COLMAP_THREADS"] = "4"
print(f"COLMAP Device: {os.environ['FLOORPLAN_COLMAP_DEVICE']} (threads: {os.environ['FLOORPLAN_COLMAP_THREADS']})")
"""))

cells.append(md("""## Step 4: Run Fast Unit Tests
Ensure all geometric, stitching, damage, and pipeline unit tests pass.
"""))

cells.append(code("""!pytest -q
"""))

cells.append(md("""## Step 5: Run Video Walkthrough on Custom / Uploaded Video (`video2`)

Upload your video to `benchmark/captures/video/video2/` (e.g. `floorscanner_test2.mp4` or your own walkthrough file).
Run the pipeline using the terminal commands below:
"""))

cells.append(code("""# Check uploaded video in the capture directory
!ls -la benchmark/captures/video/video2/
"""))

cells.append(code("""# Terminal command execution:
!python run.py benchmark/captures/video/video2 --tier video --output-dir benchmark/results/video/video2/floor_plan.png
"""))

cells.append(code("""# Run with MobileSAM + NanoOWL model-based damage detection on GPU:
!python run.py benchmark/captures/video/video2 --tier video --output-dir benchmark/results/video/video2 --damage-detector mobilesam --verbose
"""))

cells.append(code("""# Comparative run with classical heuristic damage detector:
!python run.py benchmark/captures/video/video2 --tier video --output-dir benchmark/results/video/video2_heuristic --damage-detector heuristic --verbose
"""))

cells.append(md("""## Step 6: Visualize Floor Plan & Damage Inspection Results
Displays the generated dimensioned architectural drawing and structured inspection data inline.
"""))

cells.append(code("""import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from IPython.display import display, Markdown

# Find floor_plan.png
candidates = [
    Path("benchmark/results/video/video2/floor_plan.png"),
    Path("benchmark/results/video/video2/floor_plan.png/floor_plan.png"),
    Path("benchmark/results/video/video2_heuristic/floor_plan.png"),
    Path("output/floor_plan.png")
]
plan_path = next((p for p in candidates if p.is_file()), None)

if plan_path:
    print(f"Displaying: {plan_path}")
    plt.figure(figsize=(14, 10))
    plt.imshow(mpimg.imread(str(plan_path)))
    plt.axis("off")
    plt.title("Stitched Whole-Property Floor Plan", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
else:
    print("No floor plan image found yet. Run the pipeline step above first.")
"""))

cells.append(code("""# Display structured PropertyReport tables
report_candidates = [
    Path("benchmark/results/video/video2/report.json"),
    Path("benchmark/results/video/video2/floor_plan.png/report.json"),
    Path("benchmark/results/video/video2_heuristic/report.json"),
    Path("output/report.json")
]
report_path = next((p for p in report_candidates if p.is_file()), None)

if report_path:
    rep = json.loads(report_path.read_text())
    display(Markdown(f"### Capture Summary: `{rep['capture_id']}` (Tier: {rep['capture_tier']})"))
    
    summary_df = pd.DataFrame([{
        "Rooms": rep["room_count"],
        "Total Area (m²)": f"{rep['total_floor_area']['value']:.2f} [{rep['total_floor_area']['confidence_low']:.2f}, {rep['total_floor_area']['confidence_high']:.2f}]",
        "Adjacency Links": len(rep["adjacencies"]),
        "Damage Regions": len(rep["damage_regions"]),
        "Concealed Flags": len(rep["concealed_damage_flags"]),
        "Scope Line Items": len(rep["scope_line_items"]),
        "Processing Time (s)": rep["processing_time_seconds"]
    }])
    display(summary_df)

    if rep["rooms"]:
        room_rows = [{
            "Room ID": r["id"],
            "Area (m²)": f"{r['floor_area']['value']:.2f}",
            "Ceiling (m)": f"{r['ceiling_height']['value']:.2f}",
            "Walls": len(r["walls"]),
            "Doors": sum(1 for o in r["openings"] if o["type"] == "door"),
            "Windows": sum(1 for o in r["openings"] if o["type"] == "window"),
            "Rough Estimate": r.get("rough_estimate", False)
        } for r in rep["rooms"]]
        display(Markdown("#### Room Breakdown"))
        display(pd.DataFrame(room_rows))

    if rep["damage_regions"]:
        dmg_rows = [{
            "ID": d["id"],
            "Surface": d["surface_id"],
            "Class": d["damage_class"],
            "Width (m)": f"{d['extent_width']['value']:.2f}",
            "Height (m)": f"{d['extent_height']['value']:.2f}",
            "Area (m²)": f"{d['area']['value']:.3f}",
            "Confidence": f"{d['confidence']:.2f}"
        } for d in rep["damage_regions"]]
        display(Markdown("#### Detected Damage Regions"))
        display(pd.DataFrame(dmg_rows))

    if rep["scope_line_items"]:
        scope_rows = [{
            "ID": s["id"],
            "Surface": s["surface_id"],
            "Description": s["description"],
            "Priority": s["priority"],
            "Area (m²)": f"{s['estimated_area']['value']:.2f}"
        } for s in rep["scope_line_items"]]
        display(Markdown("#### Insurance Repair Scope"))
        display(pd.DataFrame(scope_rows))
"""))

cells.append(md("""## Step 7: Optional — Run LiDAR & Photo Tiers
Run on synthetic benchmark data or sample captures to test other modalities.
"""))

cells.append(code("""# Synthetic LiDAR tier run
!python tests/create_test_ply.py
!python run.py test_data/ --tier lidar --output-dir output/lidar_test
"""))

cells.append(code("""# Synthetic Photo tier run with MobileSAM detector
!python tests/create_test_photos.py
!python run.py test_photos/ --tier photo --output-dir output/photo_test --damage-detector mobilesam
"""))

cells.append(md("""## Step 8: Package Results for Download
Compress all outputs and floor plans into a single zip bundle.
"""))

cells.append(code("""!zip -r outputs_bundle.zip benchmark/results/ output/
print("Outputs packaged to outputs_bundle.zip. Ready for download!")
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def main():
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2))
    print(f"Successfully generated {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
