"""Generate notebooks/pipeline_end_to_end_gpu.ipynb with clean cells."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "pipeline_end_to_end_gpu.ipynb"

cells = []


def _cell_id() -> str:
    return f"cell{len(cells):03d}"

def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# ----------------- CELLS -----------------

cells.append(md("""# Floor Plan Scanner: end-to-end GPU run (CLI)

Runs `python run.py` on a walkthrough video with the GPU doing the heavy work:
- **COLMAP** feature extraction and matching on CUDA (`pycolmap-cuda12`).
- **Damage detection** with OWLv2 + MobileSAM on CUDA (`--damage-detector mobilesam`), or the OpenCV heuristic.

**Kaggle setup:** Settings -> Accelerator -> **GPU** (T4 or P100), Internet **On**. Add your video as a dataset (or set `VIDEO_PATH` below).
Mapping (incremental bundle adjustment) runs on the CPU by design: it is single-threaded so repeated runs give the same model.
"""))

cells.append(md("""## Step 0: Settings and repository
Edit the values in the first cell. `BRANCH` is the git branch that gets cloned."""))

cells.append(code("""# ---- settings -------------------------------------------------------------
REPO_URL = "https://github.com/Prashanf/floorplan-scanner.git"
BRANCH = "dev"                  # branch to clone / update
VIDEO_PATH = None               # e.g. "/kaggle/input/my-videos/floorscanner_test2.mp4"; None = search for a video
VIDEO_NAME_CONTAINS = ""        # when searching: only videos whose file name contains this text
CEILING_HEIGHT = None           # known ceiling height in metres (e.g. 3.15); None = automatic scale recovery
DAMAGE_DETECTOR = "mobilesam"   # "mobilesam" (GPU model) or "heuristic" (OpenCV rules)
RUN_HEURISTIC_COMPARISON = False  # also run the other detector (repeats the whole COLMAP step)
RUN_TESTS = True                # run the unit tests first (about 30 s)
RUN_OPTIONAL_TIERS = False      # also run the synthetic LiDAR and photo tiers
"""))

cells.append(code("""import os, subprocess, sys
from pathlib import Path

REPO_NAME = REPO_URL.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]

def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, text=True)

if not Path("run.py").is_file():  # not inside the repository yet
    if not Path(REPO_NAME).is_dir():
        print(f"Cloning {REPO_URL} (branch: {BRANCH})")
        if sh("git", "clone", "-b", BRANCH, REPO_URL, check=False).returncode != 0:
            raise SystemExit(f"Could not clone branch '{BRANCH}' of {REPO_URL}. Check BRANCH and that Internet is On.")
    else:
        print(f"Updating {REPO_NAME} to the latest '{BRANCH}'")
        sh("git", "-C", REPO_NAME, "fetch", "origin", BRANCH)
        sh("git", "-C", REPO_NAME, "checkout", BRANCH)
        sh("git", "-C", REPO_NAME, "pull", "--ff-only", "origin", BRANCH)
    os.chdir(REPO_NAME)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
print("Working directory:", os.getcwd())
print("Branch:", subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip(),
      "| commit:", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
"""))

cells.append(md("""## Step 1: GPU check
Stops here when no CUDA GPU is attached, so a CPU-only session does not burn hours on COLMAP."""))

cells.append(code("""!nvidia-smi
import torch
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No GPU found. Kaggle: Settings -> Accelerator -> GPU, then restart the session.")
print("GPU:", torch.cuda.get_device_name(0), "|", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
"""))

cells.append(md("""## Step 2: Install dependencies
- Base libraries from `requirements.txt` (without the CPU `pycolmap`).
- `pycolmap-cuda12`: the CUDA build of COLMAP's Python bindings. It replaces the CPU `pycolmap` (both provide the same module, so the CPU one is removed first).
- `transformers`, `timm`, `mobile_sam` and the MobileSAM weights for the model damage detector."""))

cells.append(code("""# 1. Base packages (requirements.txt minus pycolmap, which is installed below as the CUDA build)
!grep -vi "^pycolmap" requirements.txt > /tmp/requirements_base.txt
!pip install -q -r /tmp/requirements_base.txt rawpy pandas

# 2. COLMAP with CUDA. Remove the CPU build first: both provide `import pycolmap`.
!pip uninstall -y -q pycolmap
!pip install -q pycolmap-cuda12

# 3. Model damage detector: OWLv2 (transformers) + MobileSAM. No deps for mobile_sam: torch is already here.
!pip install -q transformers timm
!pip install -q --no-deps git+https://github.com/ChaoningZhang/MobileSAM.git
!python tools/download_mobile_sam.py
"""))

cells.append(md("""## Step 3: Configure COLMAP for the GPU and verify it
- `FLOORPLAN_COLMAP_BACKEND=pycolmap`: a `colmap` binary on PATH (usually a CPU build) would otherwise be used instead.
- `FLOORPLAN_COLMAP_DEVICE=cuda`: SIFT extraction and matching on the GPU.
- `FLOORPLAN_COLMAP_THREADS`: CPU threads for the parts that stay on the CPU (matching verification)."""))

cells.append(code("""import os, shutil, multiprocessing
import pycolmap

os.environ["FLOORPLAN_COLMAP_BACKEND"] = "pycolmap"
os.environ["FLOORPLAN_COLMAP_DEVICE"] = "cuda"
os.environ["FLOORPLAN_COLMAP_THREADS"] = str(multiprocessing.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("pycolmap", pycolmap.__version__, "| built with CUDA:", getattr(pycolmap, "has_cuda", False))
print("colmap binary on PATH (ignored):", shutil.which("colmap"))
print("FLOORPLAN_COLMAP_DEVICE =", os.environ["FLOORPLAN_COLMAP_DEVICE"], "| threads =", os.environ["FLOORPLAN_COLMAP_THREADS"])
if not getattr(pycolmap, "has_cuda", False):
    raise SystemExit("pycolmap has no CUDA support: the pycolmap-cuda12 install failed or the CPU build was imported. "
                     "Restart the session and rerun Step 2.")
"""))

cells.append(md("""## Step 4: Unit tests"""))

cells.append(code("""if RUN_TESTS:
    !python -m pytest -q
else:
    print("RUN_TESTS is False: skipped")
"""))

cells.append(md("""## Step 5: Pick the video
Uses `VIDEO_PATH`, or the first video found under `/kaggle/input` (add your video as a dataset), `/content` or `benchmark/captures/video`.
The CLI reads every video in a folder as one capture, so exactly one video is linked into `capture_video/`."""))

cells.append(code("""from pathlib import Path
import shutil

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}

if VIDEO_PATH:
    video = Path(VIDEO_PATH)
    if not video.is_file():
        raise SystemExit(f"VIDEO_PATH does not exist: {video}")
else:
    roots = [Path("/kaggle/input"), Path("/content"), Path("benchmark/captures/video")]
    found = sorted(f for r in roots if r.is_dir() for f in r.rglob("*")
                   if f.is_file() and f.suffix.lower() in VIDEO_EXTS and VIDEO_NAME_CONTAINS.lower() in f.name.lower())
    if not found:
        raise SystemExit("No video found. Add one as a Kaggle dataset, or set VIDEO_PATH.")
    print("Videos found:")
    for f in found:
        print(f"  {f}  ({f.stat().st_size / 1e6:.0f} MB)")
    video = found[0]

CAPTURE_DIR = Path("capture_video")
shutil.rmtree(CAPTURE_DIR, ignore_errors=True)
CAPTURE_DIR.mkdir()
(CAPTURE_DIR / video.name).symlink_to(video.resolve())
print("Using:", video, "->", CAPTURE_DIR)
"""))

cells.append(md("""## Step 6: Run the pipeline (CLI)
The run is logged to a file (`--verbose` includes COLMAP's device line); the cell after it prints the key lines and checks that the GPU was used."""))

cells.append(code("""ceiling_flag = f"--ceiling-height {CEILING_HEIGHT}" if CEILING_HEIGHT else ""
OUT_MAIN = f"output/video_{DAMAGE_DETECTOR}"
LOG_MAIN = f"{OUT_MAIN}.log"
os.makedirs("output", exist_ok=True)
print(f"python run.py {CAPTURE_DIR} --tier video --damage-detector {DAMAGE_DETECTOR} {ceiling_flag} --output-dir {OUT_MAIN}")
"""))

cells.append(code("""!python run.py {CAPTURE_DIR} --tier video --damage-detector {DAMAGE_DETECTOR} {ceiling_flag} --output-dir {OUT_MAIN} --verbose > {LOG_MAIN} 2>&1
print("exit code ok" if Path(OUT_MAIN, "report.json").is_file() else "NO report.json: see the log below")
"""))

cells.append(code("""import re

def show_log(path, tail=15):
    text = Path(path).read_text(errors="replace")
    keys = re.compile(r"COLMAP (device|attempt|succeeded)|reconstructed|Scale|scale|Loaded|loaded|model damage|Damage:|Timing|tier:|warning|Error|Traceback")
    lines = [l for l in text.splitlines() if keys.search(l) and "httpx" not in l]
    print("\\n".join(lines[:60]))
    print("...\\n" + "\\n".join(text.splitlines()[-tail:]))
    return text

log_text = show_log(LOG_MAIN)
print("\\nGPU check:")
print("  COLMAP on GPU :", "yes" if "COLMAP device: GPU" in log_text else "NO (see the log: device line missing or CPU)")
if DAMAGE_DETECTOR == "mobilesam":
    print("  damage model  :", "yes, device=cuda" if "device=cuda" in log_text else "NO (device line missing or CPU)")
"""))

cells.append(md("""### Optional: run the other damage detector for comparison
Repeats the whole run (including COLMAP), so it is off by default (`RUN_HEURISTIC_COMPARISON`)."""))

cells.append(code("""OTHER = "heuristic" if DAMAGE_DETECTOR == "mobilesam" else "mobilesam"
OUT_OTHER = f"output/video_{OTHER}"
if RUN_HEURISTIC_COMPARISON:
    !python run.py {CAPTURE_DIR} --tier video --damage-detector {OTHER} {ceiling_flag} --output-dir {OUT_OTHER} --verbose > {OUT_OTHER}.log 2>&1
    print("done:", OUT_OTHER)
else:
    print("RUN_HEURISTIC_COMPARISON is False: skipped")
"""))

cells.append(md("""## Step 7: Floor plan and report"""))

cells.append(code("""import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from IPython.display import display, Markdown

def show_run(out_dir, title):
    plan, rep_path = Path(out_dir, "floor_plan.png"), Path(out_dir, "report.json")
    if not rep_path.is_file():
        print(f"{title}: no report in {out_dir}")
        return None
    if plan.is_file():
        plt.figure(figsize=(14, 10))
        plt.imshow(mpimg.imread(str(plan)))
        plt.axis("off")
        plt.title(f"{title}: stitched floor plan", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
    rep = json.loads(rep_path.read_text())
    display(Markdown(f"### {title}: `{rep['capture_id']}` (tier {rep['capture_tier']})"))
    total = rep["total_floor_area"]
    display(pd.DataFrame([{
        "Rooms": rep["room_count"],
        "Total area (m2)": f"{total['value']:.2f} [{total['confidence_low']:.2f}, {total['confidence_high']:.2f}]",
        "Adjacency links": len(rep["adjacencies"]),
        "Damage regions": len(rep["damage_regions"]),
        "Concealed flags": len(rep["concealed_damage_flags"]),
        "Scope items": len(rep["scope_line_items"]),
        "Processing time (s)": round(rep["processing_time_seconds"], 1),
    }]))
    if rep["rooms"]:
        display(pd.DataFrame([{
            "Room": r["id"], "Area (m2)": f"{r['floor_area']['value']:.2f}", "Ceiling (m)": f"{r['ceiling_height']['value']:.2f}",
            "Walls": len(r["walls"]), "Doors": sum(o["type"] == "door" for o in r["openings"]),
            "Windows": sum(o["type"] == "window" for o in r["openings"]), "Rough estimate": r.get("rough_estimate", False),
        } for r in rep["rooms"]]))
    if rep["damage_regions"]:
        display(pd.DataFrame([{
            "ID": d["id"], "Surface": d["surface_id"], "Class": d["damage_class"],
            "Width (m)": f"{d['extent_width']['value']:.2f}", "Height (m)": f"{d['extent_height']['value']:.2f}",
            "Confidence": f"{d['confidence']:.2f}"} for d in rep["damage_regions"]]))
    for w in rep.get("warnings", []):
        print("warning:", w)
    return rep

main_report = show_run(OUT_MAIN, DAMAGE_DETECTOR)
if RUN_HEURISTIC_COMPARISON:
    other_report = show_run(OUT_OTHER, OTHER)
    if main_report and other_report:
        print(f"\\nDamage regions: {DAMAGE_DETECTOR} {len(main_report['damage_regions'])} vs {OTHER} {len(other_report['damage_regions'])}")
"""))

cells.append(md("""## Step 8: Optional: synthetic LiDAR and photo tiers"""))

cells.append(code("""if RUN_OPTIONAL_TIERS:
    !python tests/create_test_ply.py
    !python run.py test_data --tier lidar --output-dir output/lidar_test
    !python tests/create_test_photos.py
    !python run.py test_photos --tier photo --damage-detector {DAMAGE_DETECTOR} --output-dir output/photo_test
else:
    print("RUN_OPTIONAL_TIERS is False: skipped")
"""))

cells.append(md("""## Step 9: Package results
Zips `output/` into `/kaggle/working/outputs_bundle.zip` (or the current folder outside Kaggle)."""))

cells.append(code("""import shutil
dest = "/kaggle/working/outputs_bundle" if Path("/kaggle/working").is_dir() else "outputs_bundle"
print("Wrote", shutil.make_archive(dest, "zip", "output"))
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
