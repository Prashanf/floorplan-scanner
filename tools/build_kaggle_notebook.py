"""Build notebooks/floorplan_scanner_kaggle.ipynb: the whole pipeline in one self-contained notebook.

The repository (src/, tools/, tests/, run.py, ...) is embedded in the notebook as a base64 zip,
so the notebook needs nothing but the three sample zips as a Kaggle dataset.

    python tools/build_kaggle_notebook.py
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "floorplan_scanner_kaggle.ipynb"
INCLUDE = ["src", "tools", "tests", "schema", "run.py", "requirements.txt", "pytest.ini", "README.md",
           "capture_protocol.md", "device_matrix.md"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "test_photos", "test_video"}
SKIP_NAMES = {".DS_Store"}


def repo_zip_b64() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in INCLUDE:
            base = ROOT / name
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            for path in files:
                rel = path.relative_to(ROOT)
                if SKIP_PARTS & set(rel.parts) or path.name in SKIP_NAMES or path.suffix in {".pyc", ".ply", ".mp4"}:
                    continue
                z.write(path, rel.as_posix())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_ids = iter(range(1000))


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": f"cell-{next(_ids)}", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": f"cell-{next(_ids)}", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# floorplan-scanner on Kaggle

Runs the whole pipeline (LiDAR, photo and video tiers) on the three provided sample captures and ends with a
report cell. Nothing runs on your machine.

**Before you run**
1. Create a Kaggle dataset from the three zips (`single_room.zip`, `single_scan_floor_only.zip`,
   `single_scan_with_ceiling.zip`) and add it to this notebook (Add Data). Kaggle may unzip them
   automatically; both layouts work.
2. Notebook settings: **Internet On** (for `pip install`). Accelerator: choose a **GPU** (T4 or P100) to run
   COLMAP feature extraction and matching on it, or **None** for CPU only. Only COLMAP (photo and video tiers)
   can use a GPU; the LiDAR tier and the rest of the pipeline are CPU code and already fast.
3. *Run All*. LiDAR takes about a minute; the photo and video tiers take 10 to 40 minutes (COLMAP; less with a GPU).
   Turn the tiers on and off in the first cell.

**After it finishes**: the last cell prints a digest and writes `/kaggle/working/outputs.zip` (plans, JSON,
logs, no raw data). Download `outputs.zip`, or paste the digest text, so the results can be analysed.
"""

CONFIG = """
# ---- switches -------------------------------------------------------------------------------
RUN_TESTS = True     # fast unit tests of the pipeline (about 1 minute)
RUN_LIDAR = True     # raw depth logs -> point cloud -> plan (fast)
RUN_PHOTO = True     # stills cut from the video, one folder per room -> COLMAP -> plan (slow)
RUN_VIDEO = True     # rgb.mp4 -> keyframes -> COLMAP -> plan (slow)
USE_GPU = True       # use the GPU for COLMAP feature extraction and matching when the notebook has one
COLMAP_THREADS = None  # CPU threads for COLMAP features/matching; None = all cores (1 = bit-identical repeats)
CAPTURES = ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]
WORK = "/kaggle/working"
"""

INSTALL = """
import os, subprocess, sys

def sh(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr)

has_gpu = sh("nvidia-smi -L")[0] == 0
print("GPU:", sh("nvidia-smi -L")[1].strip() if has_gpu else "none (Settings > Accelerator > GPU to enable)")

base = "open3d shapely pydantic pillow-heif rawpy trimesh click pyyaml tqdm scikit-learn"
code, out = sh(f"{sys.executable} -m pip install -q {base}")
print(out[-1500:] if code else "base packages ok")

# pycolmap: the CUDA build runs SIFT extraction and matching on the GPU; fall back to the CPU build.
installed = None
if has_gpu and USE_GPU:
    code, out = sh(f"{sys.executable} -m pip install -q pycolmap-cuda12")
    if code == 0:
        check = sh(f"{sys.executable} -c 'import pycolmap; print(pycolmap.has_cuda)'")[1].strip().splitlines()
        if check and check[-1] == "True":
            installed = "pycolmap-cuda12 (GPU)"
        else:
            print("pycolmap-cuda12 installed but CUDA is not usable:", check[-1:] if check else out[-300:])
    else:
        print("pycolmap-cuda12 install failed, using the CPU build:", out[-300:])
if installed is None:
    sh(f"{sys.executable} -m pip uninstall -y -q pycolmap-cuda12")
    code, out = sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps pycolmap")
    installed = "pycolmap (CPU)"
    print(out[-800:] if code else "pycolmap CPU build ok")
print("COLMAP build:", installed)

os.environ["FLOORPLAN_COLMAP_DEVICE"] = "auto" if (has_gpu and USE_GPU) else "cpu"
os.environ["FLOORPLAN_COLMAP_THREADS"] = str(COLMAP_THREADS or os.cpu_count() or 1)
print("COLMAP device:", os.environ["FLOORPLAN_COLMAP_DEVICE"], "| threads:", os.environ["FLOORPLAN_COLMAP_THREADS"])

import importlib
for module in ["numpy", "scipy", "cv2", "open3d", "pycolmap", "shapely", "pydantic", "sklearn", "matplotlib", "PIL", "pillow_heif", "yaml", "click"]:
    try:
        m = importlib.import_module(module)
        print(f"{module:12s}", getattr(m, "__version__", "ok"))
    except Exception as exc:
        print(f"{module:12s} MISSING: {exc}")
print("CPU cores:", os.cpu_count())
import pycolmap
print("pycolmap has_cuda:", getattr(pycolmap, "has_cuda", False))
"""

UNPACK = """
import base64, io, os, zipfile
REPO = os.path.join(WORK, "floorplan-scanner")
os.makedirs(REPO, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(REPO_B64))) as z:
    z.extractall(REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)
print("repository unpacked to", REPO)
print(sorted(os.listdir(REPO)))
"""

DATA = """
import glob, pathlib, shutil, subprocess, time, json, re

KNOWN = CAPTURES
data_raw = pathlib.Path(WORK) / "data_raw"
data_raw.mkdir(exist_ok=True)

# 1. unzip any zips that Kaggle left as zips
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    name = pathlib.Path(z).stem
    target = data_raw / name
    if not target.exists():
        print("unzipping", z)
        shutil.unpack_archive(z, target)

# 2. find capture folders: any folder with odometry.csv + depth/
roots = [pathlib.Path("/kaggle/input"), data_raw]
streams = {}
for root in roots:
    for odo in root.rglob("odometry.csv"):
        folder = odo.parent
        if not (folder / "depth").is_dir():
            continue
        parts = [p for p in folder.parts if p in KNOWN]
        name = parts[-1] if parts else folder.parent.name
        streams.setdefault(name, folder)
print("captures found:", {k: str(v) for k, v in streams.items()})
missing = [c for c in CAPTURES if c not in streams]
if missing:
    print("WARNING: not found:", missing, "- check the dataset is attached")
CAPTURES = [c for c in CAPTURES if c in streams]

# 3. build sample_data_run/ the way the pipeline expects (real folders holding links to the raw files)
RUN = pathlib.Path(REPO) / "sample_data_run"
for name, folder in streams.items():
    lidar = RUN / "lidar" / name / "capture"
    video = RUN / "video" / name
    lidar.mkdir(parents=True, exist_ok=True)
    video.mkdir(parents=True, exist_ok=True)
    for f in ["depth", "confidence", "odometry.csv", "camera_matrix.csv", "imu.csv", "rgb.mp4"]:
        for dest in (lidar, video if f in ("rgb.mp4", "camera_matrix.csv") else None):
            if dest is None or not (folder / f).exists():
                continue
            link = dest / f
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(folder / f)
OUT = pathlib.Path(REPO) / "output"
OUT.mkdir(exist_ok=True)

def run_cmd(args, log_name, timeout=None):
    \"\"\"Run a command in the repo, save its output to output/<log_name>.txt, return (ok, seconds).\"\"\"
    start = time.time()
    try:
        proc = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        text, ok = proc.stdout + "\\n" + proc.stderr, proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        text, ok = f"TIMEOUT after {timeout}s\\n{exc.stdout or ''}", False
    (OUT / f"{log_name}.txt").write_text(text)
    return ok, time.time() - start

def run_tier(tier, name, extra=()):
    folder = {"photo": "photos"}.get(tier, tier)
    suffix = "_no_drift" if "--no-drift-correction" in extra else ""
    out = f"output/sample_{tier}_{name}{suffix}"
    ok, secs = run_cmd([sys.executable, "run.py", f"sample_data_run/{folder}/{name}", "--tier", tier,
                        "--output-dir", out, "--verbose", *extra], f"sample_{tier}_{name}{suffix}_log")
    status = "OK" if ok else "FAILED"
    print(f"{tier:6s} {name:28s}{suffix:10s} {status:7s} {secs:6.0f}s")
    return ok
"""

TESTS = """
if RUN_TESTS:
    ok, secs = run_cmd([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"], "pytest_log")
    print("tests passed" if ok else "TESTS FAILED", f"({secs:.0f}s)")
    print((OUT / "pytest_log.txt").read_text()[-1500:])
"""

LIDAR = """
if RUN_LIDAR:
    for name in CAPTURES:
        run_tier("lidar", name)
"""

PHOTO = """
if RUN_PHOTO:
    # Stills for the photo tier are cut from rgb.mp4: rooms come from the LiDAR run (see tools/make_photo_sets.py).
    for name in CAPTURES:
        ok, secs = run_cmd([sys.executable, "tools/make_photo_sets.py", f"sample_data_run/lidar/{name}/capture",
                            f"sample_data_run/photos/{name}"], f"photosets_{name}_log")
        counts = (OUT / f"photosets_{name}_log.txt").read_text().count(" photos")
        print(f"photo sets {name:28s} {'OK' if ok else 'FAILED'} {counts} room folders {secs:5.0f}s")
    for name in CAPTURES:
        run_tier("photo", name)
        run_tier("photo", name, extra=("--no-drift-correction",))   # drift ablation
"""

VIDEO = """
if RUN_VIDEO:
    for name in CAPTURES:
        run_tier("video", name)
"""

VALIDATE = """
ok, _ = run_cmd([sys.executable, "tools/validate_sample_output.py", "output"], "validate_log")
print((OUT / "validate_log.txt").read_text())
"""

REPORT = """
# ============================ REPORT ============================
import glob, json, pathlib, re, zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from IPython.display import display, Markdown

OUT = pathlib.Path(REPO) / "output"
digest = []                                   # plain-text lines, pasted back for analysis
def say(line=""):
    digest.append(line); print(line)

def log_facts(path):
    text = path.read_text() if path.exists() else ""
    return {
        "geometry_failed": len(re.findall(r"(?m)^warning: geometry failed", text)),
        "no_ceiling": len(re.findall(r"no ceiling scanned", text)),
        "colmap_retry": len(re.findall(r"failed, trying next", text)),
        "colmap_configs": ",".join(sorted(set(re.findall(r"COLMAP succeeded with config: (\\w+)", text)))),
        "keyframe_sets": len(re.findall(r"reconstructing from \\d+ keyframes", text)),
        "rotated_images": sum(int(m[1]) for m in re.findall(r"Preprocessed (\\d+) images: (\\d+) rotated, (\\d+) resized", text)),
        "resized_images": sum(int(m[2]) for m in re.findall(r"Preprocessed (\\d+) images: (\\d+) rotated, (\\d+) resized", text)),
        "skipped_rooms": len(re.findall(r"(?m)^warning: reconstruction failed for", text)),
        "fallback_rooms": len(re.findall(r"Fallback: single-image room estimate", text)) // 2 or len(re.findall(r"(?m)^warning: Fallback: single-image", text)),
        "overlaps": len(re.findall(r"overlap by", text)),
        "last_error": next((l.strip() for l in reversed(text.splitlines()) if l.startswith(("Error", "RuntimeError", "ValueError", "src.tiers"))), ""),
    }

rows, room_rows = [], []
for folder in sorted(p for p in OUT.glob("sample_*") if p.is_dir()):
    tier = folder.name.split("_")[1]
    capture = folder.name[len(f"sample_{tier}_"):]
    rep_path = folder / "report.json"
    facts = log_facts(OUT / f"{folder.name}_log.txt")
    if not rep_path.exists():
        rows.append({"tier": tier, "capture": capture, "status": "no report", **facts}); continue
    rep = json.loads(rep_path.read_text())
    rows.append({"tier": tier, "capture": capture,
                 "status": "ok" if rep["room_count"] else "ok, NO ROOMS (graceful failure)", "rooms": rep["room_count"],
                 "area_m2": round(rep["total_floor_area"]["value"], 1),
                 "area_interval": f'[{rep["total_floor_area"]["confidence_low"]:.1f}, {rep["total_floor_area"]["confidence_high"]:.1f}]',
                 "adjacencies": len(rep["adjacencies"]), "damage": len(rep["damage_regions"]),
                 "flags": len(rep["concealed_damage_flags"]), "scope": len(rep["scope_line_items"]),
                 "warnings": len(rep.get("warnings", [])),
                 "seconds": rep["processing_time_seconds"], **facts})
    for room in rep["rooms"]:
        room_rows.append({"tier": tier, "capture": capture, "room": room["id"],
                          "area_m2": round(room["floor_area"]["value"], 2), "walls": len(room["walls"]),
                          "ceiling_m": round(room["ceiling_height"]["value"], 2),
                          "ceiling_seen": room.get("ceiling_observed", True),
                          "rough_estimate": room.get("rough_estimate", False),
                          "doors": sum(o["type"] == "door" for o in room["openings"]),
                          "windows": sum(o["type"] == "window" for o in room["openings"])})
# failed runs leave only a log
for log in sorted(OUT.glob("sample_*_log.txt")):
    stem = log.name[:-len("_log.txt")]
    if not (OUT / stem).is_dir() or not (OUT / stem / "report.json").exists():
        tier = stem.split("_")[1]
        capture = stem[len(f"sample_{tier}_"):]
        if not any(r["tier"] == tier and r["capture"] == capture for r in rows):
            rows.append({"tier": tier, "capture": capture, "status": "FAILED", **log_facts(log)})

summary = pd.DataFrame(rows)
display(Markdown("## Run summary (every tier x capture)"))
display(summary)
say("RUN SUMMARY"); say(summary.to_string(index=False)); say()
rooms_df = pd.DataFrame(room_rows)
if len(rooms_df):
    display(Markdown("## Rooms"))
    display(rooms_df)
    say("ROOMS"); say(rooms_df.to_string(index=False)); say()

# tests and validation
for name in ("pytest_log", "validate_log"):
    p = OUT / f"{name}.txt"
    if p.exists():
        say(f"--- {name} (tail) ---"); say("\\n".join(p.read_text().strip().splitlines()[-12:])); say()

# warnings written into the reports
for folder in sorted(p for p in OUT.glob("sample_*") if p.is_dir() and (p / "report.json").exists()):
    rep = json.loads((folder / "report.json").read_text())
    for w in rep.get("warnings", [])[:6]:
        say(f"[{folder.name}] {w}")
if any((p / "report.json").exists() for p in OUT.glob("sample_*")):
    say()

# failure details
for log in sorted(OUT.glob("sample_*_log.txt")):
    stem = log.name[:-len("_log.txt")]
    if not (OUT / stem / "report.json").exists():
        lines = [l for l in log.read_text().splitlines() if not re.match(r"^[IWE]\\d{8}", l) and "DEBUG" not in l]
        say(f"--- {stem}: failed, last lines ---"); say("\\n".join(lines[-6:])); say()

# floor plans
plans = sorted(OUT.glob("sample_*/floor_plan.png"))
display(Markdown("## Floor plans"))
for i in range(0, len(plans), 2):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, plan in zip(axes, plans[i:i + 2]):
        ax.imshow(mpimg.imread(plan)); ax.set_title(plan.parent.name); ax.axis("off")
    for ax in axes[len(plans[i:i + 2]):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()

# digest + download bundle
(OUT / "digest.txt").write_text("\\n".join(digest))
bundle = pathlib.Path(WORK) / "outputs.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
    for p in pathlib.Path(REPO).glob("sample_data_run/photos/**/*.jpg"):
        z.write(p, "photos/" + p.relative_to(pathlib.Path(REPO) / "sample_data_run/photos").as_posix())
print(f"\\nwrote {bundle} ({bundle.stat().st_size / 1e6:.1f} MB): download it, or paste the text above.")
"""


def main() -> None:
    blob = repo_zip_b64()
    cells = [
        md(INTRO),
        code(CONFIG),
        code(INSTALL),
        md("### Repository (embedded, do not edit)"),
        code('REPO_B64 = (\n' + "\n".join(f'    "{blob[i:i + 4000]}"' for i in range(0, len(blob), 4000)) + "\n)"),
        code(UNPACK),
        md("### Find the sample data and prepare folders"),
        code(DATA),
        md("### Tests"),
        code(TESTS),
        md("### LiDAR tier (raw depth logs)"),
        code(LIDAR),
        md("### Photo tier (stills cut from the video, one folder per room)"),
        code(PHOTO),
        md("### Video tier"),
        code(VIDEO),
        md("### Validation"),
        code(VALIDATE),
        md("### Report"),
        code(REPORT),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                            "name": "python3"},
                                              "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB, {len(blob) / 1e3:.0f} KB of embedded repository)")


if __name__ == "__main__":
    main()
