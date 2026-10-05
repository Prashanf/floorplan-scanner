"""Generate notebooks/pipeline_end_to_end_gpu.ipynb: fresh clone, two single-video runs with a COLMAP progress bar."""

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

cells.append(md(r"""# Floor Plan Scanner: two video runs on the GPU (CLI)

Runs `python run.py` once per video (landscape, then portrait), one run per cell, and saves each run's results separately.
COLMAP feature extraction and matching use the CUDA GPU (`pycolmap-cuda12`). While a run is going, a progress bar follows the COLMAP stages (feature extraction, matching, mapping).

**Kaggle setup:** Settings -> Accelerator -> **GPU**, Internet **On**, and add the two videos as a dataset.
Results go to `/kaggle/working/results/<landscape|portrait>/`: `report.json`, `floor_plan.png`, `run.log` (full log), `colmap_summary.json` and a `colmap/` folder (sparse model, diagnostic plot, sample keyframes).
Mapping (bundle adjustment) runs on the CPU by design: it is single-threaded so repeated runs give the same model."""))

cells.append(md(r"""## Step 0: Settings"""))

cells.append(code(r"""REPO_URL = "https://github.com/Prashanf/floorplan-scanner.git"
BRANCH = "dev"                    # branch that gets cloned (fresh clone every time the clone cell runs)

# Videos are found under /kaggle/input by a piece of the file name. Set a full path to override the search.
LANDSCAPE_MATCH = "ldscp"         # test-video3-ldscp.mp4
PORTRAIT_MATCH = "prtrt"          # test-video3-prtrt.mp4
VIDEO_LANDSCAPE = None            # e.g. "/kaggle/input/my-data/test-video3-ldscp.mp4"
VIDEO_PORTRAIT = None

CEILING_HEIGHT = None             # known ceiling height in metres (e.g. 3.15); None = automatic scale recovery
DAMAGE_MODEL = "opencv"           # "opencv" (default) or "mobilesam"
"""))

cells.append(md(r"""## Step 1: Clone the repository (always fresh)
Deletes any earlier clone and clones `BRANCH` again, so every run starts from the latest pushed code. Results live outside the clone and are not deleted."""))

cells.append(code(r"""import os, shutil, subprocess, sys
from pathlib import Path

REPO_NAME = REPO_URL.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
if Path("/kaggle/working").is_dir():
    WORK_ROOT = Path("/kaggle/working")
else:                                     # Colab or local: if we are inside an earlier clone, work next to it
    WORK_ROOT = Path.cwd().resolve()
    if (WORK_ROOT / "run.py").is_file() and WORK_ROOT.name == REPO_NAME:
        WORK_ROOT = WORK_ROOT.parent
REPO_DIR = WORK_ROOT / REPO_NAME

os.chdir(WORK_ROOT)                      # never stand inside the folder that is about to be deleted
if REPO_DIR.exists():
    print(f"Removing the old clone: {REPO_DIR}")
    shutil.rmtree(REPO_DIR)
print(f"Cloning {REPO_URL} (branch: {BRANCH})")
subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
print("Working directory:", os.getcwd())
print("Branch:", git("rev-parse", "--abbrev-ref", "HEAD"), "| commit:", git("rev-parse", "--short", "HEAD"), "|", git("log", "-1", "--pretty=%s"))
"""))

cells.append(md(r"""## Step 2: GPU check
Stops here when no CUDA GPU is attached."""))

cells.append(code(r"""!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No GPU found. Kaggle: Settings -> Accelerator -> GPU, then restart the session.")
print("GPU:", torch.cuda.get_device_name(0), "|", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
"""))

cells.append(md(r"""## Step 3: Install dependencies and configure COLMAP for the GPU
`pycolmap-cuda12` replaces the CPU `pycolmap` (both provide `import pycolmap`, so the CPU one is removed first)."""))

cells.append(code(r"""!grep -vi "^pycolmap" requirements.txt > /tmp/requirements_base.txt
!pip install -q -r /tmp/requirements_base.txt tqdm ipywidgets
!pip uninstall -y -q pycolmap
!pip install -q pycolmap-cuda12
# only needed with DAMAGE_MODEL = "mobilesam":
if DAMAGE_MODEL == "mobilesam":
    !pip install -q transformers timm
    !pip install -q --no-deps git+https://github.com/ChaoningZhang/MobileSAM.git
    !python tools/download_mobile_sam.py
"""))

cells.append(code(r"""import multiprocessing
import pycolmap

os.environ["FLOORPLAN_COLMAP_BACKEND"] = "pycolmap"       # a colmap binary on PATH (usually CPU only) would win otherwise
os.environ["FLOORPLAN_COLMAP_DEVICE"] = "cuda"            # SIFT extraction and matching on the GPU
os.environ["FLOORPLAN_COLMAP_THREADS"] = str(multiprocessing.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("pycolmap", pycolmap.__version__, "| built with CUDA:", getattr(pycolmap, "has_cuda", False))
print("FLOORPLAN_COLMAP_DEVICE =", os.environ["FLOORPLAN_COLMAP_DEVICE"], "| threads =", os.environ["FLOORPLAN_COLMAP_THREADS"])
if not getattr(pycolmap, "has_cuda", False):
    raise SystemExit("pycolmap has no CUDA support: the pycolmap-cuda12 install failed or the CPU build was imported. "
                     "Restart the session and rerun this step.")
"""))

cells.append(md(r"""## Step 4: Find the two videos"""))

cells.append(code(r"""import cv2

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
RESULTS_ROOT = WORK_ROOT / "results"
RESULTS_ROOT.mkdir(exist_ok=True)

def find_video(match, override):
    if override:
        p = Path(override)
        if not p.is_file():
            raise SystemExit(f"Video not found: {p}")
        return p
    roots = [Path("/kaggle/input"), Path("/content"), Path.home() / "Desktop"]
    hits = sorted(f for r in roots if r.is_dir() for f in r.rglob("*")
                  if f.is_file() and f.suffix.lower() in VIDEO_EXTS and match.lower() in f.name.lower())
    if not hits:
        raise SystemExit(f"No video with '{match}' in its name under /kaggle/input. Add it as a dataset or set the path above.")
    return hits[0]

def describe(path):
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"width": w, "height": h, "fps": round(fps, 2), "frames": frames,
            "duration_s": round(frames / fps, 1) if fps else None, "size_mb": round(path.stat().st_size / 1e6)}

VIDEO_LANDSCAPE = find_video(LANDSCAPE_MATCH, VIDEO_LANDSCAPE)
VIDEO_PORTRAIT = find_video(PORTRAIT_MATCH, VIDEO_PORTRAIT)
for label, path in (("landscape", VIDEO_LANDSCAPE), ("portrait", VIDEO_PORTRAIT)):
    print(f"{label:9s}", path, describe(path))
"""))

cells.append(md(r"""## Step 5: Run helper (progress bar, saving, COLMAP diagnostics)
Defines `run_video(label, video)`: runs the CLI on one video, shows a live progress bar, then saves everything for that video under `results/<label>/`.
Nothing runs here; the next two cells do the runs."""))

cells.append(code(r"""import json, re, threading, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm.auto import tqdm

RE_EXTRACT = re.compile(r"feature_extraction\.cc:\d+\]\s+Processed file \[(\d+)/(\d+)\]")
RE_MATCH = re.compile(r"pairing\.cc:\d+\]\s+Processing image \[(\d+)/(\d+)\]")
RE_REG = re.compile(r"Registering image #\d+ \(num_reg_frames=(\d+)\)")
RE_PREP = re.compile(r"Preprocessed (\d+) images: (\d+) rotated, (\d+) resized")
RE_ATTEMPT = re.compile(r"COLMAP attempt: (\w+)")
SHOW = re.compile(r"COLMAP (device|succeeded|config)|Scale (recovered|from)|scale recovery|Damage:|^tier:|^Error|Traceback|warning:", re.I)


class ProgressMonitor:
    # Reads the run's output line by line and drives one tqdm bar through the stages.
    def __init__(self, label):
        self.label, self.bar, self.stage, self.attempt = label, None, None, "strict"
        self.n_images, self.last_i, self.t0 = None, 0, time.time()
        self.stage_started, self.stage_seconds, self.attempts = time.time(), {}, []
        self._stop = threading.Event()
        self.set_stage("extracting keyframes from the video", None)
        threading.Thread(target=self._heartbeat, daemon=True).start()   # keeps the elapsed time moving between log lines

    def _heartbeat(self):
        while not self._stop.wait(1.0):
            try:
                self.bar.refresh()
            except Exception:
                pass

    def set_stage(self, name, total):
        now = time.time()
        if self.stage:
            self.stage_seconds[self.stage] = round(self.stage_seconds.get(self.stage, 0) + now - self.stage_started, 1)
        if self.bar is not None:
            self.bar.close()
        self.stage, self.stage_started, self.last_i = name, now, 0
        self.bar = tqdm(total=total, desc=f"[{self.label}] {name}", unit="img", mininterval=0.3, leave=True)

    def move(self, i, total, name):
        if self.stage != name or i < self.last_i:      # a new COLMAP attempt starts again at 1
            self.set_stage(name, total)
        self.last_i = i
        self.bar.total = total
        self.bar.n = i
        self.bar.refresh()

    def feed(self, line):
        m = RE_PREP.search(line)
        if m:
            self.n_images = int(m.group(1))
            tqdm.write(f"keyframes ready: {self.n_images} ({m.group(2)} rotated to landscape, {m.group(3)} resized)")
        m = RE_ATTEMPT.search(line)
        if m:
            self.attempt = m.group(1)
            self.attempts.append(self.attempt)
            tqdm.write(f"COLMAP configuration: {self.attempt}")
        m = RE_EXTRACT.search(line)
        if m:
            self.move(int(m.group(1)), int(m.group(2)), f"features ({self.attempt})")
            self.n_images = int(m.group(2))
        m = RE_MATCH.search(line)
        if m:
            self.move(int(m.group(1)), int(m.group(2)), f"matching ({self.attempt})")
        m = RE_REG.search(line)
        if m:
            registered = int(m.group(1))
            if not self.stage.startswith("mapping"):
                self.set_stage(f"mapping ({self.attempt})", self.n_images)
            self.bar.n = registered
            self.bar.set_postfix_str(f"{registered} registered")
            self.bar.refresh()
        if line.startswith(("Stitched", "Damage:")) and not self.stage.startswith("geometry"):
            self.set_stage("geometry, damage and plan", None)
        if SHOW.search(line) and "httpx" not in line:
            tqdm.write(line.rstrip()[:200])

    def close(self):
        self._stop.set()
        self.set_stage("done", None)
        self.bar.close()
        self.stage_seconds.pop("done", None)
        return self.stage_seconds


def _quat_to_rot(qw, qx, qy, qz):
    return np.array([[1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                     [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                     [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qy*qy + qx*qx)]])


def read_text_model(model_dir):
    # (camera centres by image name, 3D points, track lengths) from cameras/images/points3D.txt
    centres, names = {}, []
    lines = [l for l in (model_dir / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for l in lines[0::2]:
        p = l.split()
        rot = _quat_to_rot(*map(float, p[1:5]))
        centres[p[9]] = -rot.T @ np.array(list(map(float, p[5:8])))
    pts, tracks = [], []
    for l in (model_dir / "points3D.txt").read_text().splitlines():
        if l and not l.startswith("#"):
            p = l.split()
            pts.append(list(map(float, p[1:4])))
            tracks.append((len(p) - 8) // 2)
    return centres, np.array(pts).reshape(-1, 3), np.array(tracks)


def export_colmap(workspace, out_dir, n_input):
    # Largest sparse model -> text files, summary, diagnostic plot, a few keyframes.
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"workspace": str(workspace)}
    models = sorted((workspace / "sparse").glob("*/"))
    if not models:
        info["error"] = "no sparse model in the workspace"
        return info
    best, best_n = None, -1
    for m in models:
        try:
            n = pycolmap.Reconstruction(str(m)).num_reg_images()
        except Exception:
            n = len(list(m.glob("*")))
        if n > best_n:
            best, best_n = m, n
    rec = pycolmap.Reconstruction(str(best))
    rec.write_text(str(out_dir))
    try:
        (out_dir / "model_summary.txt").write_text(rec.summary())
        info["model_summary"] = rec.summary()
    except Exception as exc:
        info["model_summary_error"] = str(exc)
    info.update(models_built=len(models), registered_images=best_n, input_images=n_input)
    centres, pts, tracks = read_text_model(out_dir)
    info.update(points=int(len(pts)), mean_track_length=round(float(tracks.mean()), 2) if len(tracks) else None)
    try:
        info["mean_reprojection_error_px"] = round(float(rec.compute_mean_reprojection_error()), 3)
    except Exception:
        pass
    # images.txt holds every 2D keypoint of every image (tens of MB); keep only the pose line of each image
    pose_lines = [l for l in (out_dir / "images.txt").read_text().splitlines() if l.startswith("#")]
    body = [l for l in (out_dir / "images.txt").read_text().splitlines() if l and not l.startswith("#")][0::2]
    (out_dir / "images.txt").write_text("\n".join(pose_lines + body) + "\n")

    # diagnostic plot: points and camera path in the two widest directions, registration along the video, track lengths
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    if len(pts) > 3:
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        proj = lambda x: (np.asarray(x) - c) @ vt.T
        P, C = proj(pts), proj(np.array(list(centres.values()))) if centres else np.zeros((0, 3))
        for a, (i, j, title) in zip(ax[0], [(0, 1, "top view (two widest directions)"), (0, 2, "side view")]):
            a.scatter(P[:, i], P[:, j], s=1, c=P[:, 2 if j == 1 else 1], cmap="viridis", alpha=0.5)
            if len(C):
                a.plot(C[:, i], C[:, j], "-", c="red", lw=0.8)
                a.scatter(C[:, i], C[:, j], s=8, c="red")
                a.scatter(*C[0, [i, j]], s=60, c="lime", marker="^", label="first registered")
                a.legend(loc="best")
            a.set_aspect("equal", adjustable="datalim")
            a.set_title(f"{title}: {len(pts)} points, {len(C)} cameras")
    registered = np.zeros(n_input, bool)
    for name in centres:
        stem = Path(name).stem
        if stem.isdigit() and int(stem) < n_input:
            registered[int(stem)] = True
    ax[1][0].bar(range(n_input), registered.astype(int), width=1.0, color=np.where(registered, "tab:green", "tab:red"))
    ax[1][0].set_title(f"registered keyframes along the video: {registered.sum()} of {n_input} (green = registered)")
    ax[1][0].set_xlabel("keyframe index (time order)")
    ax[1][0].set_yticks([])
    if len(tracks):
        ax[1][1].hist(tracks, bins=range(2, int(min(tracks.max(), 60)) + 2), color="tab:blue")
    ax[1][1].set_title("track length per 3D point (images seeing it)")
    plt.tight_layout()
    fig.savefig(out_dir / "colmap_diagnostic.png", dpi=110)
    plt.close(fig)
    info["registered_keyframe_indices"] = [int(i) for i in np.flatnonzero(registered)]

    prepared = sorted((workspace / "prepared").glob("*.jpg")) or sorted((workspace / "images").glob("*.jpg"))
    (out_dir / "keyframes_sample").mkdir(exist_ok=True)
    for f in [prepared[i] for i in np.linspace(0, len(prepared) - 1, min(6, len(prepared))).astype(int)] if prepared else []:
        shutil.copy2(f, out_dir / "keyframes_sample" / f.name)
    return info


def run_video(label, video, extra_args=()):
    video = Path(video)
    out_dir = RESULTS_ROOT / label
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    capture = WORK_ROOT / "runs" / label / "capture"
    shutil.rmtree(capture.parent, ignore_errors=True)
    capture.mkdir(parents=True)
    (capture / video.name).symlink_to(video.resolve())          # exactly one video per run

    cmd = [sys.executable, "-u", "run.py", str(capture), "--tier", "video", "--damage-model", DAMAGE_MODEL,
           *(["--ceiling-height", str(CEILING_HEIGHT)] if CEILING_HEIGHT else []), *extra_args,
           "--output-dir", str(out_dir), "--verbose"]
    print("Command:", " ".join(cmd))
    print("Video:", video.name, describe(video))
    env = {**os.environ, "FLOORPLAN_KEEP_WORKSPACE": "1", "PYTHONUNBUFFERED": "1"}   # keep COLMAP's work folder to analyse it

    started, monitor = time.time(), ProgressMonitor(label)
    log_path = out_dir / "run.log"
    with open(log_path, "w") as log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                       text=True, bufsize=1, env=env) as proc:
        for line in proc.stdout:
            log.write(line)
            monitor.feed(line)
        code = proc.wait()
    stage_seconds = monitor.close()
    seconds = round(time.time() - started, 1)
    print(f"Finished in {seconds} s, exit code {code}")

    text = log_path.read_text(errors="replace")
    pick = lambda pattern, group=1: (re.findall(pattern, text) or [None])[-1]
    summary = {"label": label, "video": video.name, "video_info": describe(video), "command": " ".join(cmd),
               "exit_code": code, "seconds_total": seconds, "seconds_per_stage": stage_seconds,
               "colmap_device": pick(r"COLMAP device: (\w+)"), "colmap_attempts": monitor.attempts,
               "colmap_success_line": pick(r"(COLMAP succeeded with config: [^\n]*)"),
               "keyframes_line": pick(r"(Preprocessed \d+ images[^\n]*)"),
               "scale_line": pick(r"((?:Scale recovered from|Scale from)[^\n]*)"),
               "timing_line": pick(r"(Timing: [^\n]*)")}
    workspace = pick(r"COLMAP workspace kept in (\S+)")
    if workspace and Path(workspace).is_dir():
        try:
            summary["colmap"] = export_colmap(Path(workspace), out_dir / "colmap", monitor.n_images or 0)
        except Exception as exc:
            summary["colmap"] = {"error": f"{type(exc).__name__}: {exc}"}
        shutil.rmtree(workspace, ignore_errors=True)          # frames and the database are large; the useful parts are copied
    else:
        summary["colmap"] = {"error": "COLMAP workspace not found (the run failed before COLMAP, or too early)"}
    report_path = out_dir / "report.json"
    if report_path.is_file():
        rep = json.loads(report_path.read_text())
        summary["report"] = {"rooms": rep["room_count"], "total_floor_area_m2": round(rep["total_floor_area"]["value"], 2),
                             "damage_regions": len(rep["damage_regions"]), "warnings": rep.get("warnings", [])}
    (out_dir / "colmap_summary.json").write_text(json.dumps(summary, indent=2))

    # show the result
    if code != 0 or not report_path.is_file():
        print("\nNO report.json. Last log lines:\n" + "\n".join(text.splitlines()[-40:]))
    for name in ("floor_plan.png", "colmap/colmap_diagnostic.png"):
        if (out_dir / name).is_file():
            plt.figure(figsize=(12, 9))
            plt.imshow(mpimg.imread(str(out_dir / name)))
            plt.axis("off")
            plt.title(f"{label}: {name}")
            plt.show()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("command", "colmap")}, indent=2))
    print("COLMAP:", {k: v for k, v in summary["colmap"].items() if k not in ("model_summary", "registered_keyframe_indices")})
    print("\nSaved in", out_dir)
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out_dir)}  ({f.stat().st_size / 1e3:.0f} KB)")
    return summary
"""))

cells.append(md(r"""## Step 6: Run 1, landscape video
One command: `run.py` on the landscape video. Watch the progress bar for the COLMAP stages."""))

cells.append(code(r"""result_landscape = run_video("landscape", VIDEO_LANDSCAPE)
"""))

cells.append(md(r"""## Step 7: Run 2, portrait video
One command: `run.py` on the portrait video. Portrait keyframes are turned to landscape for COLMAP and mapped back afterwards by the pipeline."""))

cells.append(code(r"""result_portrait = run_video("portrait", VIDEO_PORTRAIT)
"""))

cells.append(md(r"""## Step 8: Compare and package
Side-by-side numbers for the two runs, then a zip of `results/` for download."""))

cells.append(code(r"""import pandas as pd
rows = []
for r in (result_landscape, result_portrait):
    c, rep = r.get("colmap", {}), r.get("report", {})
    rows.append({"video": r["label"], "exit": r["exit_code"], "seconds": r["seconds_total"],
                 "COLMAP device": r["colmap_device"], "attempts": ",".join(r["colmap_attempts"]),
                 "registered": f"{c.get('registered_images')}/{c.get('input_images')}", "points": c.get("points"),
                 "rooms": rep.get("rooms"), "area m2": rep.get("total_floor_area_m2"), "damage": rep.get("damage_regions")})
display(pd.DataFrame(rows))
print("Wrote", shutil.make_archive(str(WORK_ROOT / "video_results"), "zip", RESULTS_ROOT))
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
