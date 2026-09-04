"""
Convert the full UrbanIng-V2X dataset (34 sequences) to OpenCOOD format,
one sequence at a time: extract .7z -> convert via urbaning's official
do_one_sequence() -> merge object classes under 'vehicles' -> delete the
extracted copy (keep only the original .7z and the OpenCOOD output).

Never touches data/raw/. Writes to data/opencood_format_full/{train,validate}/,
one subfolder per sequence, and inside each: one subfolder per CAV
(infrastructure sensors included, named by their numeric ID -- OpenCOOD
makes no ego/infra distinction at this layer), containing a
timestamp.yaml + timestamp.pcd pair per frame. That per-CAV/per-timestamp
layout, and each yaml's flat 'vehicles' dict of {object_id: box params},
is exactly what OpenCOOD's own dataset loaders (see
opencood/data_utils/datasets/basedataset.py) expect -- this script's job
is entirely about producing that shape from UrbanIng-V2X's own on-disk
format, no OpenCOOD-side code needed to read it afterward.

Robust: a failure on one sequence is logged and the script moves on to the
next. Stops cleanly if free disk space drops below MIN_FREE_GB.
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import multivolumefile
import py7zr
import yaml

import urbaning.converters.utils as urbaning_utils
from urbaning.converters.opencood import do_one_sequence

# --- Disable camera image export (LiDAR-only pipeline: AttFuse/KAN-ViT never
# read camera images, see Day 2). do_one_sequence() has no flag for this, so
# we monkey-patch the two functions it actually uses for per-frame camera
# cost, without touching the installed urbaning package on disk:
#   - get_pinhole_undistort_function / get_fisheye_undistort_function build
#     the per-frame cv2.remap() closure -> replaced with an identity
#     passthrough, removing the real CPU cost (not just the disk write).
#   - cv2.imwrite is used in this module exclusively for camera PNGs (LiDAR
#     goes through o3d.io.write_point_cloud instead) -> no-op, so zero PNG
#     bytes ever hit disk.
def _identity_undistort_function(cameraParams_dict, alpha=0):
    return lambda frame: frame


urbaning_utils.get_pinhole_undistort_function = _identity_undistort_function
urbaning_utils.get_fisheye_undistort_function = _identity_undistort_function
cv2.imwrite = lambda *args, **kwargs: True

PROJECT_ROOT = Path(r"C:\Users\vPro\Projects\kan-vit-pfe")
RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "dataset"
RAW_LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "labels"
TMP_EXTRACT_ROOT = PROJECT_ROOT / "data" / "_tmp_extract_full"
OUT_ROOT = PROJECT_ROOT / "data" / "opencood_format_full"
PROGRESS_LOG = OUT_ROOT / "conversion_progress.log"
STATE_FILE = OUT_ROOT / "conversion_state.json"

MIN_FREE_GB = 20

TRAIN_SEQS = [
    "20241126_0004_crossing2_00", "20241126_0008_crossing1_01", "20241126_0010_crossing2_00",
    "20241126_0013_crossing2_00", "20241126_0014_crossing1_00", "20241126_0017_crossing1_00",
    "20241126_0018_crossing1_00", "20241126_0019_crossing2_00", "20241126_0022_crossing1_08",
    "20241126_0022_crossing1_09", "20241126_0024_crossing1_08", "20241126_0024_crossing1_18",
    "20241126_0024_crossing1_19", "20241126_0025_crossing1_00", "20241126_0025_crossing1_01",
    "20241127_0000_crossing1_00", "20241127_0003_crossing1_08", "20241127_0003_crossing1_09",
    "20241127_0010_crossing3_08", "20241127_0010_crossing3_09", "20241127_0011_crossing3_00",
    "20241127_0012_crossing3_00", "20241127_0014_crossing2_00", "20241127_0024_crossing3_08",
    "20241127_0024_crossing3_09", "20241127_0025_crossing2_00", "20241127_0026_crossing2_08",
    "20241127_0026_crossing2_09",
]
VALIDATE_SEQS = [
    "20241126_0001_crossing2_00", "20241126_0008_crossing1_00", "20241126_0024_crossing1_09",
    "20241127_0008_crossing1_00", "20241127_0009_crossing3_00", "20241127_0029_crossing2_00",
]
ALL_SEQS = [(s, "train") for s in TRAIN_SEQS] + [(s, "validate") for s in VALIDATE_SEQS]

# do_one_sequence()'s raw per-frame yaml has one dict per *object class*
# (e.g. 'car', 'pedestrian', ...), each mapping object_id -> box params,
# plus a handful of non-object scalar/pose keys at the same top level.
# NON_OBJECT_KEYS lets merge_vehicles_key() tell the two apart generically
# (skip these plus anything already named 'vehicles' or 'camera*', keep
# every other dict-valued key) instead of hardcoding UrbanIng-V2X's
# specific class names, which merge_vehicles_key merges into one flat
# 'vehicles' dict -- OpenCOOD's own datasets don't distinguish object
# classes, they expect every detectable object under this one key.
NON_OBJECT_KEYS = {'ego_speed', 'lidar_pose', 'predicted_ego_pos', 'true_ego_pos', 'vehicles'}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gb():
    total, used, free = shutil.disk_usage("C:/")
    return free / 1e9


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed": [], "failed": {}}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def merge_vehicles_key(seq_out_dir: Path):
    """Merge per-class object dicts under a flat 'vehicles' key, matching
    OpenCOOD's (OPV2V-derived) expectation, on every yaml just produced
    for this sequence."""
    n = 0
    for yaml_path in seq_out_dir.rglob("*.yaml"):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        merged = {}
        for k, v in data.items():
            if k in NON_OBJECT_KEYS or k.startswith("camera") or not isinstance(v, dict):
                continue
            merged.update(v)
        data["vehicles"] = merged
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        n += 1
    return n


def extract_sequence(seq_name: str, dest_root: Path):
    archive_base = RAW_DATASET_DIR / f"{seq_name}.7z"
    out_dir = dest_root / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with multivolumefile.open(archive_base, mode="rb") as target_archive:
        with py7zr.SevenZipFile(target_archive, mode="r") as archive:
            archive.extractall(path=out_dir)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train").mkdir(exist_ok=True)
    (OUT_ROOT / "validate").mkdir(exist_ok=True)
    TMP_EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    state = load_state()
    seqs_to_run = ALL_SEQS if only is None else [(s, sp) for s, sp in ALL_SEQS if s == only]
    if only is not None:
        log(f"=== Single-sequence test run: {only} ===")
    else:
        log(f"=== Starting full dataset conversion: {len(ALL_SEQS)} sequences ===")
    log(f"Already completed (resume): {state['completed']}")

    for seq_name, split in seqs_to_run:
        if seq_name in state["completed"]:
            log(f"SKIP (already done): {seq_name}")
            continue

        free = free_gb()
        if free < MIN_FREE_GB:
            log(f"ABORT: free disk space {free:.1f} GB < {MIN_FREE_GB} GB threshold. "
               f"Stopping cleanly before processing {seq_name}.")
            break

        log(f"--- {seq_name} ({split}) | free disk: {free:.1f} GB ---")
        seq_tmp_dir = TMP_EXTRACT_ROOT / seq_name
        try:
            t0 = time.time()
            log(f"[{seq_name}] extracting .7z ...")
            extract_sequence(seq_name, TMP_EXTRACT_ROOT)
            log(f"[{seq_name}] extracted in {time.time()-t0:.1f}s")

            t1 = time.time()
            log(f"[{seq_name}] converting via do_one_sequence() -> {split}/ ...")
            target_root = OUT_ROOT / split
            do_one_sequence([seq_name, str(TMP_EXTRACT_ROOT), str(RAW_LABELS_DIR), str(target_root)])
            log(f"[{seq_name}] converted in {time.time()-t1:.1f}s")

            seq_out_dir = target_root / seq_name
            n_yaml = merge_vehicles_key(seq_out_dir)
            log(f"[{seq_name}] merged 'vehicles' key in {n_yaml} yaml files")

            shutil.rmtree(seq_tmp_dir, ignore_errors=True)
            log(f"[{seq_name}] cleaned up extracted copy, kept only original .7z + OpenCOOD output")

            state["completed"].append(seq_name)
            state["failed"].pop(seq_name, None)
            save_state(state)
            log(f"[{seq_name}] DONE in {time.time()-t0:.1f}s total")

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            log(f"[{seq_name}] FAILED: {err}")
            log(f"[{seq_name}] traceback:\n{tb}")
            state["failed"][seq_name] = err
            save_state(state)
            shutil.rmtree(seq_tmp_dir, ignore_errors=True)
            log(f"[{seq_name}] cleaned up partial extraction after failure, moving on")
            continue

    log(f"=== Conversion loop finished. Completed: {len(state['completed'])}/34, "
       f"Failed: {len(state['failed'])} ===")
    if state["failed"]:
        log(f"Failed sequences: {state['failed']}")
    log(f"Final free disk space: {free_gb():.1f} GB")


if __name__ == "__main__":
    main()
