"""
Kaggle kernel entry point for the Day 5 pre-flight speed/VRAM diagnostic.
Pushed as-is via `kaggle kernels push` (see kernel-metadata.json in this
same folder). Not a training run: clones the repo at HEAD, installs the
handful of extra Python deps the intermediate-fusion code path actually
needs (see notes below), then runs scripts/kaggle_speed_test.py for
~20 forward+backward iterations each on AttFuse and KAN-ViT.

Dependency note: OpenCOOD/requirements.txt pulls in a lot more than this
code path uses (cumm/spconv, numba==0.49.0, opencv, scikit-image,
sklearn...). None of those are imported by point_pillar_intermediate,
point_pillar_transformer_kanvit, or the IntermediateFusionDataset/
SpVoxelPreprocessor path used here (verified by grepping the actual
import graph), and the pinned numba version has no wheel for recent
Python -- so we deliberately do NOT `pip install -r requirements.txt` /
`pip install -e .`, and just install the 4 packages that ARE imported
and aren't already on the Kaggle GPU image: open3d, shapely, einops, timm.
"""
import glob
import os
import subprocess
import sys

REPO_URL = "https://github.com/Ayaa2103/kan-vit-pfe.git"
# NOT under /kaggle/working: everything under /kaggle/working gets packaged
# as kernel "output" after the run, and the repo (plus its full .git pack)
# has no business being in the output artifacts of a diagnostic run.
REPO_DIR = "/kaggle/tmp/kan-vit-pfe"
DATASET_SLUG_HINT = "kan-vit-pfe-urbaning-v2x-opencood"


def run(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    os.makedirs("/kaggle/tmp", exist_ok=True)
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    else:
        print(f"{REPO_DIR} already present, skipping clone", flush=True)

    # installed one at a time, without --quiet, so a build failure (e.g. a
    # pinned version with no prebuilt wheel for this image's Python falling
    # back to a source build) shows its actual error instead of being
    # swallowed. shapely is intentionally NOT pinned to the exact
    # requirements.txt version (2.0.0): any 2.x release is API-compatible
    # for the Polygon/IoU usage in this code path, and pinning to the older
    # 2.0.0 risks no prebuilt wheel for a newer Python -> source build ->
    # missing GEOS headers -> failure.
    for pkg in ["open3d", "shapely>=2.0", "einops", "timm"]:
        run([sys.executable, "-m", "pip", "install", pkg])

    # locate the mounted dataset -- don't hardcode the exact mount name,
    # find whichever /kaggle/input/* directory actually has train/ and
    # validate/ subfolders matching this project's dataset.
    candidates = sorted(glob.glob("/kaggle/input/*"))
    print(f"/kaggle/input contents: {candidates}", flush=True)
    dataset_root = None
    for c in candidates:
        if DATASET_SLUG_HINT in os.path.basename(c) and \
                os.path.isdir(os.path.join(c, "train")) and \
                os.path.isdir(os.path.join(c, "validate")):
            dataset_root = c
            break
    if dataset_root is None:
        for c in candidates:
            if os.path.isdir(os.path.join(c, "train")) and \
                    os.path.isdir(os.path.join(c, "validate")):
                dataset_root = c
                break
    if dataset_root is None:
        raise RuntimeError(
            f"Could not find a mounted dataset with train/ and validate/ "
            f"subfolders under /kaggle/input. Found: {candidates}")
    print(f"Using dataset_root={dataset_root}", flush=True)

    script = os.path.join(REPO_DIR, "scripts", "kaggle_speed_test.py")
    run([sys.executable, script, "--dataset-root", dataset_root,
        "--n-iters", "20", "--n-warmup", "3", "--num-workers", "2"])


if __name__ == "__main__":
    main()
