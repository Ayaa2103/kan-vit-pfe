"""
Kaggle kernel entry point for Day 5 REAL training of KAN-ViT on the full
34-sequence UrbanIng-V2X dataset (5 epochs). Pushed as-is via
`kaggle kernels push` (see kernel-metadata.json in this same folder).

Clones the repo at HEAD, installs the extra Python deps the intermediate-
fusion training code path needs beyond what's already on the Kaggle GPU
image (see the note in kaggle_speed_test_kernel.py for why
cumm/spconv/numba aren't among them), then hands off to
scripts/kaggle_train_entry.py --model kanvit, which locates the mounted
dataset, prepares/resumes the checkpoint directory, and invokes
opencood/tools/train.py.
"""
import os
import subprocess
import sys

REPO_URL = "https://github.com/Ayaa2103/kan-vit-pfe.git"
# NOT under /kaggle/working: that gets packaged as kernel "output", and
# the repo (plus its full .git pack) has no business being in there --
# only the checkpoints written by train.py should end up as output.
REPO_DIR = "/kaggle/tmp/kan-vit-pfe"


def run(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    os.makedirs("/kaggle/tmp", exist_ok=True)
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    else:
        print(f"{REPO_DIR} already present, skipping clone", flush=True)

    for pkg in ["open3d", "shapely>=2.0", "einops", "timm", "tensorboardX"]:
        run([sys.executable, "-m", "pip", "install", pkg])

    entry = os.path.join(REPO_DIR, "scripts", "kaggle_train_entry.py")
    run([sys.executable, entry, "--model", "kanvit"])


if __name__ == "__main__":
    main()
