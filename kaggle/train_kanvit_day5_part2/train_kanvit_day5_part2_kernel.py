"""
Kaggle kernel entry point for Day 5 KAN-ViT training, PART 2 -- resuming
on the aya1001 account after ayablh's quota ran out (30/30h) with the run
cancelled at epoch 2, iter 2508/5600. Pushed as-is via `kaggle kernels
push` (see kernel-metadata.json in this same folder).

Same code as kaggle/train_kanvit/train_kanvit_kernel.py -- nothing here
is account-specific. This kernel's dataset_sources attaches TWO aya1001
datasets: the full 34-seq training data, and a small resume-checkpoint
dataset (net_epoch1.pth, net_epoch2.pth, config.yaml, both checkpoints
verified NaN/Inf-free before upload). scripts/kaggle_train_entry.py's
find_resume_checkpoint_dir() picks the latter up automatically (it just
looks for *epoch*.pth under /kaggle/input, no dataset-slug hardcoding),
copies it into the working model_dir, and train.py resumes at epoch 3
(config.yaml already has epoches: 5, unchanged -- this is the back half
of the original 5-epoch target, not 5 more epochs).

Clones the repo at HEAD, installs the extra Python deps the intermediate-
fusion training code path needs beyond what's already on the Kaggle GPU
image (see the note in kaggle_speed_test_kernel.py for why
cumm/spconv/numba aren't among them), then hands off to
scripts/kaggle_train_entry.py --model kanvit.
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
