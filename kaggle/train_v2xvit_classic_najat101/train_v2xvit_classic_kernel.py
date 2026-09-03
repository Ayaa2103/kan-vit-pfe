"""
Kaggle kernel entry point for Day 5 REAL training of V2X-ViT "classic"
(plain-MLP FeedForward, same host architecture/hyperparameters as
KAN-ViT otherwise -- see point_pillar_intermediate_fusion_v2xvit_classic_full.yaml)
on the full 34-sequence UrbanIng-V2X dataset (5 epochs). Pushed as-is via
`kaggle kernels push` (see kernel-metadata.json in this same folder).

Identical to kaggle/train_v2xvit_classic/train_v2xvit_classic_kernel.py --
this copy exists only because it runs under a different Kaggle account
(najat101, both aya1001 and ayablh having exhausted their weekly 30h GPU
quota) via a separate kernel-metadata.json pointing at the same two
aya1001-owned datasets (main dataset + epoch-0 resume checkpoint) shared
to najat101 as a collaborator -- nothing in this script or in
scripts/kaggle_train_entry.py is account-specific.

Clone the repo at HEAD, install the extra deps, hand off to
scripts/kaggle_train_entry.py --model v2xvit_classic, which locates the
mounted dataset, prepares/resumes the checkpoint directory, and invokes
opencood/tools/train.py. It will resume at epoch index 1 (5600/5600 of
epoch 0 already done, net_epoch1.pth attached as the resume checkpoint
dataset) with NUM_WORKERS=3 (already the value in train.py at HEAD).
"""
import os
import subprocess
import sys

REPO_URL = "https://github.com/Ayaa2103/kan-vit-pfe.git"
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
    run([sys.executable, entry, "--model", "v2xvit_classic"])


if __name__ == "__main__":
    main()
