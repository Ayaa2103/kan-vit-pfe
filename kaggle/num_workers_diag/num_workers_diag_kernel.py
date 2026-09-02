"""
Kaggle kernel entry point for the num_workers=3 vs num_workers=4 bounded
diagnostic (scripts/kaggle_num_workers_diag.py, 80 iterations per value,
V2X-ViT-classic config, full dataset). Isolated from and does not touch
the currently-running V2X-ViT-classic Day 5 training kernel -- separate
kernel, separate Kaggle session, read-only against the same dataset.
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

    for pkg in ["open3d", "shapely>=2.0", "einops", "timm", "psutil"]:
        run([sys.executable, "-m", "pip", "install", pkg])

    script = os.path.join(REPO_DIR, "scripts", "kaggle_num_workers_diag.py")
    run([sys.executable, script,
        "--n-iters", "80", "--n-warmup", "10", "--workers-list", "3,4"])


if __name__ == "__main__":
    main()
