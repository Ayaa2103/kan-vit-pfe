"""
Kaggle kernel entry point for the V2X-ViT "classic" speed/VRAM diagnostic
(scripts/kaggle_speed_test.py --model v2xvit_classic, ~20 iterations).
Ablation baseline for the KAN-ViT comparison: identical config to
point_pillar_intermediate_fusion_kanvit_full.yaml except the fusion
transformer's FFN is a plain MLP instead of KANFeedForward. Pushed as-is
via `kaggle kernels push` (see kernel-metadata.json in this same folder).
"""
import os
import subprocess
import sys

REPO_URL = "https://github.com/Ayaa2103/kan-vit-pfe.git"
REPO_DIR = "/kaggle/tmp/kan-vit-pfe"


def run(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def find_dataset_root(input_root="/kaggle/input", max_depth=3):
    all_dirs = [input_root]
    for depth in range(max_depth):
        next_level = []
        for d in all_dirs:
            try:
                next_level.extend(
                    os.path.join(d, e) for e in os.listdir(d)
                    if os.path.isdir(os.path.join(d, e)))
            except OSError:
                pass
        print(f"[input search depth {depth}] {next_level}", flush=True)
        for c in next_level:
            if os.path.isdir(os.path.join(c, "train")) and \
                    os.path.isdir(os.path.join(c, "validate")):
                return c
        all_dirs = next_level
    raise RuntimeError(
        f"Could not find a mounted dataset with train/ and validate/ "
        f"subfolders anywhere under {input_root}.")


def main():
    os.makedirs("/kaggle/tmp", exist_ok=True)
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    else:
        print(f"{REPO_DIR} already present, skipping clone", flush=True)

    for pkg in ["open3d", "shapely>=2.0", "einops", "timm"]:
        run([sys.executable, "-m", "pip", "install", pkg])

    dataset_root = find_dataset_root()
    print(f"Using dataset_root={dataset_root}", flush=True)

    script = os.path.join(REPO_DIR, "scripts", "kaggle_speed_test.py")
    run([sys.executable, script, "--model", "v2xvit_classic",
        "--dataset-root", dataset_root,
        "--n-iters", "20", "--n-warmup", "3", "--num-workers", "2"])


if __name__ == "__main__":
    main()
