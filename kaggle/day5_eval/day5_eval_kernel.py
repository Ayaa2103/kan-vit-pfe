"""
Kaggle kernel entry point for the Day-5 AttFuse vs KAN-ViT accuracy
evaluation (AP@0.3/0.5/0.7 on the validation split), using OpenCOOD's own
opencood/tools/inference.py unmodified -- this kernel only stages inputs
(dataset + checkpoint paths) for it. Read-only evaluation of already
trained net_epoch5.pth checkpoints: no training, no config/hyperparameter
changes.
"""
import os
import re
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Ayaa2103/kan-vit-pfe.git"
REPO_DIR = "/kaggle/tmp/kan-vit-pfe"
WORK_DIR = "/kaggle/working"


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def find_dir_containing(input_root, required_subdirs, max_depth=4):
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
            if all(os.path.isdir(os.path.join(c, s)) for s in required_subdirs):
                return c
        all_dirs = next_level
    raise RuntimeError(
        f"Could not find a directory with {required_subdirs} under {input_root}.")


def rewrite_config_dataset_paths(config_path, dataset_root):
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"^root_dir:.*$",
                 f"root_dir: {dataset_root}/train",
                 text, flags=re.MULTILINE)
    text = re.sub(r"^validate_dir:.*$",
                 f"validate_dir: {dataset_root}/validate",
                 text, flags=re.MULTILINE)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    os.makedirs("/kaggle/tmp", exist_ok=True)
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    else:
        print(f"{REPO_DIR} already present, skipping clone", flush=True)

    for pkg in ["open3d", "shapely>=2.0", "einops", "timm"]:
        run([sys.executable, "-m", "pip", "install", pkg])

    dataset_root = find_dir_containing("/kaggle/input", ["train", "validate"])
    print(f"Using dataset_root={dataset_root}", flush=True)

    ckpt_root = find_dir_containing("/kaggle/input", ["attfuse", "kanvit"])
    print(f"Using ckpt_root={ckpt_root}", flush=True)

    inference_script = os.path.join(REPO_DIR, "OpenCOOD", "opencood", "tools", "inference.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(REPO_DIR, "OpenCOOD") + \
        os.pathsep + env.get("PYTHONPATH", "")

    for tag in ["attfuse", "kanvit"]:
        print(f"\n{'=' * 20} EVAL {tag} {'=' * 20}", flush=True)
        model_dir = os.path.join(WORK_DIR, f"eval_{tag}")
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        shutil.copytree(os.path.join(ckpt_root, tag), model_dir)

        config_path = os.path.join(model_dir, "config.yaml")
        rewrite_config_dataset_paths(config_path, dataset_root)

        run([sys.executable, inference_script,
            "--model_dir", model_dir,
            "--fusion_method", "intermediate"], env=env)

        eval_yaml = os.path.join(model_dir, "eval.yaml")
        if os.path.isfile(eval_yaml):
            dest = os.path.join(WORK_DIR, f"eval_{tag}.yaml")
            shutil.copy(eval_yaml, dest)
            print(f"[{tag}] eval.yaml saved to {dest}", flush=True)


if __name__ == "__main__":
    main()
