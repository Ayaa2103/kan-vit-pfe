"""
Day 5 real-training entry point for a Kaggle kernel. Locates the mounted
training dataset, prepares (or resumes) a fixed model_dir, sets
PYTORCH_CUDA_ALLOC_CONF, and invokes opencood/tools/train.py as a
subprocess.

Key fact this script is built around (see yaml_utils.load_yaml): whenever
--model_dir is passed to train.py, it IGNORES --hypes_yaml and instead
reads model_dir/config.yaml. That's OpenCOOD's actual resume mechanism --
not something this script has to reimplement. So --model_dir is ALWAYS
passed here, even on a from-scratch run, which means config.yaml must
already exist in model_dir before train.py starts; this script writes it
itself, either fresh (root_dir/validate_dir patched to the Kaggle mount
path) or carried over from a resume-checkpoint dataset (see
find_resume_checkpoint_dir) -- in both cases root_dir/validate_dir are
(re)patched to the CURRENT session's mount path, so a resume never runs
against a stale path from a previous session.

To resume a timed-out run: download its /kaggle/working output
(`kaggle kernels output`), upload the checkpoint files (net_epoch*.pth +
config.yaml) as a Kaggle Dataset, attach it to this kernel's
dataset_sources, and push again -- find_resume_checkpoint_dir finds it
automatically, no code changes needed.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "attfuse": {
        "hypes": os.path.join(REPO_ROOT, "OpenCOOD", "opencood", "hypes_yaml",
                              "point_pillar_intermediate_fusion_full.yaml"),
        "model_dir_name": "ckpt_attfuse_day5",
    },
    "kanvit": {
        "hypes": os.path.join(REPO_ROOT, "OpenCOOD", "opencood", "hypes_yaml",
                              "point_pillar_intermediate_fusion_kanvit_full.yaml"),
        "model_dir_name": "ckpt_kanvit_day5",
    },
}


def _walk_dirs(input_root, max_depth):
    """Yield directories found by walking up to max_depth levels under input_root."""
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
        yield next_level
        all_dirs = next_level


def find_dataset_root(input_root="/kaggle/input", max_depth=3):
    for level in _walk_dirs(input_root, max_depth):
        for c in level:
            if os.path.isdir(os.path.join(c, "train")) and \
                    os.path.isdir(os.path.join(c, "validate")):
                return c
    raise RuntimeError(
        f"Could not find a mounted dataset with train/ and validate/ "
        f"subfolders anywhere under {input_root}.")


def find_resume_checkpoint_dir(input_root="/kaggle/input", max_depth=3):
    """
    Look for an attached dataset containing a previous run's checkpoints
    (net_epoch*.pth files) -- distinct from the main training-data
    dataset, which has train/validate subfolders but no .pth files.
    Returns None if nothing is found (i.e. this is a from-scratch run).
    """
    for level in _walk_dirs(input_root, max_depth):
        for c in level:
            if glob.glob(os.path.join(c, "*epoch*.pth")):
                return c
    return None


def prepare_model_dir(model_dir, hypes_path, dataset_root, resume_dir):
    os.makedirs(model_dir, exist_ok=True)
    config_dest = os.path.join(model_dir, "config.yaml")

    if resume_dir is not None:
        print(f"Resume checkpoint dataset found at {resume_dir}, "
             f"copying its contents into {model_dir}", flush=True)
        for f in os.listdir(resume_dir):
            shutil.copy2(os.path.join(resume_dir, f),
                        os.path.join(model_dir, f))
        if not os.path.exists(config_dest):
            raise RuntimeError(
                f"{resume_dir} has checkpoint files but no config.yaml -- "
                f"can't safely resume without the original config (model "
                f"architecture, hyperparameters, etc).")
        source_for_patch = config_dest
    else:
        print(f"No resume checkpoint dataset found -- starting fresh.",
             flush=True)
        source_for_patch = hypes_path

    # root_dir/validate_dir are (re)patched to THIS session's mount path
    # every time, fresh or resumed -- a resumed run must not keep a path
    # baked in from a previous Kaggle session in case the mount changes.
    with open(source_for_patch, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r'^root_dir:.*$',
                 f'root_dir: "{os.path.join(dataset_root, "train")}"',
                 text, count=1, flags=re.MULTILINE)
    text = re.sub(r'^validate_dir:.*$',
                 f'validate_dir: "{os.path.join(dataset_root, "validate")}"',
                 text, count=1, flags=re.MULTILINE)
    with open(config_dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {config_dest} (root_dir/validate_dir patched to "
         f"{dataset_root})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    args = parser.parse_args()
    cfg = MODELS[args.model]

    dataset_root = find_dataset_root()
    print(f"Using dataset_root={dataset_root}", flush=True)

    resume_dir = find_resume_checkpoint_dir()

    model_dir = os.path.join("/kaggle/working", cfg["model_dir_name"])
    prepare_model_dir(model_dir, cfg["hypes"], dataset_root, resume_dir)

    # KAN-ViT's Day 5 speed-test diagnostic showed a tight VRAM margin even
    # at batch_size=1 (13.96/15.64GB reserved on a T4). expandable_segments
    # reduces CUDA caching-allocator fragmentation, buying real headroom
    # without changing training semantics.
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "OpenCOOD") + \
        os.pathsep + env.get("PYTHONPATH", "")

    train_script = os.path.join(REPO_ROOT, "OpenCOOD", "opencood", "tools",
                                "train.py")
    cmd = [sys.executable, train_script,
          "--hypes_yaml", cfg["hypes"],  # required by argparse, but
                                          # ignored: --model_dir makes
                                          # train.py read model_dir/config.yaml
                                          # instead (see module docstring)
          "--model_dir", model_dir]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
