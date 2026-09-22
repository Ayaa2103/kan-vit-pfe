"""
Kaggle kernel entry point for rendering real qualitative detection results
(LiDAR point cloud + predicted vs ground-truth 3D boxes) for a handful of
validation frames, across the three trained models. Wraps
scripts/kaggle_visualize_detections.py, which itself only calls OpenCOOD's
own visualize_result()/vis_utils pipeline -- no modified model/dataset code.

Open3D's legacy Visualizer needs a GL context even for off-screen
capture_screen_image(), which a bare Kaggle container doesn't provide --
installs and runs everything under Xvfb (a virtual X server) via xvfb-run.

Report-figure pass: instead of guessing frame indices, --select has
kaggle_visualize_detections.py scan a candidate pool of validation
frames and keep the ones where V2X-ViT-classic's TP count at strict
IoU=0.7 (the same eval_utils matching the real AP numbers use) beats
KAN-ViT's by the most -- i.e. frames where the AP gap at strict
thresholds is actually visible, not just a number in a table. Each
selected frame is rendered for all three models (cropped to the
box region, thickened outlines, title + legend burned in -- see
scripts/annotate_vis_frame.py) plus a V2X-ViT-classic-vs-KAN-ViT
side-by-side composite.
"""
import os
import re
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Ayaa2103/kan-vit-pfe.git"
REPO_DIR = "/kaggle/tmp/kan-vit-pfe"
WORK_DIR = "/kaggle/working"

# Xvfb/Open3D headless capture confirmed working (see git history: three
# failed test pushes -- black PNGs from a missing XDG_RUNTIME_DIR, then
# missing Mesa software GL, then capture_screen_image()'s do_render
# defaulting to False -- fixed in vis_utils.py) on a single model/frame,
# then the full 3-model x 4-frame set with the original, unannotated
# (uncropped, no legend) renderer.
TAGS = ["attfuse", "v2xvit_classic", "kanvit"]
COMPARE_A, COMPARE_B = "v2xvit_classic", "kanvit"
TOP_N = 4
SELECT_STRIDE = 15
MIN_GT = 5


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

    for pkg in ["open3d", "shapely>=2.0", "einops", "timm", "Pillow"]:
        run([sys.executable, "-m", "pip", "install", pkg])

    # Open3D's legacy Visualizer.create_window() needs a GL/X context even
    # for off-screen capture -- xvfb-run provides the virtual X display,
    # but Xvfb has no GLX/3D support of its own: without Mesa's software
    # rasterizer (llvmpipe) explicitly forced via LIBGL_ALWAYS_SOFTWARE,
    # Open3D's GL context "succeeds" but renders nothing, producing a
    # solid-black capture (confirmed on the first two test pushes: no
    # crash, correct box counts logged, black PNG both times -- the
    # XDG_RUNTIME_DIR fix alone was not sufficient).
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "xvfb", "libgl1-mesa-dri",
        "libglu1-mesa", "mesa-utils"])

    dataset_root = find_dir_containing("/kaggle/input", ["train", "validate"])
    print(f"Using dataset_root={dataset_root}", flush=True)

    ckpt_root = find_dir_containing("/kaggle/input", TAGS)
    print(f"Using ckpt_root={ckpt_root}", flush=True)

    model_dir_args = []
    for tag in TAGS:
        model_dir = os.path.join(WORK_DIR, f"vis_{tag}")
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        shutil.copytree(os.path.join(ckpt_root, tag), model_dir)
        rewrite_config_dataset_paths(os.path.join(model_dir, "config.yaml"),
                                     dataset_root)
        model_dir_args += ["--model-dir", model_dir, "--tag", tag]

    out_dir = os.path.join(WORK_DIR, "vis")
    os.makedirs(out_dir, exist_ok=True)

    script = os.path.join(REPO_DIR, "scripts", "kaggle_visualize_detections.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(REPO_DIR, "OpenCOOD") + \
        os.pathsep + env.get("PYTHONPATH", "")
    xdg_dir = "/tmp/xdg-runtime"
    os.makedirs(xdg_dir, mode=0o700, exist_ok=True)
    env["XDG_RUNTIME_DIR"] = xdg_dir
    # Force Mesa's software rasterizer (llvmpipe) -- see the apt-get
    # comment above for why this, not just Xvfb, is needed.
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["GALLIUM_DRIVER"] = "llvmpipe"

    # Diagnostic: confirm a real GL renderer is behind the virtual
    # display before spending time on the actual render. If this prints
    # "Mesa" / "llvmpipe" the GL context is real; if it errors or prints
    # nothing, the render below is expected to fail the same way again
    # and the log here is what the next fix should be based on.
    try:
        run(["xvfb-run", "-a", "--server-args=-screen 0 1920x1080x24",
            "glxinfo", "-B"], env=env)
    except Exception as e:
        print(f"glxinfo diagnostic failed (non-fatal): {e}", flush=True)

    cmd = ["xvfb-run", "-a",
          "--server-args=-screen 0 1920x1080x24",
          sys.executable, script,
          *model_dir_args,
          "--select",
          "--compare-a", COMPARE_A, "--compare-b", COMPARE_B,
          "--top-n", str(TOP_N),
          "--select-stride", str(SELECT_STRIDE),
          "--min-gt", str(MIN_GT),
          "--out-dir", out_dir]
    run(cmd, env=env)

    print("\nOutput files:", flush=True)
    for f in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, f)
        print(f"  {f}  ({os.path.getsize(full)} bytes)", flush=True)


if __name__ == "__main__":
    main()
