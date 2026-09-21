"""
Render real qualitative detection results (LiDAR point cloud + predicted
vs ground-truth 3D boxes) for a handful of validation frames, using
OpenCOOD's own visualize_result()/vis_utils pipeline exactly as
inference.py would call it with --save_vis -- this script only bounds the
frame count (inference.py has no such flag) and loops over multiple
model_dirs so the same frame indices can be compared across models. No
change to inference.py or any model/dataset code.

Open3D's legacy Visualizer needs a display to create its GL context, even
for off-screen capture_screen_image() -- run this under Xvfb (see the
Kaggle kernel entry script in this same repo) or a real display.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "OpenCOOD") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "OpenCOOD"))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset


def render_model(model_dir, tag, frame_indices, out_dir):
    print(f"\n{'=' * 20} {tag} {'=' * 20}", flush=True)
    config_path = os.path.join(model_dir, "config.yaml")
    hypes = yaml_utils.load_yaml(config_path, None)

    dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"[{tag}] validation dataset: {len(dataset)} frames", flush=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_utils.create_model(hypes)
    model.to(device)
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()

    max_idx = max(frame_indices)
    wanted = set(frame_indices)
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if i > max_idx:
                break
            if i not in wanted:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            pred_box_tensor, pred_score, gt_box_tensor = \
                inference_utils.inference_intermediate_fusion(
                    batch_data, model, dataset)
            save_path = os.path.join(out_dir, f"{tag}_frame{i:04d}.png")
            dataset.visualize_result(pred_box_tensor, gt_box_tensor,
                                     batch_data['ego']['origin_lidar'],
                                     False, save_path, dataset=dataset)
            print(f"[{tag}] frame {i}: saved {save_path} "
                 f"({0 if pred_box_tensor is None else len(pred_box_tensor)} "
                 f"pred boxes, {0 if gt_box_tensor is None else len(gt_box_tensor)} gt boxes)",
                 flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", action="append", required=True,
                        help="model_dir (config.yaml + net_epoch*.pth) to render. "
                             "Repeat for multiple models, paired positionally with --tag.")
    parser.add_argument("--tag", action="append", required=True,
                        help="Short label for each --model-dir, same order.")
    parser.add_argument("--frame-indices", type=str, default="0",
                        help="Comma-separated validation-set frame indices to render.")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/vis")
    args = parser.parse_args()

    assert len(args.model_dir) == len(args.tag), \
        "--model-dir and --tag must be given the same number of times"
    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | total memory: {props.total_memory / 1e9:.2f}GB",
             flush=True)

    for model_dir, tag in zip(args.model_dir, args.tag):
        render_model(model_dir, tag, frame_indices, args.out_dir)

    print("\nAll renders done.", flush=True)


if __name__ == "__main__":
    main()
