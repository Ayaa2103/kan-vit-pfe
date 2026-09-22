"""
Render real qualitative detection results (LiDAR point cloud + predicted
vs ground-truth 3D boxes) for the report/soutenance, using OpenCOOD's own
build_o3d_geometries()/save_o3d_visualization_cropped() (see
opencood/visualization/vis_utils.py) -- this script only bounds the frame
count (inference.py has no such flag), loops over multiple model_dirs so
the same frame indices can be compared across models, and (new) can pick
which frames to render by how much V2X-ViT-classic and KAN-ViT actually
disagree at a strict IoU threshold, instead of guessing indices blindly.
No change to inference.py or any model/dataset code.

Two phases, both available from one invocation:
1. --select: for a --compare-a/--compare-b pair of model_dirs, scan a
   candidate pool of validation frames, compute each frame's TP count at
   --select-iou (reusing eval_utils.caluclate_tp_fp -- the exact same
   greedy score-sorted IoU matching the real AP numbers are computed
   with, not a new metric), and keep the --top-n frames where
   compare-a's TP count exceeds compare-b's by the most (ties broken by
   more total ground-truth boxes, so a picked frame is also visually
   busy enough to be a good report figure, not a near-empty scene).
2. Render (crop + thicken box lines + title + legend, via
   annotate_vis_frame.py) each selected frame for every --model-dir, plus
   a side-by-side composite of --compare-a and --compare-b on that frame.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "OpenCOOD") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "OpenCOOD"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils

from scripts.annotate_vis_frame import annotate_frame, side_by_side

DISPLAY_NAMES = {
    "attfuse": "AttFuse",
    "v2xvit_classic": "V2X-ViT classique",
    "kanvit": "KAN-ViT",
}


def load_model_and_dataset(model_dir):
    config_path = os.path.join(model_dir, "config.yaml")
    hypes = yaml_utils.load_yaml(config_path, None)
    dataset = build_dataset(hypes, visualize=True, train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_utils.create_model(hypes)
    model.to(device)
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()
    return model, dataset, device


def run_frame(model, dataset, device, loader_iter, target_idx, cur_idx):
    """Advance loader_iter to target_idx, return (pred, score, gt, batch_data, new_cur_idx)."""
    batch_data = None
    while cur_idx <= target_idx:
        batch_data = next(loader_iter)
        cur_idx += 1
    batch_data = train_utils.to_device(batch_data, device)
    with torch.no_grad():
        pred_box_tensor, pred_score, gt_box_tensor = \
            inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
    return pred_box_tensor, pred_score, gt_box_tensor, batch_data, cur_idx


def select_frames(model_dir_a, model_dir_b, stride, iou_thresh, top_n, min_gt):
    print(f"\n{'=' * 20} SELECT FRAMES ({iou_thresh} IoU gap) {'=' * 20}", flush=True)
    model_a, dataset, device = load_model_and_dataset(model_dir_a)
    model_b, _, _ = load_model_and_dataset(model_dir_b)
    n = len(dataset)
    candidates = list(range(0, n, stride))
    print(f"Scanning {len(candidates)} candidate frames (stride={stride}) "
         f"out of {n}", flush=True)

    loader_a = DataLoader(dataset, batch_size=1, num_workers=0,
                          collate_fn=dataset.collate_batch_test,
                          shuffle=False, pin_memory=False, drop_last=False)
    loader_b = DataLoader(dataset, batch_size=1, num_workers=0,
                          collate_fn=dataset.collate_batch_test,
                          shuffle=False, pin_memory=False, drop_last=False)
    it_a, it_b = iter(loader_a), iter(loader_b)
    cur_a, cur_b = 0, 0

    results = []
    for idx in candidates:
        pred_a, score_a, gt_a, _, cur_a = run_frame(model_a, dataset, device, it_a, idx, cur_a)
        pred_b, score_b, gt_b, _, cur_b = run_frame(model_b, dataset, device, it_b, idx, cur_b)

        stat_a = {iou_thresh: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        stat_b = {iou_thresh: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        eval_utils.caluclate_tp_fp(pred_a, score_a, gt_a, stat_a, iou_thresh)
        eval_utils.caluclate_tp_fp(pred_b, score_b, gt_b, stat_b, iou_thresh)
        tp_a = sum(stat_a[iou_thresh]['tp'])
        tp_b = sum(stat_b[iou_thresh]['tp'])
        n_gt = stat_a[iou_thresh]['gt']

        gap = tp_a - tp_b
        results.append({'idx': idx, 'tp_a': tp_a, 'tp_b': tp_b, 'gap': gap, 'gt': n_gt})
        print(f"[select] frame {idx}: gt={n_gt} tp_a(compare-a)={tp_a} "
             f"tp_b(compare-b)={tp_b} gap={gap}", flush=True)

    del model_a, model_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eligible = [r for r in results if r['gt'] >= min_gt]
    eligible.sort(key=lambda r: (r['gap'], r['gt']), reverse=True)
    picked = eligible[:top_n]
    print(f"\nTop {top_n} frames by (compare-a - compare-b) TP@{iou_thresh} gap, "
         f"gt>={min_gt}:", flush=True)
    for r in picked:
        print(f"  frame {r['idx']}: gap={r['gap']} (tp_a={r['tp_a']}, "
             f"tp_b={r['tp_b']}, gt={r['gt']})", flush=True)
    return [r['idx'] for r in picked]


def render_model(model_dir, tag, frame_indices, out_dir, raw_dir):
    print(f"\n{'=' * 20} RENDER {tag} {'=' * 20}", flush=True)
    model, dataset, device = load_model_and_dataset(model_dir)
    print(f"[{tag}] validation dataset: {len(dataset)} frames", flush=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)

    display_name = DISPLAY_NAMES.get(tag, tag)
    annotated_paths = {}
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

            o3d_pcd, oabbs_pred, oabbs_gt = vis_utils.build_o3d_geometries(
                pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'])
            raw_path = os.path.join(raw_dir, f"{tag}_frame{i:04d}_raw.png")
            vis_utils.save_o3d_visualization_cropped(
                o3d_pcd, oabbs_pred, oabbs_gt, raw_path)

            n_pred = 0 if pred_box_tensor is None else len(pred_box_tensor)
            n_gt = 0 if gt_box_tensor is None else len(gt_box_tensor)
            title = f"{display_name} -- frame {i:04d} ({n_pred} predictions, {n_gt} ground truth)"
            out_path = os.path.join(out_dir, f"{tag}_frame{i:04d}.png")
            annotate_frame(raw_path, out_path, title)
            annotated_paths[i] = out_path
            print(f"[{tag}] frame {i}: saved {out_path} "
                 f"({n_pred} pred boxes, {n_gt} gt boxes)", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return annotated_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", action="append", required=True,
                        help="model_dir (config.yaml + net_epoch*.pth) to render. "
                             "Repeat for multiple models, paired positionally with --tag.")
    parser.add_argument("--tag", action="append", required=True,
                        help="Short label for each --model-dir, same order.")
    parser.add_argument("--frame-indices", type=str, default=None,
                        help="Comma-separated validation-set frame indices to "
                             "render. Ignored if --select is given.")
    parser.add_argument("--select", action="store_true",
                        help="Auto-select frames by TP@--select-iou gap "
                             "between --compare-a and --compare-b instead of "
                             "using --frame-indices.")
    parser.add_argument("--compare-a", type=str, default=None,
                        help="tag (from --tag) expected to score HIGHER at "
                             "--select-iou -- e.g. v2xvit_classic.")
    parser.add_argument("--compare-b", type=str, default=None,
                        help="tag expected to score LOWER -- e.g. kanvit.")
    parser.add_argument("--select-iou", type=float, default=0.7)
    parser.add_argument("--select-stride", type=int, default=15,
                        help="Scan every Nth validation frame as a candidate.")
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--min-gt", type=int, default=5,
                        help="Minimum ground-truth box count for a frame to "
                             "be eligible -- avoids picking a near-empty scene.")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/vis")
    args = parser.parse_args()

    assert len(args.model_dir) == len(args.tag), \
        "--model-dir and --tag must be given the same number of times"
    tag_to_dir = dict(zip(args.tag, args.model_dir))

    raw_dir = os.path.join(args.out_dir, "_raw")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | total memory: {props.total_memory / 1e9:.2f}GB",
             flush=True)

    if args.select:
        assert args.compare_a in tag_to_dir and args.compare_b in tag_to_dir, \
            "--compare-a/--compare-b must be tags passed via --tag"
        frame_indices = select_frames(
            tag_to_dir[args.compare_a], tag_to_dir[args.compare_b],
            args.select_stride, args.select_iou, args.top_n, args.min_gt)
    else:
        assert args.frame_indices, "--frame-indices required when not using --select"
        frame_indices = [int(x) for x in args.frame_indices.split(",")]

    per_model_paths = {}
    for tag, model_dir in tag_to_dir.items():
        per_model_paths[tag] = render_model(model_dir, tag, frame_indices,
                                            args.out_dir, raw_dir)

    if args.compare_a and args.compare_b and \
            args.compare_a in per_model_paths and args.compare_b in per_model_paths:
        print(f"\n{'=' * 20} SIDE-BY-SIDE {'=' * 20}", flush=True)
        for i in frame_indices:
            paths = []
            for tag in (args.compare_a, args.compare_b):
                p = per_model_paths[tag].get(i)
                if p:
                    paths.append(p)
            if len(paths) == 2:
                combo_path = os.path.join(
                    args.out_dir, f"compare_{args.compare_a}_vs_{args.compare_b}_frame{i:04d}.png")
                side_by_side(paths, combo_path)
                print(f"saved {combo_path}", flush=True)

    print("\nAll renders done.", flush=True)


if __name__ == "__main__":
    main()
