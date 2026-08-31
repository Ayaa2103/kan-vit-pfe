"""
Day 5 post-fix validation for KAN-ViT ONLY. Must be run and confirmed
clean BEFORE relaunching the full 5-epoch training kernel -- the previous
run diverged to NaN ~51 steps into epoch 0 and kept training on corrupted
weights for 5 full epochs (~10h) undetected. This script exercises the
exact same per-iteration logic train.py now uses (forward -> loss ->
check every loss component for NaN/Inf -> backward -> clip_grad_norm_ ->
step) for --n-iters iterations (default 200, comfortably past step 51),
and fails loudly the moment anything goes non-finite instead of
continuing silently.

Not a speed/VRAM diagnostic (see kaggle_speed_test.py for that) and not
a training run (no checkpoints saved) -- purely a "does the clamp fix in
chebykan_layer.py actually hold" check.
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "OpenCOOD") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "OpenCOOD"))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

# must match GRAD_CLIP_MAX_NORM in opencood/tools/train.py
GRAD_CLIP_MAX_NORM = 10

KANVIT_HYPES = os.path.join(
    REPO_ROOT, "OpenCOOD", "opencood", "hypes_yaml",
    "point_pillar_intermediate_fusion_kanvit_full.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=None,
                        help="Directory containing train/ and validate/. "
                             "If omitted, uses the config's own root_dir/validate_dir.")
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(KANVIT_HYPES, None)
    if args.dataset_root:
        hypes['root_dir'] = os.path.join(args.dataset_root, 'train')
        hypes['validate_dir'] = os.path.join(args.dataset_root, 'validate')
    print(f"root_dir={hypes['root_dir']}", flush=True)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | total memory: {props.total_memory / 1e9:.2f}GB",
             flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = build_dataset(hypes, visualize=False, train=True)
    print(f"train dataset: {len(train_dataset)} frames", flush=True)

    loader = DataLoader(train_dataset,
                        batch_size=hypes['train_params']['batch_size'],
                        num_workers=args.num_workers,
                        collate_fn=train_dataset.collate_batch_train,
                        shuffle=True, pin_memory=False, drop_last=True)

    model = train_utils.create_model(hypes)
    model.to(device)

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)

    it = iter(loader)
    t0 = time.time()
    for i in range(args.n_iters):
        try:
            batch_data = next(it)
        except StopIteration:
            it = iter(loader)
            batch_data = next(it)

        batch_data = train_utils.to_device(batch_data, device)

        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        output_dict = model(batch_data['ego'])
        final_loss = criterion(output_dict, batch_data['ego']['label_dict'])

        non_finite = {name: val for name, val in criterion.loss_dict.items()
                     if torch.is_tensor(val) and not torch.isfinite(val)}
        if non_finite:
            detail = ", ".join(f"{name}={val.item()}"
                              for name, val in criterion.loss_dict.items())
            print(f"\n*** NON-FINITE LOSS at iter {i + 1}/{args.n_iters}: "
                 f"{detail} ***", flush=True)
            print("VALIDATION FAILED -- the chebykan_layer.py clamp did "
                 "not prevent divergence. Do NOT launch the full run.",
                 flush=True)
            sys.exit(1)

        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                      max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"iter {i + 1}/{args.n_iters} loss={final_loss.item():.4f} "
                 f"reg={criterion.loss_dict['reg_loss'].item():.4f} "
                 f"conf={criterion.loss_dict['conf_loss'].item():.4f} "
                 f"elapsed={time.time() - t0:.0f}s", flush=True)

    total_time = time.time() - t0
    print(f"\nVALIDATION PASSED: {args.n_iters} iterations, no NaN/Inf "
         f"detected in total/reg/conf loss at any step. "
         f"Total time: {total_time:.0f}s ({total_time / args.n_iters:.2f}s/iter)",
         flush=True)


if __name__ == "__main__":
    main()
