"""
Day 5 pre-flight diagnostic: measure REAL peak VRAM and real time/iteration
for AttFuse and KAN-ViT on a dedicated Kaggle GPU, on the full 34-sequence
UrbanIng-V2X dataset. NOT a training run -- ~20 forward+backward iterations
per model, just enough to get past cudnn autotune/allocator warmup and read
stable numbers. Used to decide epoch budget / batch size before any real
training is launched.

Meant to be run inside the Kaggle kernel produced by kaggle/kaggle_speed_test_kernel.py,
after that kernel has cloned this repo and put OpenCOOD on sys.path. Can also
be run locally (against the local dataset) by leaving --dataset-root unset.
"""
import argparse
import gc
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

ATTFUSE_HYPES = os.path.join(
    REPO_ROOT, "OpenCOOD", "opencood", "hypes_yaml",
    "point_pillar_intermediate_fusion_full.yaml")
KANVIT_HYPES = os.path.join(
    REPO_ROOT, "OpenCOOD", "opencood", "hypes_yaml",
    "point_pillar_intermediate_fusion_kanvit_full.yaml")

N_ITERS = 20
N_WARMUP = 3  # excluded from the steady-state timing average


def run_model(hypes_path, tag, dataset_root=None, n_iters=N_ITERS,
             n_warmup=N_WARMUP, num_workers=2):
    print(f"\n{'=' * 20} {tag} {'=' * 20}", flush=True)
    hypes = yaml_utils.load_yaml(hypes_path, None)

    if dataset_root:
        hypes['root_dir'] = os.path.join(dataset_root, 'train')
        hypes['validate_dir'] = os.path.join(dataset_root, 'validate')
    print(f"[{tag}] root_dir={hypes['root_dir']}", flush=True)
    print(f"[{tag}] validate_dir={hypes['validate_dir']}", flush=True)

    t0 = time.time()
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    print(f"[{tag}] train dataset: {len(train_dataset)} frames "
         f"(built in {time.time() - t0:.1f}s)", flush=True)

    loader = DataLoader(train_dataset,
                        batch_size=hypes['train_params']['batch_size'],
                        num_workers=num_workers,
                        collate_fn=train_dataset.collate_batch_train,
                        shuffle=True, pin_memory=False, drop_last=True)

    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] total params: {total_params:,}", flush=True)

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    it = iter(loader)
    times = []
    for i in range(n_iters):
        try:
            batch_data = next(it)
        except StopIteration:
            it = iter(loader)
            batch_data = next(it)

        batch_data = train_utils.to_device(batch_data, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_iter0 = time.time()

        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        output_dict = model(batch_data['ego'])
        final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
        final_loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_iter1 = time.time()

        times.append(t_iter1 - t_iter0)
        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[{tag}] iter {i + 1}/{n_iters} loss={final_loss.item():.4f} "
             f"time={t_iter1 - t_iter0:.3f}s alloc={mem:.2f}GB", flush=True)

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0

    steady_times = times[n_warmup:] if len(times) > n_warmup else times
    avg_time = sum(steady_times) / len(steady_times)

    print(f"[{tag}] SUMMARY: total_params={total_params:,} | "
         f"avg_time/iter(steady, skip first {n_warmup})={avg_time:.3f}s | "
         f"peak_alloc={peak_alloc:.2f}GB | peak_reserved={peak_reserved:.2f}GB",
         flush=True)

    result = {'tag': tag, 'total_params': total_params, 'avg_time': avg_time,
             'peak_alloc': peak_alloc, 'peak_reserved': peak_reserved}

    # free everything before the next model is loaded in the same process
    del model, optimizer, criterion, loader, train_dataset, it
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=None,
                        help="Directory containing train/ and validate/ "
                             "(e.g. /kaggle/input/kan-vit-pfe-urbaning-v2x-opencood). "
                             "If omitted, uses each config's own root_dir/validate_dir.")
    parser.add_argument("--n-iters", type=int, default=N_ITERS)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | total memory: {props.total_memory / 1e9:.2f}GB",
             flush=True)
    else:
        print("WARNING: no CUDA device visible -- this diagnostic is meant to "
             "run on a GPU kernel. Numbers below will be meaningless/CPU-only.",
             flush=True)

    results = []
    results.append(run_model(ATTFUSE_HYPES, "AttFuse", args.dataset_root,
                             args.n_iters, args.n_warmup, args.num_workers))
    results.append(run_model(KANVIT_HYPES, "KAN-ViT", args.dataset_root,
                             args.n_iters, args.n_warmup, args.num_workers))

    print(f"\n{'=' * 20} FINAL SUMMARY {'=' * 20}", flush=True)
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU total memory: {total_gb:.2f}GB", flush=True)
    for r in results:
        print(f"{r['tag']:>10s}: params={r['total_params']:,} | "
             f"avg_time/iter={r['avg_time']:.3f}s | "
             f"peak_alloc={r['peak_alloc']:.2f}GB | "
             f"peak_reserved={r['peak_reserved']:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
