"""
Day 4 KAN-ViT smoke test: confirm the forward pass (through loss) runs
cleanly on the Day 2 mini-sample, then a quick param-count/latency
comparison against the Day 3 AttFuse baseline. Not a training run.
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, r"C:\Users\vPro\Projects\kan-vit-pfe\OpenCOOD")

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

KANVIT_HYPES = r"C:\Users\vPro\Projects\kan-vit-pfe\OpenCOOD\opencood\hypes_yaml\point_pillar_intermediate_fusion_kanvit_smoketest.yaml"
ATTFUSE_HYPES = r"C:\Users\vPro\Projects\kan-vit-pfe\OpenCOOD\opencood\hypes_yaml\point_pillar_intermediate_fusion_smoketest.yaml"
N_ITERS = 8


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def run_model(hypes_path, tag, timing_iters):
    print(f"\n{'='*20} {tag} {'='*20}", flush=True)
    hypes = yaml_utils.load_yaml(hypes_path, None)

    train_dataset = build_dataset(hypes, visualize=False, train=True)
    print(f"[{tag}] train dataset: {len(train_dataset)} frames", flush=True)

    loader = DataLoader(train_dataset,
                        batch_size=hypes['train_params']['batch_size'],
                        num_workers=0,
                        collate_fn=train_dataset.collate_batch_train,
                        shuffle=True, pin_memory=False, drop_last=True)

    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)

    total_params = count_params(model)
    print(f"[{tag}] total params: {total_params:,}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    it = iter(loader)
    times = []
    losses = []
    for i in range(timing_iters):
        try:
            batch_data = next(it)
        except StopIteration:
            it = iter(loader)
            batch_data = next(it)

        batch_data = train_utils.to_device(batch_data, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        output_dict = model(batch_data['ego'])
        final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
        final_loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        times.append(t1 - t0)
        losses.append(final_loss.item())
        print(f"[{tag}] iter {i+1}/{timing_iters} loss={final_loss.item():.4f} "
             f"time={t1-t0:.3f}s", flush=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    # skip the first iteration (warmup / cudnn autotune) for the latency avg
    steady_times = times[1:] if len(times) > 1 else times
    avg_time = sum(steady_times) / len(steady_times)

    print(f"[{tag}] SUMMARY: total_params={total_params:,} | "
         f"avg_fwd_bwd_time(steady)={avg_time:.3f}s | peak_gpu_mem={peak_mem:.2f}GB | "
         f"loss_first={losses[0]:.4f} loss_last={losses[-1]:.4f}", flush=True)

    return {'tag': tag, 'total_params': total_params, 'avg_time': avg_time,
           'peak_mem': peak_mem, 'model': model}


def main():
    kan_result = run_model(KANVIT_HYPES, "KAN-ViT", N_ITERS)
    print("\nKAN-ViT FORWARD PASS VALIDATED - no crash, no shape error", flush=True)

    att_result = run_model(ATTFUSE_HYPES, "AttFuse (Day3 baseline)", N_ITERS)

    # count params of just the replaced FFN layer(s) for KAN-ViT
    kan_ffn_params = 0
    for name, module in kan_result['model'].named_modules():
        if module.__class__.__name__ == 'KANFeedForward':
            kan_ffn_params += count_params(module)

    # for reference: what a plain MLP FeedForward with the same dims would cost
    from opencood.models.sub_modules.base_transformer import FeedForward
    plain_ffn = FeedForward(256, 256, dropout=0.3)
    plain_ffn_params = count_params(plain_ffn)

    print(f"\n{'='*20} COMPARISON {'='*20}", flush=True)
    print(f"AttFuse (Day3 baseline, no FFN at all):")
    print(f"  total params: {att_result['total_params']:,}")
    print(f"  avg fwd+bwd time: {att_result['avg_time']:.3f}s | peak GPU mem: {att_result['peak_mem']:.2f}GB")
    print(f"\nKAN-ViT (V2X-ViT backbone, FFN -> ChebyKAN):")
    print(f"  total params: {kan_result['total_params']:,}")
    print(f"  replaced FFN layer(s) params (all {3} KANFeedForward instances, depth=3): {kan_ffn_params:,}")
    print(f"  equivalent plain-MLP FeedForward params (per layer, for reference): {plain_ffn_params:,}")
    print(f"  avg fwd+bwd time: {kan_result['avg_time']:.3f}s | peak GPU mem: {kan_result['peak_mem']:.2f}GB")
    print(f"\nNote: AttFuse and KAN-ViT are different model families (point_pillar_intermediate "
         f"vs point_pillar_transformer) with very different backbones/heads -- the total param "
         f"and timing comparison is illustrative of overall model scale, not a like-for-like "
         f"ablation. The FFN-vs-FFN params comparison (KAN vs plain MLP) is the fair one.")


if __name__ == "__main__":
    main()
