"""
Quick smoke test for the full-scale converted dataset (34 sequences,
data/opencood_format_full/). Not a real training run: bounded to a few
hundred iterations, just enough to confirm the dataloader + training loop
+ backward pass + checkpoint saving hold up at this scale without
crashing or OOMing on the 4GB GPU. Reuses the same building blocks as
opencood/tools/train.py.
"""
import os
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, r"C:\Users\vPro\Projects\kan-vit-pfe\OpenCOOD")

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

HYPES_PATH = r"C:\Users\vPro\Projects\kan-vit-pfe\OpenCOOD\opencood\hypes_yaml\point_pillar_intermediate_fusion_smoketest_full.yaml"
N_TRAIN_STEPS = 300
N_VAL_STEPS = 100


def main():
    hypes = yaml_utils.load_yaml(HYPES_PATH, None)

    print('-----------------Dataset Building------------------', flush=True)
    t0 = time.time()
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    print(f"train dataset: {len(train_dataset)} frames | val dataset: {len(val_dataset)} frames "
         f"(built in {time.time()-t0:.1f}s)", flush=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=train_dataset.collate_batch_train,
                              shuffle=True, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=train_dataset.collate_batch_train,
                            shuffle=False, pin_memory=False, drop_last=True)

    print('---------------Creating Model------------------', flush=True)
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    saved_path = train_utils.setup_train(hypes)
    print(f"checkpoints will be saved to: {saved_path}", flush=True)

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)

    print(f"\nTraining {N_TRAIN_STEPS} steps (bounded smoke test, not a full epoch)...", flush=True)
    losses = []
    t_train0 = time.time()
    it = iter(train_loader)
    for i in range(N_TRAIN_STEPS):
        try:
            batch_data = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch_data = next(it)

        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        batch_data = train_utils.to_device(batch_data, device)
        output_dict = model(batch_data['ego'])
        final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
        final_loss.backward()
        optimizer.step()

        losses.append(final_loss.item())
        if i % 20 == 0 or i == N_TRAIN_STEPS - 1:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            mem_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            print(f"[{i+1}/{N_TRAIN_STEPS}] loss={final_loss.item():.4f} "
                 f"| GPU alloc={mem:.2f}GB reserved={mem_reserved:.2f}GB "
                 f"| elapsed={time.time()-t_train0:.0f}s", flush=True)

    t_train = time.time() - t_train0
    print(f"\nTrain phase done in {t_train:.1f}s ({t_train/N_TRAIN_STEPS:.2f}s/it)", flush=True)
    print(f"Loss first-10 avg: {statistics.mean(losses[:10]):.4f} | "
         f"last-10 avg: {statistics.mean(losses[-10:]):.4f}", flush=True)

    print(f"\nValidating {N_VAL_STEPS} steps (bounded)...", flush=True)
    val_losses = []
    t_val0 = time.time()
    with torch.no_grad():
        it_val = iter(val_loader)
        for i in range(N_VAL_STEPS):
            try:
                batch_data = next(it_val)
            except StopIteration:
                it_val = iter(val_loader)
                batch_data = next(it_val)
            model.eval()
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data['ego'])
            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            val_losses.append(final_loss.item())
    t_val = time.time() - t_val0
    print(f"Validation phase done in {t_val:.1f}s ({t_val/N_VAL_STEPS:.2f}s/it)", flush=True)
    print(f"Validation loss (mean over {N_VAL_STEPS} steps): {statistics.mean(val_losses):.4f}", flush=True)

    ckpt_path = os.path.join(saved_path, 'net_smoketest.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}", flush=True)

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"Peak GPU memory: allocated={peak_mem:.2f}GB reserved={peak_reserved:.2f}GB "
             f"(GPU total ~4GB)", flush=True)

    print("\nSMOKE TEST COMPLETE - no crash, no OOM", flush=True)


if __name__ == "__main__":
    main()
