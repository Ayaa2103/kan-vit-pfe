# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

# KAN-ViT's ChebyKANLayer (chebykan_layer.py) computes acos(tanh(x)) --
# tanh saturates to exactly +-1.0 in float32 once |x| gets large enough,
# and acos's gradient is -1/sqrt(1-x^2), which is infinite exactly at
# +-1. Without clipping, one such step can produce inf/nan gradients that
# poison every parameter on the next optimizer.step(). Applies to all
# models, not just KAN-ViT -- harmless for architectures that never
# approach this (e.g. AttFuse has no ChebyKAN layers at all).
GRAD_CLIP_MAX_NORM = 10

# Was 8, then 2, now 3. Day 5: AttFuse's 5-epoch Kaggle run took ~7.5h
# against a ~1.8h estimate -- a ~5x slowdown consistent with 8 dataloader
# workers oversubscribing a Kaggle GPU instance's few vCPUs, so this was
# dropped to 2. V2X-ViT-classic's real run then showed the same kind of
# gap again at num_workers=2 (~2.2-4.8s/it, in a clean alternating
# fast/slow pattern -- the round-robin signature of exactly 2 workers not
# keeping up), which a bounded 80-iteration diagnostic explained: the
# original ~0.9-1.1s/it speed-test measured GPU forward/backward only
# (its timer starts after next(it) returns), never the DataLoader/
# SpVoxelPreprocessor CPU cost the real per-iteration loop actually pays.
# That diagnostic, timed loop-inclusive like this file's tqdm actually
# is, confirmed num_workers=3 recovers ~0.94s/it with a regular
# (non-sawtooth) rate and all 4 of the T4 instance's vCPUs at ~93-95%
# with no oversubscription (3 workers + main process = 4 processes for
# 4 cores, an exact fit); num_workers=4 (5 processes for 4 cores) was
# marginally worse (higher tail latency). 3 is the confirmed optimum for
# this instance, not a guess.
NUM_WORKERS = 3

# Day 5 investigation: real training ran at ~5h10/epoch (~3.3s/it average)
# despite the speed diagnostic measuring ~0.9-1.1s/it, and the raw
# per-iteration log timings were erratic (1.2s one iter, 4-6.5s the next,
# occasional much worse spikes) rather than uniformly slow. GPU compute
# for a fixed-shape batch is deterministic -- that variance pattern is
# the signature of an I/O-bound dataloader (each sample needs open3d to
# read 1-3 agents' .pcd files, ~1.8-1.9MB each, off Kaggle's mounted
# dataset), not a GPU compute ceiling. Three safe, model-unrelated fixes
# for that, applied uniformly to every config since this is shared
# DataLoader code:
#   - persistent_workers: without it, PyTorch tears down and respawns
#     the whole worker pool (re-importing torch/open3d in each new
#     process) at the start of every single epoch.
#   - prefetch_factor: more per-worker lookahead buffer to absorb read
#     latency jitter instead of the training loop stalling on it.
#   - pin_memory: standard-issue faster host->GPU transfer, unrelated to
#     the stall investigation but free and correct to enable.
# These change nothing about what gets computed -- only how eagerly data
# is fetched -- so they don't touch batch_size/lr/epochs or any modeling
# parameter. Follow-up (see NUM_WORKERS above): these alone weren't
# enough -- the real ceiling on a 2-worker pool was CPU-side voxelization
# throughput, not read/transfer latency, so no amount of extra buffering
# fixed it. Raising NUM_WORKERS to match the instance's vCPU count did.
PIN_MEMORY = True
PREFETCH_FACTOR = 4

# Day 5 investigation continued: V2X-ViT-classic's real run showed
# s/it climbing steadily with WALL-CLOCK TIME elapsed (not with iteration
# count) even at NUM_WORKERS=3 -- 1.1-1.5s/it right after a resume,
# ~5.7s/it by the next epoch, on TWO different Kaggle accounts/instances.
# That time-correlated (not iteration-correlated) shape doesn't match a
# per-sample data-loading cost (which would depend on which frames get
# drawn, not on how long the process has been up), so a periodic
# mid-epoch checkpoint is cheap insurance regardless of the actual root
# cause (in-process leak vs Kaggle-instance-level degradation): it lets
# a stalled run be restarted well before a 12h wall-clock cap costs a
# whole epoch. 0 (or None) disables this and leaves behavior identical
# to before. Deliberately reuses the SAME filename the end-of-epoch save
# for this epoch will use (net_epoch{epoch+1}.pth) with 'epoch': epoch
# (not epoch+1) inside -- load_saved_model resumes from the checkpoint's
# stored 'epoch' field, not from the filename, so this doesn't require
# any change to the resume/checkpoint-discovery logic in train_utils.py
# or scripts/kaggle_train_entry.py: a restart after a mid-epoch save
# just redoes the current epoch from these weights (cheap: at most
# CHECKPOINT_EVERY_N_ITERS iterations of redundant compute) instead of
# restarting that epoch from scratch or, worse, from the previous
# epoch's checkpoint. Left at 0 (disabled, no behavior change) until
# explicitly turned on for a restart -- not enabled by this same commit,
# so a routine future `kaggle kernels push` doesn't silently start
# behaving differently.
CHECKPOINT_EVERY_N_ITERS = 0


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=NUM_WORKERS,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  pin_memory=PIN_MEMORY,
                                  persistent_workers=NUM_WORKERS > 0,
                                  prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=NUM_WORKERS,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False,
                                pin_memory=PIN_MEMORY,
                                persistent_workers=NUM_WORKERS > 0,
                                prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=NUM_WORKERS,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=PIN_MEMORY,
                                  persistent_workers=NUM_WORKERS > 0,
                                  prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=NUM_WORKERS,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=PIN_MEMORY,
                                persistent_workers=NUM_WORKERS > 0,
                                prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    # optimizer setup -- created before the resume-checkpoint load below,
    # since restoring optimizer state (Adam's exp_avg/exp_avg_sq) requires
    # an optimizer instance to load that state into.
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model,
                                                         optimizer,
                                                         device)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        train_losses = []

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])


            # Day 5 incident: KAN-ViT went NaN ~51 steps into epoch 0 and
            # kept training on corrupted weights for 5 full epochs (~10h)
            # with nothing catching it -- the loop has no NaN guard, and
            # NaN loss/gradients don't raise on their own. Check every
            # loss component (not just total_loss) right after it's
            # computed, and stop hard the moment any of them isn't finite.
            non_finite = {name: val for name, val in criterion.loss_dict.items()
                         if torch.is_tensor(val) and not torch.isfinite(val)}
            if non_finite:
                detail = ", ".join(f"{name}={val.item()}"
                                  for name, val in criterion.loss_dict.items())
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, iter {i + 1}/"
                    f"{len(train_loader)}: {detail}. Stopping immediately "
                    f"instead of continuing to train on corrupted weights.")

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)
            train_losses.append(final_loss.item())

            if not opt.half:
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                              max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                # gradients are still loss-scaled at this point; unscale
                # before clipping or the norm threshold is meaningless
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                              max_norm=GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

            if CHECKPOINT_EVERY_N_ITERS and (i + 1) % CHECKPOINT_EVERY_N_ITERS == 0:
                # 'epoch': epoch (not epoch+1) -- this epoch isn't done,
                # so a resume from this file must redo it, not skip to
                # the next one. See CHECKPOINT_EVERY_N_ITERS above.
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        train_ave_loss = statistics.mean(train_losses)
        print('At epoch %d, the training loss is %f' % (epoch, train_ave_loss))
        writer.add_scalar('Train_Loss_epoch', train_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            # full training state (weights + optimizer + epoch), not just
            # the bare weights, so a resumed run (same --model_dir) picks
            # the optimizer's momentum/variance back up instead of
            # restarting Adam from zero. See load_saved_model in
            # train_utils.py for the matching read side.
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
