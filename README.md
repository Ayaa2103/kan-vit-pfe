# Lightweight KAN-ViT for Real-Time Cooperative Perception

A lightweight KAN-ViT architecture for real-time cooperative perception in V2X scenarios, built on the OpenCOOD framework and evaluated on the UrbanIng-V2X dataset.

## What this compares

Three intermediate-fusion, PointPillars-based cooperative-perception models, all trained on the same 34-sequence UrbanIng-V2X split with the same hyperparameters (see "Results" below for exactly what's held fixed):

| Model | Fusion transformer | Params | Config |
|---|---|---|---|
| **AttFuse** | none (single attention layer, no FFN) | 6.58M | [`point_pillar_intermediate_fusion_full.yaml`](OpenCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion_full.yaml) |
| **V2X-ViT (classic)** | V2X-ViT, standard 2-layer MLP feed-forward | 13.54M | [`point_pillar_intermediate_fusion_v2xvit_classic_full.yaml`](OpenCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion_v2xvit_classic_full.yaml) |
| **KAN-ViT** (this project's contribution) | V2X-ViT, feed-forward replaced by a Chebyshev-polynomial KAN layer | 14.72M | [`point_pillar_intermediate_fusion_kanvit_full.yaml`](OpenCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion_kanvit_full.yaml) |

KAN-ViT's only architectural difference from the V2X-ViT baseline is `KANFeedForward` — see [`chebykan_layer.py`](OpenCOOD/opencood/models/sub_modules/chebykan_layer.py) for the Kolmogorov-Arnold layer itself and [`v2xvit_kan.py`](OpenCOOD/opencood/models/fuse_modules/v2xvit_kan.py) for how it's substituted into the encoder without touching attention, STTF, or RTE. The full model assembly (PointPillars backbone → fusion → detection heads) is in [`point_pillar_transformer_kanvit.py`](OpenCOOD/opencood/models/point_pillar_transformer_kanvit.py); [`point_pillar_transformer.py`](OpenCOOD/opencood/models/point_pillar_transformer.py) is the same assembly with the standard V2X-ViT fusion instead (`model.core_method: point_pillar_transformer`, used by V2X-ViT-classic). AttFuse is a separate, simpler model (`model.core_method: point_pillar_intermediate`) with no transformer or feed-forward at all — a single attention-based fusion layer.

## Repository structure

```
OpenCOOD/               Vendored OpenCOOD framework (github.com/DerrickXuNu/OpenCOOD), extended with:
  opencood/models/sub_modules/chebykan_layer.py       ChebyKANLayer + KANFeedForward
  opencood/models/fuse_modules/v2xvit_kan.py          KAN-ViT fusion encoder/transformer
  opencood/models/point_pillar_transformer_kanvit.py  Full KAN-ViT model
  opencood/hypes_yaml/*_full.yaml                     Day-5 real-training configs (this project's)
  opencood/tools/train.py                             Training loop (see its module-level comments
                                                        for the DataLoader/NaN/checkpoint fixes found
                                                        during this project's training runs)
  opencood/tools/inference.py                         Stock OpenCOOD evaluation script (AP@0.3/0.5/0.7)
scripts/
  convert_full_dataset.py     Raw UrbanIng-V2X (.7z) -> OpenCOOD-format dataset
  kaggle_train_entry.py       Generic train/resume entry point used by every Kaggle training kernel
  kaggle_speed_test.py        Bounded GPU speed/VRAM diagnostic (not a training run)
  kaggle_num_workers_diag.py  DataLoader num_workers diagnostic, timed like the real training loop
  kaggle_nan_check.py         Bounded validation run for the ChebyKAN NaN-divergence fix
  smoketest_*.py              Local sanity checks before committing to a full Kaggle run
kaggle/                 One folder per Kaggle kernel actually pushed (kernel-metadata.json +
                         entry script) -- training, evaluation, and diagnostic kernels, across three
                         Kaggle accounts once each account's weekly 30h GPU quota ran out.
data/                   Not in this repo (gitignored) -- see "Dataset" below.
```

## Setup

```bash
pip install torch torchvision  # match your CUDA version
pip install open3d shapely einops timm tensorboardX
```

`OpenCOOD/requirements.txt` / `pip install -e OpenCOOD` pull in a lot more than the intermediate-fusion + PointPillars + V2X-ViT code path actually uses (`cumm`, `numba==0.49.0` with no wheel for recent Python, `scikit-image`, ...) and aren't needed — the four packages above, plus PyTorch, are everything actually imported by this path (verified by tracing the import graph; see the dependency note at the top of [`kaggle/kaggle_speed_test_kernel.py`](kaggle/kaggle_speed_test_kernel.py)). No `pip install -e .` step is required either: every entry point below adds `OpenCOOD/` to `PYTHONPATH` itself (or, when run from the repo root, `import opencood...` resolves once `OpenCOOD/` is on `sys.path`).

## Dataset

Trained on [UrbanIng-V2X](https://urban-in-v2x.github.io/) (34 sequences, LiDAR-only, infrastructure + vehicle CAVs). Raw data and converted output are both gitignored (`/data/`) — not part of this repository.

1. Get the raw UrbanIng-V2X `.7z` archives + labels (see the dataset's own site).
2. Convert to OpenCOOD's expected format:
   ```bash
   python scripts/convert_full_dataset.py
   ```
   See the module docstring in [`convert_full_dataset.py`](scripts/convert_full_dataset.py) for exactly what this does (per-sequence extract → convert → merge object classes under a flat `vehicles` key → clean up) and the resulting on-disk layout.

## Training

Locally:
```bash
python OpenCOOD/opencood/tools/train.py --hypes_yaml OpenCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion_kanvit_full.yaml --model_dir <checkpoint_dir>
```
`--model_dir` is also how a run resumes: if `<checkpoint_dir>/config.yaml` + `net_epoch*.pth` already exist, `--hypes_yaml` is ignored and training picks up from the latest epoch (weights + optimizer state) — see `load_saved_model` in [`train_utils.py`](OpenCOOD/opencood/tools/train_utils.py).

On Kaggle (what every real training run in this project actually used, given local GPU memory constraints): each `kaggle/train_*/` folder pairs a `kernel-metadata.json` with an entry script that clones this repo, installs the deps above, and calls [`scripts/kaggle_train_entry.py --model {attfuse,kanvit,v2xvit_classic}`](scripts/kaggle_train_entry.py), which locates the mounted dataset, prepares or resumes the checkpoint directory, and invokes `train.py`.

## Evaluation

Stock OpenCOOD `inference.py`, unmodified, computes AP@0.3/0.5/0.7 on the validation split:
```bash
python OpenCOOD/opencood/tools/inference.py --model_dir <checkpoint_dir> --fusion_method intermediate
```
`<checkpoint_dir>` must contain `config.yaml` + at least one `net_epoch*.pth` (the highest epoch is used automatically). The `kaggle/day5_eval*/` kernels run this same script against each trained model's checkpoint on GPU (a full validation pass, 1200 frames, is far too slow on CPU).

## Results

Validation split: 6 sequences, 1200 frames. All three models trained for 5 epochs, batch size 1, same learning rate/scheduler/score threshold/seed/`max_num`/`max_cav`. `cav_lidar_range`/`feature_stride` are shared between KAN-ViT and V2X-ViT-classic (same V2X-ViT host architecture) but not with AttFuse, which uses its own (different model family, no transformer). See the `_full.yaml` configs linked above for the exact values.

| Model | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---|---|
| AttFuse | 0.411 | 0.333 | 0.195 |
| V2X-ViT (classic) | 0.306 | 0.222 | 0.093 |
| KAN-ViT | 0.309 | 0.139 | 0.015 |

KAN-ViT and V2X-ViT-classic are close at AP@0.3 but diverge sharply at stricter IoU thresholds, where the KAN feed-forward's localization precision falls well behind both the standard MLP feed-forward (V2X-ViT-classic) and the no-FFN baseline (AttFuse) on this dataset.
