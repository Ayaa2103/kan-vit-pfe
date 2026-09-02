"""
Bounded, isolated diagnostic to test whether num_workers > 2 closes the gap
between kaggle_speed_test.py's optimistic ~0.9-1.1s/it (GPU-only, timed
after next(it) returns -- excludes DataLoader/voxelization wait) and the
real V2X-ViT-classic Day 5 run's observed ~2.2-4.8s/it (full loop time,
DataLoader included, matching how train.py's tqdm bar times each
iteration).

Not a training run: no backward/optimizer.step() needed to time the
DataLoader, but included anyway (cheap, and keeps the per-iteration cost
identical to train.py's real loop) for n-iters steps per num_workers
value, on the FULL dataset with V2X-ViT-classic's own config (same
batch_size=1, same DataLoader kwargs as train.py: pin_memory, prefetch_factor,
persistent_workers). Times the WHOLE loop body (data fetch through
optimizer.step()), like train.py's tqdm -- not GPU-only like the old
speed-test script.

Also samples per-core CPU utilization throughout (psutil) to check for
oversubscription on the T4 instance's 4 vCPUs, the same failure mode that
made num_workers=8 catastrophic previously.

Runs in its own Kaggle session (concurrent with, and without touching,
the currently running V2X-ViT-classic training kernel).
"""
import argparse
import os
import statistics
import sys
import threading
import time

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "OpenCOOD") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "OpenCOOD"))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

HYPES_PATH = os.path.join(
    REPO_ROOT, "OpenCOOD", "opencood", "hypes_yaml",
    "point_pillar_intermediate_fusion_v2xvit_classic_full.yaml")

PIN_MEMORY = True
PREFETCH_FACTOR = 4


class CpuSampler:
    """Background thread sampling per-core CPU% every 0.5s via psutil."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []  # list of list[float] (per-core %) per tick
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        import psutil
        # first call with interval=None just primes the internal counters
        psutil.cpu_percent(percpu=True)
        while not self._stop.is_set():
            time.sleep(self.interval)
            self.samples.append(psutil.cpu_percent(percpu=True))

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return "no samples collected"
        n_cores = len(self.samples[0])
        per_core_avg = [statistics.mean(s[c] for s in self.samples)
                        for c in range(n_cores)]
        per_core_max = [max(s[c] for s in self.samples)
                        for c in range(n_cores)]
        overall_avg = statistics.mean(per_core_avg)
        return (f"{n_cores} logical cores | "
               f"avg%/core={[round(v, 1) for v in per_core_avg]} | "
               f"max%/core={[round(v, 1) for v in per_core_max]} | "
               f"overall avg={overall_avg:.1f}%")


def run_one(num_workers, dataset_root, n_iters, n_warmup, hypes):
    tag = f"num_workers={num_workers}"
    print(f"\n{'=' * 20} {tag} {'=' * 20}", flush=True)

    hypes = dict(hypes)  # shallow copy is enough, we only touch top-level keys
    if dataset_root:
        hypes['root_dir'] = os.path.join(dataset_root, 'train')
        hypes['validate_dir'] = os.path.join(dataset_root, 'validate')

    t0 = time.time()
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    print(f"[{tag}] train dataset: {len(train_dataset)} frames "
         f"(built in {time.time() - t0:.1f}s)", flush=True)

    loader = DataLoader(train_dataset,
                        batch_size=hypes['train_params']['batch_size'],
                        num_workers=num_workers,
                        collate_fn=train_dataset.collate_batch_train,
                        shuffle=True,
                        pin_memory=PIN_MEMORY,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
                        drop_last=True)

    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    sampler = CpuSampler(interval=0.5)
    sampler.start()

    times = []
    it = enumerate(loader)
    # This is the exact shape of train.py's loop: the timer wraps the
    # WHOLE body, including `next()` on the loader (data fetch +
    # voxelization), not just the GPU forward/backward like the old
    # kaggle_speed_test.py did.
    for step in range(n_iters):
        t_iter0 = time.time()

        i, batch_data = next(it)
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        batch_data = train_utils.to_device(batch_data, device)
        output_dict = model(batch_data['ego'])
        final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_iter1 = time.time()

        times.append(t_iter1 - t_iter0)
        print(f"[{tag}] iter {step + 1}/{n_iters} loss={final_loss.item():.4f} "
             f"time={t_iter1 - t_iter0:.3f}s", flush=True)

    sampler.stop()

    steady = times[n_warmup:] if len(times) > n_warmup else times
    mean_t = statistics.mean(steady)
    stdev_t = statistics.stdev(steady) if len(steady) > 1 else 0.0
    cov = stdev_t / mean_t if mean_t else 0.0

    print(f"[{tag}] SUMMARY: mean={mean_t:.3f}s/it | stdev={stdev_t:.3f}s | "
         f"coeff_of_variation={cov:.2f} (lower=more regular, "
         f"higher=sawtooth/irregular) | min={min(steady):.3f}s | "
         f"max={max(steady):.3f}s", flush=True)
    print(f"[{tag}] CPU usage: {sampler.summary()}", flush=True)

    del model, optimizer, criterion, loader, train_dataset, it
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'num_workers': num_workers, 'mean': mean_t, 'stdev': stdev_t,
           'cov': cov, 'min': min(steady), 'max': max(steady)}


def find_dataset_root(input_root="/kaggle/input", max_depth=3):
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
            if os.path.isdir(os.path.join(c, "train")) and \
                    os.path.isdir(os.path.join(c, "validate")):
                return c
        all_dirs = next_level
    return None  # allow falling back to the config's own root_dir/validate_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--n-iters", type=int, default=80)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--workers-list", type=str, default="3,4")
    args = parser.parse_args()

    try:
        import psutil  # noqa: F401
    except ImportError:
        print("psutil not installed -- CPU usage reporting will fail.",
             flush=True)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | total memory: {props.total_memory / 1e9:.2f}GB",
             flush=True)
    print(f"CPU logical cores (os.cpu_count()): {os.cpu_count()}", flush=True)

    dataset_root = args.dataset_root or find_dataset_root()
    print(f"Using dataset_root={dataset_root}", flush=True)

    hypes = yaml_utils.load_yaml(HYPES_PATH, None)

    results = []
    for nw in [int(x) for x in args.workers_list.split(",")]:
        results.append(run_one(nw, dataset_root, args.n_iters, args.n_warmup, hypes))

    print(f"\n{'=' * 20} FINAL COMPARISON {'=' * 20}", flush=True)
    for r in results:
        print(f"num_workers={r['num_workers']}: mean={r['mean']:.3f}s/it | "
             f"stdev={r['stdev']:.3f}s | cov={r['cov']:.2f} | "
             f"range=[{r['min']:.3f}, {r['max']:.3f}]s", flush=True)


if __name__ == "__main__":
    main()
