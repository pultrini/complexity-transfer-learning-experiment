"""Measure the cost of computing the LMC complexity C(theta) per checkpoint.

Fills block 2 of the revision (Sec. III-D): "(XX s for ResNet50 on our hardware,
XX% of the time of one pre-training epoch)".

The timed function is ``Trainer._complexity_metrics`` -- the same code path used
during pre-training -- so the measured time is exactly what the experiments paid
per epoch. No training and no dataset are needed.

The cost of one pre-training epoch can be given in two ways:
  * --epoch-seconds ARCH:SCENARIO=SECONDS (measured from your logs; preferred), or
  * a synthetic estimate: time of one optimizer step on a random batch
    (128 x 1 x 224 x 224, Adam) x number of steps per epoch. This excludes data
    loading and validation, so it UNDER-estimates the epoch time and therefore
    OVER-estimates the relative overhead of C (conservative for the paper).

Usage (from the repository root, on the GPU workstation):
    uv run python scripts/benchmark_complexity_cost.py --device cuda:0
    uv run python scripts/benchmark_complexity_cost.py --device cuda:0 \
        --variants adaptive_linear fixed_256 normalized hard_binning
    uv run python scripts/benchmark_complexity_cost.py --device cuda:0 \
        --epoch-seconds resnet50:S1=412.5 efficientnet_v2_s:S1=530.0

Outputs: results/complexity_cost/complexity_cost.{csv,json} and a Markdown
table printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import ModelFactory  # noqa: E402
from src.trainer import Trainer  # noqa: E402

# Source datasets of the paper (Table I): number of classes K and size of the
# 80% training split used in pre-training.
SCENARIOS = {
    "S1": {"source": "TissueMNIST", "n_classes": 8, "n_train": 189_109},
    "S2": {"source": "MNIST", "n_classes": 10, "n_train": 56_000},
    "S3": {"source": "Tiny ImageNet", "n_classes": 200, "n_train": 96_000},
}
ARCHITECTURES = ("resnet50", "efficientnet_v2_s")
BATCH_SIZE = 128
INPUT_SHAPE = (1, 224, 224)


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def time_call(fn, device: str, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return times


def time_train_step(model: nn.Module, device: str, n_classes: int,
                    repeats: int, warmup: int) -> list[float]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(BATCH_SIZE, *INPUT_SHAPE, device=device)
    y = torch.randint(0, n_classes, (BATCH_SIZE,), device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    return time_call(step, device, repeats, warmup)


def parse_epoch_seconds(items: list[str]) -> dict[tuple[str, str], float]:
    parsed = {}
    for item in items:
        key, value = item.split("=")
        arch, scenario = key.split(":")
        parsed[(arch, scenario)] = float(value)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--variants", nargs="+", default=["adaptive_linear"],
                        help="complexity variants as in config.experiment_config")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--step-repeats", type=int, default=20)
    parser.add_argument("--skip-train-step", action="store_true",
                        help="do not estimate the epoch time from a synthetic step")
    parser.add_argument("--epoch-seconds", nargs="*", default=[],
                        help="measured epoch times, e.g. resnet50:S1=412.5")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "complexity_cost"))
    args = parser.parse_args()

    measured_epoch = parse_epoch_seconds(args.epoch_seconds)
    torch.manual_seed(0)
    rows = []

    for arch in args.architectures:
        for scenario in args.scenarios:
            info = SCENARIOS[scenario]
            model = ModelFactory(info["source"], info["n_classes"],
                                 architecture=arch).create()
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            step_times = None
            if not args.skip_train_step:
                step_model = ModelFactory(info["source"], info["n_classes"],
                                          architecture=arch).create().to(args.device)
                step_times = time_train_step(step_model, args.device, info["n_classes"],
                                             args.step_repeats, args.warmup)
                del step_model
            steps_per_epoch = math.ceil(info["n_train"] / BATCH_SIZE)

            for variant in args.variants:
                trainer = Trainer(model, args.device, complexity_variant=variant)
                c_times = time_call(trainer._complexity_metrics, args.device,
                                    args.repeats, args.warmup)
                c_median = statistics.median(c_times)

                row = {
                    "architecture": arch,
                    "scenario": scenario,
                    "n_classes": info["n_classes"],
                    "n_trainable_params": n_params,
                    "variant": variant,
                    "device": args.device,
                    "c_seconds_median": c_median,
                    "c_seconds_min": min(c_times),
                    "c_seconds_max": max(c_times),
                    "repeats": args.repeats,
                    "steps_per_epoch": steps_per_epoch,
                }
                if step_times is not None:
                    step_median = statistics.median(step_times)
                    est_epoch = step_median * steps_per_epoch
                    row["train_step_seconds_median"] = step_median
                    row["epoch_seconds_synthetic"] = est_epoch
                    row["c_percent_of_epoch_synthetic"] = 100 * c_median / est_epoch
                if (arch, scenario) in measured_epoch:
                    epoch = measured_epoch[(arch, scenario)]
                    row["epoch_seconds_measured"] = epoch
                    row["c_percent_of_epoch_measured"] = 100 * c_median / epoch
                rows.append(row)
                print(f"{arch:18s} {scenario} {variant:24s} C = {c_median:.4f} s",
                      file=sys.stderr)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                    if k in rows[0] else 99)
    with open(out_dir / "complexity_cost.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(args.device)
        if args.device.startswith("cuda") else None,
        "batch_size": BATCH_SIZE,
        "input_shape": INPUT_SHAPE,
        "note": "synthetic epoch = median optimizer step on random data x steps; "
                "excludes data loading and validation (overhead is an upper bound)",
    }
    (out_dir / "complexity_cost.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")

    pct_key = ("c_percent_of_epoch_measured" if measured_epoch
               else "c_percent_of_epoch_synthetic")
    print("\n| Architecture | Scenario | M | Variant | C time (s, median) | % of epoch |")
    print("|---|---|---:|---|---:|---:|")
    for r in rows:
        pct = r.get(pct_key, r.get("c_percent_of_epoch_synthetic"))
        pct_s = f"{pct:.3f}" if pct is not None else "--"
        print(f"| {r['architecture']} | {r['scenario']} | {r['n_trainable_params']:,} | "
              f"{r['variant']} | {r['c_seconds_median']:.4f} | {pct_s} |")
    print(f"\nSaved to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
