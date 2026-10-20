"""Generate Overleaf rows from MLflow exports of source train+validation runs.

Matches criando_graficos/tabela.py: per-run extrema, population SD (ddof=0),
and average epoch. Does not infer optimizer, LR or architecture from metrics.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def read_extrema(path, maximize):
    groups = defaultdict(list)
    names = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            run_id = row["Run ID"]
            value, step = float(row["value"]), int(row["step"])
            if not run_id or not math.isfinite(value):
                raise ValueError(f"Invalid run/value in {path}")
            names[run_id] = row["Run"]
            groups[run_id].append((step, value))
    if not groups:
        raise ValueError(f"Empty export: {path}")
    result = {}
    for run_id, rows in sorted(groups.items()):
        rows.sort()
        if [step for step, _ in rows] != list(range(1, 11)):
            raise ValueError(f"Expected epochs 1..10 exactly once: {path}, {run_id}")
        if "_max_complexity_run_iter" not in names[run_id]:
            raise ValueError(f"Unexpected strategy: {names[run_id]}")
        epoch, value = (max if maximize else min)(rows, key=lambda item: item[1])
        result[run_id] = {"run": names[run_id], "value": value, "epoch": epoch}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--imagenet-label", default="Imagenet (unconfirmed)")
    parser.add_argument("--optimizer-label", default="Optimizer/LR unconfirmed")
    args = parser.parse_args()
    summaries, runs, latex = [], [], []
    for accuracy_path in sorted(args.data_root.glob("*/*/val_accuracy.csv")):
        scenario, folder = accuracy_path.parent.relative_to(args.data_root).parts
        model = "ResNet50" if folder.lower() == "resnet50" else args.imagenet_label
        accuracy = read_extrema(accuracy_path, True)
        loss = read_extrema(accuracy_path.with_name("val_loss.csv"), False)
        if accuracy.keys() != loss.keys() or len(accuracy) != 5:
            raise ValueError(f"Expected five matching run IDs: {accuracy_path}")
        if any(accuracy[key]["run"] != loss[key]["run"] for key in accuracy):
            raise ValueError(f"Run names differ between exports: {accuracy_path}")
        acc_values = [item["value"] * 100 for item in accuracy.values()]
        loss_values = [item["value"] for item in loss.values()]
        summary = {
            "scenario": scenario, "model": model, "n": len(accuracy),
            "accuracy_mean_percent": mean(acc_values),
            "accuracy_std_percent_ddof0": pstdev(acc_values),
            "accuracy_epoch_mean": mean(item["epoch"] for item in accuracy.values()),
            "loss_mean": mean(loss_values), "loss_std_ddof0": pstdev(loss_values),
            "loss_epoch_mean": mean(item["epoch"] for item in loss.values()),
        }
        summaries.append(summary)
        for run_id, item in accuracy.items():
            runs.append({"scenario": scenario, "model_folder": folder,
                         "run_id": run_id, "run": item["run"],
                         "max_accuracy": item["value"], "accuracy_epoch": item["epoch"],
                         "min_loss": loss[run_id]["value"], "loss_epoch": loss[run_id]["epoch"]})
        latex.extend([
            f"% {scenario}; n=5; SD ddof=0; {args.optimizer_label}",
            args.optimizer_label + " (source train + val) & Max Complexity TL - " + model
            + f" & {mean(acc_values):.2f} $\\pm$ {pstdev(acc_values):.2f}"
            + f" & {summary['accuracy_epoch_mean']:.1f}"
            + f" & {mean(loss_values):.4f} $\\pm$ {pstdev(loss_values):.4f}"
            + f" & {summary['loss_epoch_mean']:.1f} & N/A " + r"\\",
            "",
        ])
    if not summaries:
        raise ValueError(f"No exports found under {args.data_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overleaf_rows.tex").write_text("\n".join(latex), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (args.output_dir / "per_run_extrema.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    print("\n".join(latex))


if __name__ == "__main__":
    main()
