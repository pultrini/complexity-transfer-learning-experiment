"""Analyze a neighborhood checkpoint sweep (``--sweep-radius``).

Input: the aggregated ``<workflow>_all_metrics.json`` written by the
orchestrator. Every target step that started from a source checkpoint carries
``transfer_*`` fields (anchor, offset, source_epoch, source_complexity, ...),
copied from the checkpoint sidecar by ``Experiment._transfer_metadata``.

For each iteration and each anchor (Max-C, Min-loss), the neighborhood is the
set of distinct source epochs transferred around that anchor. Near the edges
(e.g. t* = 1) the run substitutes out-of-range offsets by the nearest unused
epochs, so offsets are the TRUE offsets source_epoch - t*. Reported per
neighborhood:

* local oracle   best target accuracy among the transferred epochs;
* regret (pp)    local oracle minus the accuracy of the anchor checkpoint;
* rank           position of the anchor among the epochs (1 = best);
* Kendall tau    between the source criterion and target accuracy across the
                 epochs (C for Max-C; -val_loss for Min-loss; both are
                 computed for every neighborhood).

The regret is LOCAL: it only says whether a nearby epoch would have been better,
not whether a distant one would. It is a lower bound on the global regret.

Usage:
    uv run python scripts/analyze_neighborhood_sweep.py \
        results/<...>/metrics/medmnist_all_metrics.json --out-dir results/<...>/sweep_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ANCHORS = ("max_complexity", "min_loss")
LABELS = {"max_complexity": "Max-C", "min_loss": "Min-loss"}


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else None


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values), sd


def collect(data: dict) -> list[dict]:
    rows = []
    for iteration, steps in sorted(data.items()):
        scratch = steps.get("target_no_transfer", {}).get("Max_accuracy")
        for step_name, metrics in steps.items():
            anchor = metrics.get("transfer_anchor")
            if anchor not in ANCHORS:
                continue
            rows.append({
                "iteration": iteration,
                "step": step_name,
                "anchor": anchor,
                "offset": int(metrics["transfer_offset"]),
                "anchor_epoch": int(metrics["transfer_anchor_epoch"]),
                "source_epoch": int(metrics["transfer_source_epoch"]),
                "slot": int(metrics.get("transfer_slot", metrics["transfer_offset"])),
                "shifted": bool(metrics.get("transfer_shifted", False)),
                "source_complexity": float(metrics["transfer_source_complexity"]),
                "source_val_loss": float(metrics["transfer_source_val_loss"]),
                "target_accuracy_pct": 100 * float(metrics["Max_accuracy"]),
                "target_min_loss": float(metrics["Min_loss"]),
                "scratch_accuracy_pct": 100 * scratch if scratch is not None else None,
            })
    return rows


def neighborhoods(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["iteration"], row["anchor"])][row["source_epoch"]].append(row)

    results = []
    for (iteration, anchor), by_epoch in sorted(grouped.items()):
        points = []
        for epoch, items in sorted(by_epoch.items()):
            points.append({
                "epoch": epoch,
                "acc": statistics.fmean(r["target_accuracy_pct"] for r in items),
                "complexity": items[0]["source_complexity"],
                "val_loss": items[0]["source_val_loss"],
                "is_anchor": any(r["offset"] == 0 for r in items),
            })
        anchor_points = [p for p in points if p["is_anchor"]]
        if not anchor_points:
            continue
        anchor_acc = anchor_points[0]["acc"]
        accs = [p["acc"] for p in points]
        oracle = max(accs)
        best_epoch = points[accs.index(oracle)]["epoch"]
        rank = 1 + sum(a > anchor_acc for a in accs)
        results.append({
            "iteration": iteration,
            "anchor": anchor,
            "anchor_epoch": anchor_points[0]["epoch"],
            "n_epochs": len(points),
            "epoch_range": f"{points[0]['epoch']}-{points[-1]['epoch']}",
            "anchor_accuracy_pct": anchor_acc,
            "local_oracle_accuracy_pct": oracle,
            "local_oracle_epoch": best_epoch,
            "local_regret_pp": oracle - anchor_acc,
            "anchor_rank": rank,
            "anchor_is_local_best": rank == 1,
            "tau_complexity": kendall_tau_b([p["complexity"] for p in points], accs),
            "tau_neg_val_loss": kendall_tau_b([-p["val_loss"] for p in points], accs),
        })
    return results


def profile(rows: list[dict]) -> list[dict]:
    by_offset: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_offset[(row["anchor"], row["offset"])].append(row["target_accuracy_pct"])
    out = []
    for (anchor, offset), values in sorted(by_offset.items()):
        mean, sd = mean_sd(values)
        out.append({"anchor": anchor, "offset": offset, "n": len(values),
                    "accuracy_mean_pct": mean, "accuracy_sd_pct": sd})
    return out


def summarize(hoods: list[dict]) -> list[dict]:
    summary = []
    for anchor in ANCHORS:
        items = [h for h in hoods if h["anchor"] == anchor]
        if not items:
            continue
        regret = mean_sd([h["local_regret_pp"] for h in items])
        rank = mean_sd([h["anchor_rank"] for h in items])
        tau_c = mean_sd([h["tau_complexity"] for h in items])
        tau_l = mean_sd([h["tau_neg_val_loss"] for h in items])
        summary.append({
            "anchor": anchor,
            "n_iterations": len(items),
            "mean_neighborhood_size": statistics.fmean(h["n_epochs"] for h in items),
            "local_regret_pp_mean": regret[0], "local_regret_pp_sd": regret[1],
            "anchor_rank_mean": rank[0],
            "anchor_is_local_best": sum(h["anchor_is_local_best"] for h in items),
            "tau_complexity_mean": tau_c[0], "tau_complexity_sd": tau_c[1],
            "tau_neg_val_loss_mean": tau_l[0], "tau_neg_val_loss_sd": tau_l[1],
        })
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def write_markdown(path: Path, label: str, summary: list[dict], prof: list[dict]) -> None:
    lines = [f"# Neighborhood sweep -- {label}", "",
             "Local regret: best target accuracy among the transferred neighbor epochs "
             "minus the accuracy of the anchor checkpoint (pp; lower bound on the global regret).",
             "", "| Anchor | n | Epochs/neighborhood | Local regret (pp) | Mean rank | "
             "Anchor is local best | tau(C, acc) | tau(-loss, acc) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(
            f"| {LABELS[s['anchor']]} | {s['n_iterations']} | {fmt(s['mean_neighborhood_size'], 1)} | "
            f"{fmt(s['local_regret_pp_mean'])} ± {fmt(s['local_regret_pp_sd'])} | "
            f"{fmt(s['anchor_rank_mean'], 1)} | {s['anchor_is_local_best']}/{s['n_iterations']} | "
            f"{fmt(s['tau_complexity_mean'])} ± {fmt(s['tau_complexity_sd'])} | "
            f"{fmt(s['tau_neg_val_loss_mean'])} ± {fmt(s['tau_neg_val_loss_sd'])} |")
    lines += ["", "## Target accuracy by offset (mean ± SD over iterations)", "",
              "| Anchor | Offset | n | Accuracy (%) |", "|---|---:|---:|---:|"]
    for p in prof:
        lines.append(f"| {LABELS[p['anchor']]} | {p['offset']:+d} | {p['n']} | "
                     f"{fmt(p['accuracy_mean_pct'])} ± {fmt(p['accuracy_sd_pct'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path_stem: Path, label: str, prof: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for anchor, marker in zip(ANCHORS, ("o", "s"), strict=True):
        pts = [p for p in prof if p["anchor"] == anchor]
        if not pts:
            continue
        ax.errorbar([p["offset"] for p in pts], [p["accuracy_mean_pct"] for p in pts],
                    yerr=[p["accuracy_sd_pct"] for p in pts], marker=marker, capsize=2,
                    linewidth=1, markersize=3, label=LABELS[anchor])
    ax.axvline(0, color="0.6", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Offset from anchor epoch $t^\\star$ (epochs)")
    ax.set_ylabel("Target accuracy (%)")
    ax.set_title(label, fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("all_metrics", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    data = json.loads(args.all_metrics.read_text(encoding="utf-8"))
    rows = collect(data)
    if not rows:
        raise SystemExit("No target step with transfer_* metadata found. "
                         "Was the workflow run with --sweep-radius > 0 on this branch?")
    hoods = neighborhoods(rows)
    prof = profile(rows)
    summary = summarize(hoods)

    out_dir = args.out_dir or args.all_metrics.parent / "sweep_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "sweep_checkpoints.csv", rows)
    write_csv(out_dir / "sweep_neighborhoods.csv", hoods)
    write_csv(out_dir / "sweep_profile.csv", prof)
    write_csv(out_dir / "sweep_summary.csv", summary)
    label = args.label or args.all_metrics.stem
    write_markdown(out_dir / "sweep_summary.md", label, summary, prof)
    plot(out_dir / "sweep_profile", label, prof)
    print((out_dir / "sweep_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
