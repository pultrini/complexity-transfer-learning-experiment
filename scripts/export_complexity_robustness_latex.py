"""Export complexity-robustness checkpoint epochs as CSV, JSON, and LaTeX."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

SCENARIOS = {
    "medmnist": "S1",
    "mnist": "S2",
    "tinyimagenet_catsdogs": "S3",
}
ARCHITECTURES = {
    "resnet50": "ResNet50",
    "efficientnet_v2_s": "EffNetV2-S",
}
VARIANT_LABELS = {
    "adaptive_linear": r"As used (adaptive $N_b$, linear)",
    "fixed_64": r"Fixed $N_b=64$",
    "fixed_256": r"Fixed $N_b=256$",
    "fixed_1024": r"Fixed $N_b=1024$",
    "normalized": r"Normalized $\tilde C$",
    "jensen_shannon": r"Jensen--Shannon $D$",
    "hard_binning": "Hard binning",
    "no_bn_bias": "No BN/bias parameters",
    "quantile_normalization": "Quantile normalization",
    "mean_per_layer": "Mean of per-layer $C$",
}


def sample_standard_deviation(values: list[int]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def read_source_epochs(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    epochs = []
    for iteration, steps in sorted(data.items()):
        source = steps.get("source")
        if not source or "epoch_max_complexity" not in source:
            raise ValueError(f"Missing source epoch in {path} ({iteration})")
        epoch = source["epoch_max_complexity"]
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError(f"Invalid epoch in {path} ({iteration}): {epoch!r}")
        epochs.append(epoch)
    if not epochs:
        raise ValueError(f"No iterations found in {path}")
    return epochs


def collect(root: Path) -> list[dict]:
    rows = []
    for scenario, scenario_label in SCENARIOS.items():
        for architecture, architecture_label in ARCHITECTURES.items():
            for variant, variant_label in VARIANT_LABELS.items():
                metrics_path = (
                    root
                    / scenario
                    / architecture
                    / variant
                    / "metrics"
                    / f"{scenario}_all_metrics.json"
                )
                if not metrics_path.exists():
                    continue
                epochs = read_source_epochs(metrics_path)
                rows.append(
                    {
                        "scenario": scenario_label,
                        "architecture": architecture_label,
                        "variant": variant,
                        "variant_label": variant_label,
                        "epochs": epochs,
                        "mean_epoch": mean(epochs),
                        "std_epoch": sample_standard_deviation(epochs),
                        "n": len(epochs),
                        "metrics_path": str(metrics_path),
                    }
                )
    if not rows:
        raise ValueError(f"No robustness metrics found below {root}")
    return rows


def latex_cell(row: dict) -> str:
    """Render a compact rounded mean; raw values remain in CSV/JSON."""
    return str(math.floor(row["mean_epoch"] + 0.5))


def render_latex(rows: list[dict]) -> str:
    indexed = {
        (row["variant"], row["architecture"], row["scenario"]): row for row in rows
    }
    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{ResNet50 (S1/S2/S3)} & "
        r"\multicolumn{3}{c}{EffNetV2-S (S1/S2/S3)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Variant & S1 & S2 & S3 & S1 & S2 & S3 \\",
        r"\midrule",
    ]
    for variant, label in VARIANT_LABELS.items():
        cells = []
        for architecture in ARCHITECTURES.values():
            for scenario in SCENARIOS.values():
                row = indexed.get((variant, architecture, scenario))
                cells.append(latex_cell(row) if row else r"\XX")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robustness_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    rows = collect(args.robustness_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "complexity_robustness_table.tex").write_text(
        render_latex(rows), encoding="utf-8"
    )
    (args.output_dir / "complexity_robustness_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "complexity_robustness_epochs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "scenario",
            "architecture",
            "variant",
            "mean_epoch",
            "std_epoch",
            "n",
            "epochs",
            "metrics_path",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row[field] for field in fieldnames if field in row},
                    "epochs": ",".join(str(epoch) for epoch in row["epochs"]),
                }
            )

    print((args.output_dir / "complexity_robustness_table.tex").read_text())


if __name__ == "__main__":
    main()
