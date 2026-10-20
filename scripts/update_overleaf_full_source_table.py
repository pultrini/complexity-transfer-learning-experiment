"""Insert new rows and audit tests by calling the existing statistics code."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_root", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "existing_statistics", args.analysis_root / "testes_estatisticos.py"
    )
    statistics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(statistics)
    conditions = statistics.collect_conditions(str(args.analysis_root / "data_tabelas"))
    summaries = json.loads((args.output_dir / "summary.json").read_text())
    blocks = {}
    for scenario in ("medmnist", "MNIST", "tiny_imagenet"):
        rows = [r for r in summaries if r["scenario"] == scenario]
        block = [r"\cline{2-7}"]
        for index, row in enumerate(rows):
            model = row["model"]
            folder = "Imagenet" if model == "EfficientNetV2-S" else "Resnet50"
            source = args.analysis_root / "data/Dados_todos_dados_treinamento" / scenario / folder
            conditions[(scenario.lower().replace("_", ""), "Adam", 0.001, model)] = {
                "Max Complexity TL (source train + val)": {
                    "acc": statistics.read_per_run_extreme(str(source / "val_accuracy.csv"), "max"),
                    "loss": statistics.read_per_run_extreme(str(source / "val_loss.csv"), "min"),
                }
            }
            optimizer = r"\multirow{2}{*}{Adam $\eta=0.001$}" if index == 0 else ""
            block.append(
                f"& {optimizer} & Max Complexity TL (source train + val) - {model}"
                + f" & {row['accuracy_mean_percent']:.2f} $\\pm$ {row['accuracy_std_percent_ddof0']:.2f}"
                + f" & {row['loss_mean']:.4f} $\\pm$ {row['loss_std_ddof0']:.4f}"
                + f" & {row['accuracy_epoch_mean']:.1f} & --- " + r"\\"
            )
        blocks[scenario] = "\n".join(block) + "\n"

    reports = []
    for (scenario, optimizer, lr, model), groups in sorted(conditions.items()):
        for metric in ("acc", "loss"):
            result = statistics.run_condition_tests(groups, metric)
            reports.append({
                "scenario": scenario, "optimizer": optimizer, "lr": lr,
                "model": model, "metric": metric,
                "sample_sizes": {name: len(values[metric]) for name, values in groups.items()},
                **result,
            })
    (args.output_dir / "statistical_tests_audit.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False, default=lambda value: value.item()), encoding="utf-8"
    )
    with (args.output_dir / "new_condition_tests.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", "model", "metric", "optimizer", "lr", "n", "status", "kruskal_p"])
        for result in reports:
            if result["lr"] == 0.001:
                writer.writerow([result["scenario"], result["model"], result["metric"],
                                 "Adam", 0.001, 5, result["note"], ""])

    table = args.table.read_text()
    # Insert before the end of each existing scenario block.
    for following_scenario, previous in (("Scenario 2", "medmnist"), ("Scenario 3", "MNIST")):
        end = table.rfind(r"\midrule", 0, table.index(following_scenario))
        table = table[:end] + blocks[previous] + table[end:]
    table = table.replace(r"\bottomrule", blocks["tiny_imagenet"] + r"\bottomrule", 1)
    # Repair empty multirow arguments in the supplied table.
    table = table.replace(r"\multirow{}{}{\textbf", r"\multirow{20}{*}{\textbf")
    table = table.replace(r"\multirow{}{}{Adam", r"\multirow{6}{*}{Adam")
    table = table.replace(r"\multirow{}{}{SGD", r"\multirow{6}{*}{SGD")
    table = table.replace(r"\begin{tabular}{llccccc}", r"\begin{tabular}{lllcccc}")
    note = (
        r"\multicolumn{7}{l}{\small Source train + val: source pretrained on training and validation data; $n=5$.} \\" + "\n"
        r"\multicolumn{7}{l}{\small Adam $\eta=0.001$: significance not tested (matching Min Loss TL and No Transfer exports unavailable).} \\" + "\n"
        r"\multicolumn{7}{l}{\small ---: transfer epoch unavailable in the supplied exports.} \\" + "\n"
    )
    table = table.replace(r"\end{tabular}", note + r"\end{tabular}")
    (args.output_dir / "consolidated_results_updated.tex").write_text(table, encoding="utf-8")
    (args.output_dir / "new_rows_consolidated.tex").write_text(
        "\n".join(f"% {scenario}\n{block}" for scenario, block in blocks.items()), encoding="utf-8"
    )
    new_reports = [r for r in reports if r["lr"] == 0.001]
    assert len(new_reports) == 12
    assert all(r["kw_p"] is None and not r["pairwise"] for r in new_reports)
    print(f"Evaluated {len(reports)} conditions/metrics with the existing code.")
    print("All 12 new condition/metric tests report insufficient groups.")
    print("Existing table values and significance markers preserved; six new rows inserted.")


if __name__ == "__main__":
    main()
