"""Four-group comparisons reusing criando_graficos/testes_estatisticos.py."""
import argparse
import csv
import importlib.util
import json
import re

from pathlib import Path

OLD = "Max Complexity TL (train only)"
NEW = "Max Complexity TL (train + validation; Adam 0.001)"
LOSS = "Min Loss TL"
BASE = "No Transfer"


def marker(result, group):
    symbols = set()
    for pair in result["pairwise"]:
        if not pair["significant"] or pair["better_group"] != group:
            continue
        other = ({pair["group_a"], pair["group_b"]} - {group}).pop()
        if other == BASE:
            symbols.add("*")
        for groups, symbol in (({OLD, LOSS}, r"\dagger"), ({OLD, NEW}, r"\ddagger"), ({NEW, LOSS}, r"\S")):
            if {group, other} == groups:
                symbols.add(symbol)
    return "$^{" + "".join(s for s in ("*", r"\dagger", r"\ddagger", r"\S") if s in symbols) + "}$" if symbols else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_root", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--transfer-epochs-table", type=Path,
                        help="User-provided table containing transfer epochs for the new runs.")
    args = parser.parse_args()
    transfer_epochs = {}
    if args.transfer_epochs_table:
        scenario = None
        for line in args.transfer_epochs_table.read_text().splitlines():
            match = re.search(r"Scenario ([123])", line)
            if match:
                scenario = {"1": "medmnist", "2": "mnist", "3": "tinyimagenet"}[match[1]]
            if "Max Complexity TL (source train + val) - " in line:
                model = "EfficientNetV2-S" if "EfficientNetV2-S" in line else "ResNet50"
                epoch = line.split("&")[-1].replace("\\\\", "").strip()
                if not re.fullmatch(r"\d+(?:\.\d+)?", epoch):
                    raise ValueError(f"Invalid supplied transfer epoch: {epoch}")
                transfer_epochs[scenario, model] = epoch
        if len(transfer_epochs) != 6:
            raise ValueError("Expected six supplied transfer epochs.")
    spec = importlib.util.spec_from_file_location("existing", args.analysis_root / "testes_estatisticos.py")
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    conditions = stats.collect_conditions(str(args.analysis_root / "data_tabelas"))
    summaries = json.loads((args.output_dir / "summary.json").read_text())
    samples, summary_map = {}, {}
    for row in summaries:
        key = (row["scenario"].lower().replace("_", ""), row["model"])
        folder = "Imagenet" if row["model"] == "EfficientNetV2-S" else "Resnet50"
        root = args.analysis_root / "data/Dados_todos_dados_treinamento" / row["scenario"] / folder
        samples[key] = {metric: stats.read_per_run_extreme(str(root / filename), mode)
                        for metric, filename, mode in (("acc", "val_accuracy.csv", "max"), ("loss", "val_loss.csv", "min"))}
        summary_map[key] = row
    reports, lookup, pair_rows, direct_rows = [], {}, [], []
    for (scenario, optimizer, lr, model), original in sorted(conditions.items()):
        groups = {OLD if name == "Max Complexity TL" else name: data for name, data in original.items()}
        groups[NEW] = samples[scenario, model]
        assert len(groups) == 4
        for metric in ("acc", "loss"):
            result = stats.run_condition_tests(groups, metric)
            metadata = dict(scenario=scenario, optimizer=optimizer, lr=lr, model=model, metric=metric)
            reports.append({**metadata, "sample_sizes": {name: len(data[metric]) for name, data in groups.items()}, **result})
            lookup[scenario, optimizer, lr, model, metric] = result
            pair_rows.extend({**metadata, "kruskal_p": result["kw_p"], **p} for p in result["pairwise"])
            direct = next((p for p in result["pairwise"] if {p["group_a"], p["group_b"]} == {OLD, NEW}), None)
            direct_rows.append({**metadata, "kruskal_p": result["kw_p"], "p_holm": direct["p_holm"] if direct else "",
                                "better_group": direct["better_group"] if direct else "",
                                "status": "tested" if direct else "posthoc not run: Kruskal-Wallis p >= 0.05"})
    (args.output_dir / "four_group_tests.json").write_text(json.dumps(reports, indent=2, default=lambda v: v.item()))
    for filename, rows in (("pairwise_four_groups.csv", pair_rows), ("complexity_train_vs_train_validation.csv", direct_rows)):
        with (args.output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    output, snippets = [], []
    scenario = optimizer = lr = None
    for line in args.table.read_text().splitlines():
        if line.startswith(r"\caption{"):
            line = r"\caption{Consolidated Transfer Learning Results for All Scenarios.}"
        if line.startswith(r"\multicolumn{7}"):
            continue
        match = re.search(r"Scenario ([123])", line)
        if match:
            scenario = {"1": "medmnist", "2": "mnist", "3": "tinyimagenet"}[match[1]]
        match = re.search(r"(Adam|SGD) \$\\eta = ([0-9.]+)", line)
        if match:
            optimizer, lr = match[1], float(match[2])
        if " & " in line and any(name + " - " in line for name in ("Max Complexity TL", LOSS, BASE)):
            model = "EfficientNetV2-S" if "EfficientNetV2-S" in line else "ResNet50"
            group = OLD if "Max Complexity TL - " in line else LOSS if "Min Loss TL - " in line else BASE
            cells = re.sub(r"\$\^\{[^}]*\}\$", "", line).split("&")
            # The first row of a block has its optimizer on the previous line.
            for index, metric in ((len(cells) - 4, "acc"), (len(cells) - 3, "loss")):
                cells[index] = cells[index].rstrip() + marker(lookup[scenario, optimizer, lr, model, metric], group) + " "
            output.append("&".join(cells).replace("Max Complexity TL - ", OLD + " - "))
            if group == BASE and model == "ResNet50":
                block = [f"% New sample compared with {scenario}, {optimizer} {lr}"]
                for model in ("EfficientNetV2-S", "ResNet50"):
                    row = summary_map[scenario, model]
                    block.append("& & " + NEW + " - " + model
                                 + f" & {row['accuracy_mean_percent']:.2f} $\\pm$ {row['accuracy_std_percent_ddof0']:.2f}"
                                 + marker(lookup[scenario, optimizer, lr, model, "acc"], NEW)
                                 + f" & {row['loss_mean']:.4f} $\\pm$ {row['loss_std_ddof0']:.4f}"
                                 + marker(lookup[scenario, optimizer, lr, model, "loss"], NEW)
                                 + f" & {row['accuracy_epoch_mean']:.1f} & {transfer_epochs.get((scenario, model), '---')} " + r"\\")
                output.extend(block)
                snippets.extend(block)
            continue
        if line.startswith(r"\end{tabular}"):
            notes = [
                r"Values: mean $\pm$ SD. N/A: no transfer. The same train+validation group (Adam $\eta=0.001$) is compared in each block.",
                r"Kruskal--Wallis followed, when significant, by Mann--Whitney U with Holm correction over six pairs per block/model/metric ($p<0.05$).",
                r"$^{*}$ Better than No Transfer; $^{\dagger}$ better than the other historical TL strategy.",
                r"$^{\ddagger}$ Better than the other Max Complexity configuration; $^{\S}$ better than the other of new Max Complexity and Min Loss TL.",
            ]
            output.extend(r"\multicolumn{7}{l}{\small " + note + r"} \\" for note in notes)
        line = line.replace(r"\multirow{}{}{\textbf", r"\multirow{24}{*}{\textbf")
        line = line.replace(r"\multirow{}{}{Adam", r"\multirow{8}{*}{Adam").replace(r"\multirow{}{}{SGD", r"\multirow{8}{*}{SGD")
        line = line.replace(r"\begin{tabular}{llccccc}", r"\begin{tabular}{lllcccc}").replace(r"\textbf{Optimizer}", r"\textbf{Historical optimizer}")
        output.append(line)
    (args.output_dir / "consolidated_four_groups.tex").write_text("\n".join(output) + "\n")
    (args.output_dir / "new_rows_four_groups.tex").write_text("\n".join(snippets) + "\n")
    assert len(reports) == 36
    assert all(len(r["pairwise"]) == (6 if r["kw_p"] < 0.05 else 0) for r in reports)
    print(f"Computed {len(reports)} omnibus tests and {len(pair_rows)} post-hoc comparisons.")
    for row in direct_rows:
        if row["better_group"]:
            print(row)


if __name__ == "__main__":
    main()
