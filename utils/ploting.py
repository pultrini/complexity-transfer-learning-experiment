"""
ieee_plotting.py
================
Generates IEEE-style plots directly from MLflow metric history, without a
manual CSV export step. Meant to be called automatically at the end of
Orchestrator.run_workflow() / LMOrchestrator.run_workflow(), so plots are
always up to date with whatever has finished running so far.

Design notes:
  - "multi_config" plots (No Transfer / Min Loss / Max Complexity for one
    architecture) only need the architecture that just finished, so they are
    regenerated unconditionally after every run.
  - "multi_model" plots (comparing architectures against each other) need
    every architecture's data to exist. This module checks which
    architectures currently have MLflow runs for the given workflow and
    silently skips (with a printed notice) any comparison it cannot complete
    yet -- the next call (once the remaining architecture finishes) will
    pick it up automatically.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient

MODEL_COLORS = ["#2166ac", "#d6604d", "#4dac26", "#8073ac"]
CONFIG_COLOR_MAP = {
    "No Transfer": "#d6604d",
    "Max Complexity": "#1a9641",
    "Min Loss": "#2166ac",
}

FONT_SIZE = 18
FIG_SIZE = (8, 5)
LINE_WIDTH = 2.2
MARKER_SIZE = 6
FILL_ALPHA = 0.18
DPI = 300

# Human-readable labels for known architectures, in a stable plotting order.
ARCHITECTURE_LABELS = {
    "efficientnet_v2_s": "EfficientNetV2",
    "resnet50": "ResNet-50",
}


def _fetch_metric_stats(
    client: MlflowClient, experiment_name: str, filter_string: str, metric_key: str
) -> pd.DataFrame | None:
    """Fetch a metric's per-epoch history across all matching runs and
    aggregate into mean +- std per step.

    Returns None if the experiment, matching runs, or metric data don't
    exist yet (e.g. called before that configuration has finished running).
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_string)
    if not runs:
        return None

    rows = []
    for run in runs:
        history = client.get_metric_history(run.info.run_id, metric_key)
        rows.extend({"step": point.step, "value": point.value} for point in history)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    stats = df.groupby("step")["value"].agg(["mean", "std"]).reset_index().rename(columns={"step": "x"})
    stats["std"] = stats["std"].fillna(0)
    stats["low"] = stats["mean"] - stats["std"]
    stats["high"] = stats["mean"] + stats["std"]
    return stats


def _apply_ieee_style(fig, ax, title: str, x_label: str, y_label: str) -> None:
    """IEEE house style: white background, no top/right spines, subtle grid."""
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["bottom"].set_color("#222222")

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#cccccc", alpha=0.85)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.set_title(title, fontsize=FONT_SIZE + 4, fontweight="bold", pad=14)
    ax.set_xlabel(x_label, fontsize=FONT_SIZE + 2, labelpad=8)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE + 2, labelpad=8)
    ax.tick_params(axis="both", labelsize=FONT_SIZE, length=4, width=0.9)

    legend = ax.get_legend()
    if legend:
        legend.get_title().set_fontsize(FONT_SIZE - 2)
        legend.get_title().set_fontweight("bold")
        for text in legend.get_texts():
            text.set_fontsize(FONT_SIZE - 2)
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_edgecolor("#aaaaaa")
        legend.get_frame().set_alpha(0.95)


def _render_plot(
    series: list[tuple[str, pd.DataFrame, str]],
    title: str,
    x_label: str,
    y_label: str,
    legend_title: str,
    output_path: Path,
) -> None:
    """Render one figure from a list of (label, stats_df, color) tuples."""
    sns.set_theme(style="white", font="serif")
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for label, stats, color in series:
        ax.plot(
            stats["x"], stats["mean"], label=label, color=color,
            linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, zorder=3,
        )
        ax.fill_between(stats["x"], stats["low"], stats["high"], color=color, alpha=FILL_ALPHA, zorder=2)

    ax.legend(title=legend_title, loc="best", framealpha=0.95, handlelength=1.8, handletextpad=0.5)
    _apply_ieee_style(fig, ax, title=title, x_label=x_label, y_label=y_label)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok  {output_path.name}")


def _discover_architectures(client: MlflowClient, experiment_name: str, workflow_name: str) -> list[str]:
    """Return which known architectures currently have logged runs for this workflow."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    found = []
    for arch in ARCHITECTURE_LABELS:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.workflow = '{workflow_name}' and params.architecture = '{arch}'",
            max_results=1,
        )
        if runs:
            found.append(arch)
    return found


def _plot_multi_model_stage(
    client: MlflowClient,
    experiment_name: str,
    workflow_name: str,
    step_name: str,
    metric_key: str,
    title: str,
    y_label: str,
    legend_title: str,
    output_path: Path,
) -> None:
    """Plot a single stage's metric across every architecture that has data for it.

    Skips (with a notice) if fewer than two architectures currently have
    data -- a comparison plot needs at least two series to be meaningful.
    """
    architectures = _discover_architectures(client, experiment_name, workflow_name)
    if len(architectures) < 2:
        print(
            f"  [info] Apenas {len(architectures)} arquitetura(s) com dados para "
            f"'{workflow_name}/{step_name}' -- aguardando as demais para gerar {output_path.name}."
        )
        return

    series = []
    for i, arch in enumerate(architectures):
        stats = _fetch_metric_stats(
            client,
            experiment_name,
            filter_string=f"params.workflow = '{workflow_name}' and params.step = '{step_name}' "
            f"and params.architecture = '{arch}'",
            metric_key=metric_key,
        )
        if stats is None:
            continue
        series.append((ARCHITECTURE_LABELS[arch], stats, MODEL_COLORS[i % len(MODEL_COLORS)]))

    if len(series) < 2:
        return

    _render_plot(series, title=title, x_label="Step", y_label=y_label, legend_title=legend_title, output_path=output_path)


def _plot_multi_config_stage(
    client: MlflowClient,
    experiment_name: str,
    workflow_name: str,
    architecture: str,
    metric_key: str,
    title: str,
    y_label: str,
    legend_title: str,
    output_path: Path,
) -> None:
    """Plot No Transfer / Max Complexity / Min Loss for one architecture."""
    step_labels = {
        "target_no_transfer": "No Transfer",
        "target_max_complexity": "Max Complexity",
        "target_min_loss": "Min Loss",
    }

    series = []
    for step_name, label in step_labels.items():
        stats = _fetch_metric_stats(
            client,
            experiment_name,
            filter_string=f"params.workflow = '{workflow_name}' and params.step = '{step_name}' "
            f"and params.architecture = '{architecture}'",
            metric_key=metric_key,
        )
        if stats is None:
            print(f"  [info] Sem dados ainda para '{step_name}' ({architecture}) -- pulando {output_path.name}.")
            return
        series.append((label, stats, CONFIG_COLOR_MAP[label]))

    _render_plot(series, title=title, x_label="Epoch", y_label=y_label, legend_title=legend_title, output_path=output_path)


def generate_workflow_plots(
    workflow_name: str,
    mlflow_experiment_name: str,
    architecture: str,
    source_dataset_label: str,
    target_dataset_label: str,
    output_dir: str = "results/plots",
    tracking_uri: str | None = None,
) -> None:
    """Generate every IEEE plot derivable from the current MLflow state for
    this workflow, called automatically at the end of a workflow run.

    Args:
        workflow_name: The workflow's internal name (e.g. 'medmnist'), used
            to filter runs via the 'workflow' param.
        mlflow_experiment_name: The MLflow experiment these runs were logged
            under (e.g. 'medmnist_transfer').
        architecture: The architecture that just finished running -- used to
            (re)generate its multi-config plots unconditionally.
        source_dataset_label: Display name of the source dataset for plot
            titles (e.g. 'TissueMNIST').
        target_dataset_label: Display name of the target dataset for plot
            titles (e.g. 'BloodMNIST').
        output_dir: Directory where .pdf/.png plots are written.
        tracking_uri: MLflow tracking URI. If omitted, uses whatever is
            already configured (MLFLOW_TRACKING_URI env var or mlflow default).
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    output_dir_path = Path(output_dir)
    arch_label = ARCHITECTURE_LABELS.get(architecture, architecture)

    print(f"\nGenerating plots for workflow '{workflow_name}' (architecture: {arch_label})...\n")

    # --- Multi-config plots for the architecture that just finished ---
    _plot_multi_config_stage(
        client, mlflow_experiment_name, workflow_name, architecture,
        metric_key="val_accuracy",
        title=f"{arch_label} \u2014 {target_dataset_label}",
        y_label="Accuracy",
        legend_title="Validation Accuracy\n(Mean \u00b1 Std)",
        output_path=output_dir_path / f"accuracy_{workflow_name}_{architecture}.pdf",
    )
    _plot_multi_config_stage(
        client, mlflow_experiment_name, workflow_name, architecture,
        metric_key="val_loss",
        title=f"{arch_label} \u2014 {target_dataset_label}",
        y_label="Loss",
        legend_title="Validation Loss\n(Mean \u00b1 Std)",
        output_path=output_dir_path / f"loss_{workflow_name}_{architecture}.pdf",
    )

    # --- Multi-model comparison plots (source pretraining stage) ---
    # Only completes once every architecture has finished; otherwise skipped
    # with a notice and picked up automatically on a later call.
    _plot_multi_model_stage(
        client, mlflow_experiment_name, workflow_name, step_name="source",
        metric_key="val_accuracy",
        title=f"Accuracy - {source_dataset_label}",
        y_label="Accuracy",
        legend_title="Validation Accuracy\n(Mean \u00b1 STD)",
        output_path=output_dir_path / f"accuracy_{workflow_name}_source.pdf",
    )
    _plot_multi_model_stage(
        client, mlflow_experiment_name, workflow_name, step_name="source",
        metric_key="val_loss",
        title=f"Loss - {source_dataset_label}",
        y_label="Loss",
        legend_title="Validation Loss\n(Mean \u00b1 STD)",
        output_path=output_dir_path / f"loss_{workflow_name}_source.pdf",
    )
    _plot_multi_model_stage(
        client, mlflow_experiment_name, workflow_name, step_name="source",
        metric_key="complexity",
        title=f"Complexity - {source_dataset_label}",
        y_label="Complexity",
        legend_title="Training Complexity\n(Mean \u00b1 STD)",
        output_path=output_dir_path / f"complexity_{workflow_name}_source.pdf",
    )

    print(f"\nPlots (up to date) available in '{output_dir_path}/'.")
