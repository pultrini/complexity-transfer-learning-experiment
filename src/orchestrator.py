import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import torch

from config.experiment_config import ExperimentConfig
from config.workflows import Workflow, WorkflowStep
from src.experiment import Experiment
from utils.statistics import StatisticsCalculator


@dataclass(frozen=True)
class ExperimentSpec:
    """A single preset experiment configuration used in the standard model loop."""

    name: str
    model_id: int
    dataset_name: str
    num_epochs: int
    strategy: str
    checkpoint_type: str | None
    output_file: str


class _AggregatableStep(Protocol):
    """Structural type for anything that can be run and aggregated per iteration.

    Both ``ExperimentSpec`` and ``WorkflowStep`` satisfy this shape.
    """

    name: str
    output_file: str


class Orchestrator:
    """Runs multiple iterations of experiments and aggregates the results."""

    MODEL_CONFIGS: ClassVar[list[ExperimentSpec]] = [
        ExperimentSpec(
            name="model1_normal",
            model_id=1,
            dataset_name="TissueMNIST",
            num_epochs=10,
            strategy="normal",
            checkpoint_type=None,
            output_file="results/metrics/metrics_model1.json",
        ),
        ExperimentSpec(
            name="model2_complexity",
            model_id=2,
            dataset_name="BloodMNIST",
            num_epochs=5,
            strategy="complexity",
            checkpoint_type="loss",
            output_file="results/metrics/metrics_model2.json",
        ),
        ExperimentSpec(
            name="model3_loss",
            model_id=3,
            dataset_name="BloodMNIST",
            num_epochs=5,
            strategy="min_loss",
            checkpoint_type="complexity",
            output_file="results/metrics/metrics_model3.json",
        ),
        ExperimentSpec(
            name="model4_mnist",
            model_id=4,
            dataset_name="MNIST",
            num_epochs=10,
            strategy="normal",
            checkpoint_type=None,
            output_file="results/metrics/metrics_model4.json",
        ),
        ExperimentSpec(
            name="model5_fashionmnist",
            model_id=5,
            dataset_name="FashionMNIST",
            num_epochs=10,
            strategy="normal",
            checkpoint_type=None,
            output_file="results/metrics/metrics_model5.json",
        ),
    ]

    def __init__(
        self,
        max_iterations: int = 5,
        base_seed: int = 1234,
        metrics_dir: str = "results/metrics",
        results_dir: str = "results",
        device: str | None = None,
        models_dir: str = "models",
    ):
        self.max_iterations = max_iterations
        self.base_seed = base_seed
        self.metrics_dir = metrics_dir
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_names = [spec.name for spec in self.MODEL_CONFIGS]

    def run(self) -> None:
        """Run the standard model loop across all iterations and compute final statistics."""
        print(f"Starting loop of {self.max_iterations} iterations...")

        aggregated_path = Path(self.metrics_dir) / "all_metrics.json"

        def build_config(spec: ExperimentSpec, _iteration: int) -> ExperimentConfig:
            return ExperimentConfig(
                dataset_name=spec.dataset_name,
                num_epochs=spec.num_epochs,
                device=self.device,
                output_file=spec.output_file,
                strategy=spec.strategy,
                checkpoint_type=spec.checkpoint_type,
                models_dir=self.models_dir,
            )

        all_metrics = self._run_iterations(
            steps=self.MODEL_CONFIGS, build_config=build_config, aggregated_path=aggregated_path
        )

        print("\n==== Loop finished. Computing final statistics... ====")
        self._save_final_statistics(all_metrics)

    def run_workflow(self, workflow: Workflow) -> None:
        """Run a complete transfer learning workflow across all iterations."""
        print(f"\n{'=' * 60}")
        print(f"Running workflow: {workflow.name}")
        print(f"Transfer learning: {workflow.source_dataset} -> {workflow.target_dataset}")
        print(f"{'=' * 60}\n")

        aggregated_path = Path(self.metrics_dir) / f"{workflow.name}_all_metrics.json"

        def build_config(step: WorkflowStep, iteration: int) -> ExperimentConfig:
            return ExperimentConfig(
                dataset_name=step.dataset_name,
                num_epochs=step.num_epochs,
                device=self.device,
                output_file=step.output_file,
                strategy=step.strategy,
                checkpoint_path=step.checkpoint_path,
                models_dir=self.models_dir,
                mlflow_experiment_name=workflow.mlflow_experiment_name,
                mlflow_run_name=f"{step.mlflow_run_name}_iter{iteration}",
                workflow_name=workflow.name,
                step_name=step.name,
            )

        all_metrics = self._run_iterations(
            steps=workflow.steps, build_config=build_config, aggregated_path=aggregated_path
        )

        print(f"\n==== Workflow {workflow.name} finished. Computing final statistics... ====")
        step_names = [step.name for step in workflow.steps]
        self._save_final_statistics(all_metrics, prefix=workflow.name, model_names=step_names)

    def _run_iterations(
        self,
        *,
        steps: list[_AggregatableStep],
        build_config: Callable[[_AggregatableStep, int], ExperimentConfig],
        aggregated_path: Path,
    ) -> dict[str, dict[str, dict | None]]:
        """Run all iterations for a set of steps, aggregating metrics after each one.

        Shared by both the standard model loop (``run``) and transfer learning
        workflows (``run_workflow``) — the only difference between them is how
        each step is turned into an ``ExperimentConfig``.
        """
        all_metrics: dict[str, dict[str, dict | None]] = {}

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n===== Iteration {iteration}/{self.max_iterations} =====")

            for step in steps:
                config = build_config(step, iteration)
                Experiment(config).run()

            metrics_per_step = {step.name: self._read_metrics(step.output_file) for step in steps}

            if all(metrics_per_step.values()):
                all_metrics[f"iteration_{iteration}"] = metrics_per_step
                self._save_aggregated_metrics(all_metrics, aggregated_path)
                print(f">>> Iteration metrics saved to {aggregated_path.name}")
            else:
                print("Some metrics were not found for this iteration. Skipping aggregation.")

        return all_metrics

    def _save_aggregated_metrics(self, all_metrics: dict, aggregated_path: Path) -> None:
        """Persist the aggregated metrics dict to disk as JSON."""
        aggregated_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated_path.write_text(json.dumps(all_metrics, indent=4), encoding="utf-8")

    def _save_final_statistics(
        self, all_metrics: dict, prefix: str = "", model_names: list[str] | None = None
    ) -> None:
        """Compute and persist final mean/std statistics across all iterations."""
        if not all_metrics:
            print("No aggregated metrics found.")
            return

        model_names = model_names or self.model_names
        stats_filename = f"{prefix}_results.json" if prefix else "results.json"
        stats_path = Path(self.results_dir) / stats_filename

        aggregated_name = f"{prefix}_all_metrics.json" if prefix else "all_metrics.json"
        aggregated_path = Path(self.metrics_dir) / aggregated_name

        if not aggregated_path.exists():
            print(f"Aggregated file '{aggregated_path}' was not found.")
            return

        data = json.loads(aggregated_path.read_text(encoding="utf-8"))

        # ddof=1 reproduces the sample standard deviation (N-1) used previously
        # by this orchestrator, as opposed to the population std (ddof=0) used
        # elsewhere in the codebase.
        calculator = StatisticsCalculator(model_names, ddof=1)
        final_results = calculator.calculate(data)

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        calculator.save(final_results, str(stats_path))

        print("\n--- FINAL RESULTS ---")
        print(json.dumps(final_results, indent=4))
        print(f"\nStatistics computed and saved to {stats_path}!")

    def _read_metrics(self, file_path: str) -> dict | None:
        """Read a metrics JSON file, returning None if it doesn't exist."""
        path = Path(file_path)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        print(f"Metrics file not found: {file_path}")
        return None
