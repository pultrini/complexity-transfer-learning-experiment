import gc
import json
from pathlib import Path

import torch

from config.lm_experiment_config import LMExperimentConfig
from config.lm_workflow import LMWorkflow
from src.lm_experiment import LMExperiment
from utils.statistics import LM_METRICS, StatisticsCalculator


class LMOrchestrator:
    """Runs multiple iterations of a language-model transfer learning
    workflow and aggregates the results.

    Mirrors ``Orchestrator`` (vision), but only supports workflow-style runs
    (there is no standard-loop/MODEL_CONFIGS equivalent for LM experiments
    yet) and uses ``LM_METRICS`` (no accuracy) when aggregating statistics.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        metrics_dir: str = "results/metrics",
        results_dir: str = "results",
        device: str | None = None,
        models_dir: str = "models",
        data_root: str = "data",
        tokenizer_name: str = "roberta-base",
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_attention_heads: int = 4,
        seq_length: int = 32,
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        max_train_samples: int | None = None,
    ):
        self.max_iterations = max_iterations
        self.metrics_dir = metrics_dir
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.data_root = data_root
        self.tokenizer_name = tokenizer_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_train_samples = max_train_samples
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def run_workflow(self, workflow: LMWorkflow) -> None:
        """Run a complete LM transfer learning workflow across all iterations."""
        print(f"\n{'=' * 60}")
        print(f"Running LM workflow: {workflow.name}")
        print(f"Transfer learning: {workflow.source_dataset} -> {workflow.target_dataset}")
        print(f"{'=' * 60}\n")

        aggregated_path = Path(self.metrics_dir) / f"{workflow.name}_all_metrics.json"
        all_metrics: dict[str, dict[str, dict | None]] = {}

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n===== Iteration {iteration}/{self.max_iterations} =====")

            for step in workflow.steps:
                config = self._build_config(workflow, step, iteration)
                LMExperiment(config).run()
                self._release_gpu_memory()

            metrics_per_step = {
                step.name: self._read_metrics(step.output_file) for step in workflow.steps
            }

            if all(metrics_per_step.values()):
                all_metrics[f"iteration_{iteration}"] = metrics_per_step
                self._save_aggregated_metrics(all_metrics, aggregated_path)
                print(f">>> Iteration metrics saved to {aggregated_path.name}")
            else:
                print("Some metrics were not found for this iteration. Skipping aggregation.")

        print(f"\n==== Workflow {workflow.name} finished. Computing final statistics... ====")
        step_names = [step.name for step in workflow.steps]
        self._save_final_statistics(all_metrics, prefix=workflow.name, model_names=step_names)

    def _build_config(self, workflow: LMWorkflow, step, iteration: int) -> LMExperimentConfig:
        return LMExperimentConfig(
            dataset_name=step.dataset_name,
            num_epochs=step.num_epochs,
            device=self.device,
            output_file=step.output_file,
            strategy=step.strategy,
            checkpoint_path=step.checkpoint_path,
            models_dir=self.models_dir,
            data_root=self.data_root,
            tokenizer_name=self.tokenizer_name,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            seq_length=self.seq_length,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_train_samples=self.max_train_samples,
            mlflow_experiment_name=workflow.mlflow_experiment_name,
            mlflow_run_name=f"{step.mlflow_run_name}_iter{iteration}",
            workflow_name=workflow.name,
            step_name=step.name,
        )

    def _release_gpu_memory(self) -> None:
        """Force garbage collection and release cached CUDA memory between runs."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_aggregated_metrics(self, all_metrics: dict, aggregated_path: Path) -> None:
        aggregated_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated_path.write_text(json.dumps(all_metrics, indent=4), encoding="utf-8")

    def _save_final_statistics(
        self, all_metrics: dict, prefix: str, model_names: list[str]
    ) -> None:
        if not all_metrics:
            print("No aggregated metrics found.")
            return

        stats_path = Path(self.results_dir) / f"{prefix}_results.json"
        aggregated_path = Path(self.metrics_dir) / f"{prefix}_all_metrics.json"

        if not aggregated_path.exists():
            print(f"Aggregated file '{aggregated_path}' was not found.")
            return

        data = json.loads(aggregated_path.read_text(encoding="utf-8"))

        calculator = StatisticsCalculator(model_names, ddof=1, metrics=LM_METRICS)
        final_results = calculator.calculate(data)

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        calculator.save(final_results, str(stats_path))

        print("\n--- FINAL RESULTS ---")
        print(json.dumps(final_results, indent=4))
        print(f"\nStatistics computed and saved to {stats_path}!")

    def _read_metrics(self, file_path: str) -> dict | None:
        path = Path(file_path)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        print(f"Metrics file not found: {file_path}")
        return None
