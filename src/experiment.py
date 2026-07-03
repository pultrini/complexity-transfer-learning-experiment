import json
import math
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from config.experiment_config import ExperimentConfig
from src.dataset import DatasetManager
from src.model import ModelFactory
from src.trainer import Trainer

LEARNING_RATE = 0.001


class Experiment:
    """Orchestrates a single training experiment run."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset_manager = DatasetManager(
            dataset_name=config.dataset_name,
            data_root=config.data_root,
        )
        self.model_factory = ModelFactory(
            dataset_name=config.dataset_name,
            n_classes=self.dataset_manager.n_classes,
            models_dir=config.models_dir,
            architecture=config.model_architecture,   # ← novo
        )

    def run(self) -> dict[str, float]:
        """Run the full experiment and return the final aggregated metrics."""
        print(f"--- Starting run for dataset: {self.config.dataset_name} ---")

        self._setup_mlflow()
        run_name = self.config.mlflow_run_name or (
            f"{self.config.dataset_name}_{self.config.strategy}_run"
        )

        with mlflow.start_run(run_name=run_name):
            self._log_run_params()

            model = self.model_factory.create(
                checkpoint_type=self.config.checkpoint_type,
                checkpoint_path=self.config.checkpoint_path,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            trainer = Trainer(model, self.config.device)

            history = self._train_loop(trainer, criterion, optimizer)
            results = self._summarize_results(history)

            mlflow.log_metrics(
                {
                    "best_accuracy": results["Max_accuracy"],
                    "min_loss": results["Min_loss"],
                    "max_complexity": results["Max_complexity"],
                }
            )
            mlflow.log_params(
                {
                    "epoch_min_loss": results["epoch_min_loss"],
                    "epoch_max_complexity": results["epoch_max_complexity"],
                }
            )

            self._save_results(results)
            mlflow.log_artifact(self.config.output_file)

            print(f"Run results saved to: {self.config.output_file}")
            return results

    def _log_run_params(self) -> None:
        """Log static run parameters to MLflow (dataset, strategy, workflow metadata)."""
        mlflow.log_params(
            {
                "dataset": self.config.dataset_name,
                "num_epochs": self.config.num_epochs,
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "strategy": self.config.strategy,
                "architecture": self.config.model_architecture,
            }
        )
        if self.config.workflow_name:
            mlflow.log_param("workflow", self.config.workflow_name)
        if self.config.step_name:
            mlflow.log_param("step", self.config.step_name)
        if self.config.checkpoint_path:
            mlflow.log_param("transfer_from", self.config.checkpoint_path)

    def _setup_mlflow(self) -> None:
        """Configure the MLflow tracking URI and active experiment."""
        mlflow.set_tracking_uri(self.config.mlflow_uri)

        if self.config.mlflow_experiment_name:
            mlflow.set_experiment(experiment_name=self.config.mlflow_experiment_name)
        elif self.config.mlflow_experiment_id:
            mlflow.set_experiment(experiment_id=self.config.mlflow_experiment_id)

    def _train_loop(
        self,
        trainer: Trainer,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, list[float]]:
        """Run the full training loop, logging metrics and saving checkpoints.

        Returns:
            A dict with the per-epoch history of ``accuracy``, ``loss``, and
            ``complexity``.
        """
        history: dict[str, list[float]] = {"accuracy": [], "loss": [], "complexity": []}
        best_loss = math.inf
        best_complexity = -math.inf
        checkpoint_prefix = self.config.dataset_name

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = trainer.train_epoch(
                self.dataset_manager.train_loader,
                criterion,
                optimizer
            )
            val_metrics = trainer.validate_epoch(self.dataset_manager.val_loader, criterion)

            val_acc = val_metrics["acc"]
            val_loss = val_metrics["loss"]
            complexity = train_metrics["complexity"]

            print(
                f"Epoch {epoch}/{self.config.num_epochs} -> "
                f"Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}"
            )

            history["accuracy"].append(val_acc)
            history["loss"].append(val_loss)
            history["complexity"].append(complexity)

            self._log_epoch_metrics(epoch, val_acc, val_loss, train_metrics)

            if val_loss < best_loss:
                self._save_checkpoint(trainer.model, f"{checkpoint_prefix}_min_loss.pth")
                best_loss = val_loss

            if complexity > best_complexity:
                self._save_checkpoint(trainer.model, f"{checkpoint_prefix}_max_complexity.pth")
                best_complexity = complexity

        return history

    def _log_epoch_metrics(
        self, epoch: int, val_acc: float, val_loss: float, train_metrics: dict
    ) -> None:
        """Log a single epoch's metrics to MLflow."""
        mlflow.log_metric("val_accuracy", float(val_acc), step=epoch)
        mlflow.log_metric("val_loss", float(val_loss), step=epoch)
        mlflow.log_metric("complexity", float(train_metrics["complexity"]), step=epoch)
        mlflow.log_metric("disequilibrium", float(train_metrics["disequilibrium"]), step=epoch)
        mlflow.log_metric("entropy", float(train_metrics["entropy"]), step=epoch)

    def _save_checkpoint(self, model: nn.Module, filename: str) -> None:
        """Save the model's state dict to the configured models directory."""
        checkpoint_dir = Path(self.config.models_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / filename
        torch.save(model.state_dict(), checkpoint_path)

    def _summarize_results(self, history: dict[str, list[float]]) -> dict[str, float]:
        """Reduce the per-epoch history into final best-value metrics.

        Epochs are 1-indexed to match the printed training log.
        """
        return {
            "Max_accuracy": float(np.max(history["accuracy"])),
            "Min_loss": float(np.min(history["loss"])),
            "Max_complexity": float(np.max(history["complexity"])),
            "epoch_min_loss": int(np.argmin(history["loss"])) + 1,
            "epoch_max_complexity": int(np.argmax(history["complexity"])) + 1,
        }

    def _save_results(self, results: dict[str, float]) -> None:
        """Save the results dict to a JSON file."""
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=4), encoding="utf-8")
