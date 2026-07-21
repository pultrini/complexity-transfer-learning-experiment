import json
import math
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn

from config.lm_experiment_config import LMExperimentConfig
from src.lm_dataset import LMDatasetManager
from src.lm_model_factory import LanguageModelFactory
from src.lm_trainer import LMTrainer
from utils.optmizers import build_optmizer


class LMExperiment:
    """Orchestrates a single language model training experiment run.

    Mirrors ``Experiment`` (vision) in structure and MLflow logging
    conventions, but reports ``Min_loss``/``Max_complexity`` only -- there is
    no accuracy-equivalent metric for a language modeling objective.
    """

    def __init__(self, config: LMExperimentConfig):
        self.config = config
        self.dataset_manager = LMDatasetManager(
            dataset_name=config.dataset_name,
            tokenizer_name=config.tokenizer_name,
            seq_length=config.seq_length,
            data_root=config.data_root,
            batch_size=config.batch_size,
            max_train_samples=config.max_train_samples,
        )
        self.model_factory = LanguageModelFactory(
            vocab_size=self.dataset_manager.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            seq_length=config.seq_length,
            models_dir=config.models_dir,
        )

    def run(self) -> dict[str, float]:
        """Run the full experiment and return the final aggregated metrics."""
        print(f"--- Starting LM run for dataset: {self.config.dataset_name} ---")

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
            optimizer = build_optmizer(
                name=self.config.optimizer_name,
                parameters=model.parameters(),
                learning_rate=self.config.learning_rate,
            )
            criterion = nn.CrossEntropyLoss()
            trainer = LMTrainer(model, self.config.device)

            total_steps = len(self.dataset_manager.train_loader) * self.config.num_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=max(total_steps, 1),
                pct_start=0.1,
                anneal_strategy="cos",
            )

            history = self._train_loop(trainer, criterion, optimizer, scheduler)
            results = self._summarize_results(history)

            mlflow.log_metrics(
                {
                    "min_loss": results["Min_loss"],
                    "max_complexity": results["Max_complexity"],
                }
            )

            self._save_results(results)
            mlflow.log_artifact(self.config.output_file)

            print(f"Run results saved to: {self.config.output_file}")
            return results

    def _log_run_params(self) -> None:
        """Log static run parameters to MLflow."""
        mlflow.log_params(
            {
                "dataset": self.config.dataset_name,
                "num_epochs": self.config.num_epochs,
                "optimizer": self.config.optimizer_name,
                "learning_rate": self.config.learning_rate,
                "strategy": self.config.strategy,
                "architecture": "transformer_lm",
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "seq_length": self.config.seq_length,
                "tokenizer": self.config.tokenizer_name,
                "vocab_size": self.dataset_manager.vocab_size,
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
        trainer: LMTrainer,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> dict[str, list[float]]:
        """Run the full training loop, logging metrics and saving checkpoints."""
        history: dict[str, list[float]] = {"loss": [], "complexity": []}
        best_loss = math.inf
        best_complexity = -math.inf
        checkpoint_prefix = self.config.dataset_name
        vocab_size = self.dataset_manager.vocab_size

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = trainer.train_epoch(
                self.dataset_manager.train_loader,
                criterion,
                optimizer,
                vocab_size,
                max_grad_norm=self.config.max_grad_norm,
            )
            val_metrics = trainer.validate_epoch(
                self.dataset_manager.val_loader, criterion, vocab_size
            )
            scheduler.step()

            val_loss = val_metrics["loss"]
            complexity = train_metrics["complexity"]

            print(
                f"Epoch {epoch}/{self.config.num_epochs} -> "
                f"Val Loss: {val_loss:.4f} | Complexity: {complexity:.4f}"
            )

            history["loss"].append(val_loss)
            history["complexity"].append(complexity)

            self._log_epoch_metrics(epoch, val_loss, train_metrics)

            if val_loss < best_loss:
                self._save_checkpoint(trainer.model, f"{checkpoint_prefix}_min_loss.pth")
                best_loss = val_loss

            if complexity > best_complexity:
                self._save_checkpoint(trainer.model, f"{checkpoint_prefix}_max_complexity.pth")
                best_complexity = complexity

        return history

    def _log_epoch_metrics(self, epoch: int, val_loss: float, train_metrics: dict) -> None:
        """Log a single epoch's metrics to MLflow."""
        mlflow.log_metric("val_loss", float(val_loss), step=epoch)
        mlflow.log_metric("train_loss", float(train_metrics["loss"]), step=epoch)
        mlflow.log_metric("complexity", float(train_metrics["complexity"]), step=epoch)
        mlflow.log_metric("disequilibrium", float(train_metrics["disequilibrium"]), step=epoch)
        mlflow.log_metric("entropy", float(train_metrics["entropy"]), step=epoch)

    def _save_checkpoint(self, model: nn.Module, filename: str) -> None:
        """Save the model's state dict to the configured models directory."""
        checkpoint_dir = Path(self.config.models_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / filename)

    def _summarize_results(self, history: dict[str, list[float]]) -> dict[str, float]:
        """Reduce the per-epoch history into final best-value metrics."""
        return {
            "Min_loss": float(np.min(history["loss"])),
            "Max_complexity": float(np.max(history["complexity"])),
        }

    def _save_results(self, results: dict[str, float]) -> None:
        """Save the results dict to a JSON file."""
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=4), encoding="utf-8")
