import json
import math
import shutil
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from config.experiment_config import ExperimentConfig
from src.dataset import DatasetManager
from src.model import ModelFactory
from src.trainer import Trainer
from utils.optmizers import build_optmizer


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
            optimizer = build_optmizer(
                name=self.config.optimizer_name,
                parameters=model.parameters(),
                learning_rate=self.config.learning_rate
            )
            criterion = nn.CrossEntropyLoss()
            trainer = Trainer(
                model,
                self.config.device,
                complexity_variant=self.config.complexity_variant,
            )

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

            results.update(self._transfer_metadata())
            self._save_results(results)
            mlflow.log_artifact(self.config.output_file)
            self._log_model_and_checkpoints(model)

            print(f"Run results saved to: {self.config.output_file}")
            return results

    def close(self) -> None:
        """Release DataLoader workers held by this experiment."""
        self.dataset_manager.close()

    def _log_run_params(self) -> None:
        """Log static run parameters to MLflow (dataset, strategy, workflow metadata)."""
        mlflow.log_params(
            {
                "dataset": self.config.dataset_name,
                "num_epochs": self.config.num_epochs,
                "optimizer": self.config.optimizer_name,
                "learning_rate": self.config.learning_rate,
                "strategy": self.config.strategy,
                "architecture": self.config.model_architecture,
                "train_with_validation": self.config.train_with_validation,
                "evaluation_split": (
                    "test" if self.config.train_with_validation else "validation"
                ),
                "complexity_variant": self.config.complexity_variant,
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
            mlflow.set_experiment(experiment_id=str(self.config.mlflow_experiment_id))

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
        history: dict[str, list[float]] = {
            "accuracy": [], "loss": [], "complexity": [], "entropy": [], "disequilibrium": [],
        }
        best_loss = math.inf
        best_complexity = -math.inf
        checkpoint_prefix = self.config.dataset_name

        use_windows = (
            self.config.complexity_window_fractions is not None and len(self.config.complexity_window_fractions) > 0
        )
        use_sweep = bool(self.config.sweep_offsets)
        keep_all_epochs = use_windows or use_sweep
        temp_dir = Path()
        if keep_all_epochs:
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "local"
            temp_dir = Path(self.config.models_dir) / f"temp_{checkpoint_prefix}_{run_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)

        train_loader = (
            self.dataset_manager.train_with_validation_loader
            if self.config.train_with_validation
            else self.dataset_manager.train_loader
        )
        evaluation_loader = (
            self.dataset_manager.test_loader
            if self.config.train_with_validation
            else self.dataset_manager.val_loader
        )

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = trainer.train_epoch(train_loader, criterion, optimizer)
            val_metrics = trainer.validate_epoch(evaluation_loader, criterion)

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
            history["entropy"].append(train_metrics["entropy"])
            history["disequilibrium"].append(train_metrics["disequilibrium"])

            self._log_epoch_metrics(epoch, val_acc, val_loss, train_metrics)

            if val_loss < best_loss:
                self._save_checkpoint(trainer.model, f"{checkpoint_prefix}_min_loss.pth")
                best_loss = val_loss

            if keep_all_epochs:
                torch.save(trainer.model.state_dict(), temp_dir / f"epoch_{epoch}.pth")

            if not use_windows and complexity > best_complexity:
                self._save_checkpoint(trainer.model, f"{checkpoint_prefix}_max_complexity.pth")
                best_complexity = complexity

        if use_windows:
            self._process_complexity_windows(history, temp_dir, checkpoint_prefix)
        if use_sweep:
            self._process_neighborhood_sweep(history, temp_dir, checkpoint_prefix)
        if keep_all_epochs:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Always describe the anchor checkpoints just written, so a sidecar left
        # over from an earlier run can never be attached to a newer checkpoint.
        models_dir = Path(self.config.models_dir)
        for anchor in ("max_complexity", "min_loss"):
            t_star = self.anchor_epoch(history, anchor)
            self._write_sidecar(
                models_dir / f"{checkpoint_prefix}_{anchor}.pth", history, anchor, 0, t_star, t_star
            )
        self._save_history(history)
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

    def _log_model_and_checkpoints(self, model: nn.Module) -> None:
        """Track the trained model and its selected checkpoints in MLflow."""
        mlflow.pytorch.log_model(
            model,
            name="trained_model",
            serialization_format="pickle",
        )

        checkpoint_prefix = self.config.dataset_name
        checkpoint_names = [
            f"{checkpoint_prefix}_min_loss.pth",
            f"{checkpoint_prefix}_max_complexity.pth",
        ]
        for fraction in self.config.complexity_window_fractions or []:
            if fraction < 1.0:
                checkpoint_names.append(
                    f"{checkpoint_prefix}_max_complexity_w{int(fraction * 100)}.pth"
                )

        checkpoint_dir = Path(self.config.models_dir)
        for checkpoint_name in checkpoint_names:
            checkpoint_path = checkpoint_dir / checkpoint_name
            if checkpoint_path.exists():
                mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")


    def _process_complexity_windows(self, history: dict, temp_dir: Path, checkpoint_prefix: str) -> None:
        fractions = self.config.complexity_window_fractions or [1.0]

        for frac in fractions:
            max_epoch = math.ceil(frac * self.config.num_epochs)
            window_complexities = history["complexity"][:max_epoch]
            best_epoch = int(np.argmax(window_complexities)) + 1

            src_ckpt = temp_dir / f"epoch_{best_epoch}.pth"

            if frac == 1.0:
                dst_ckpt = Path(self.config.models_dir) / f"{checkpoint_prefix}_max_complexity.pth"
            else:
                frac_label = f"w{int(frac * 100)}"
                dst_ckpt = Path(self.config.models_dir) / f"{checkpoint_prefix}_max_complexity_{frac_label}.pth"

            shutil.copy2(src_ckpt, dst_ckpt)

    @staticmethod
    def anchor_epoch(history: dict[str, list[float]], anchor: str) -> int:
        """1-indexed epoch of the anchor checkpoint (same rule as the main workflow)."""
        if anchor == "max_complexity":
            return int(np.argmax(history["complexity"])) + 1
        if anchor == "min_loss":
            return int(np.argmin(history["loss"])) + 1
        raise ValueError(f"Unknown sweep anchor: {anchor!r}")

    @staticmethod
    def sweep_checkpoint_name(prefix: str, anchor: str, offset: int) -> str:
        sign = "p" if offset > 0 else "m"
        return f"{prefix}_{anchor}_off{sign}{abs(offset)}.pth"

    def _process_neighborhood_sweep(
        self, history: dict[str, list[float]], temp_dir: Path, checkpoint_prefix: str
    ) -> None:
        """Keep only the checkpoints around each anchor epoch (t* + offset).

        Each requested offset fills one slot. If t* + offset falls outside
        [1, T] (e.g. t* = 1, offset -2), the slot is filled with the nearest
        unused valid epoch to t* instead (``shifted=True``), so the neighborhood
        always contains distinct epochs and no fine-tuning is spent on duplicates.
        The sidecar stores the epoch actually transferred and its true offset
        (``source_epoch - anchor_epoch``); analyses must use those, not the slot.
        """
        models_dir = Path(self.config.models_dir)
        num_epochs = len(history["complexity"])
        for anchor in self.config.sweep_anchors:
            t_star = self.anchor_epoch(history, anchor)
            for slot, epoch in self.neighborhood_epochs(
                t_star, self.config.sweep_offsets or [], num_epochs
            ):
                dst = models_dir / self.sweep_checkpoint_name(checkpoint_prefix, anchor, slot)
                shutil.copy2(temp_dir / f"epoch_{epoch}.pth", dst)
                self._write_sidecar(dst, history, anchor, slot, t_star, epoch)

    @staticmethod
    def neighborhood_epochs(
        t_star: int, offsets: list[int], num_epochs: int
    ) -> list[tuple[int, int]]:
        """Map each requested offset (slot) to a distinct source epoch != t*."""
        used = {t_star}
        assigned: dict[int, int] = {}
        for offset in offsets:
            epoch = t_star + offset
            if 1 <= epoch <= num_epochs and epoch not in used:
                assigned[offset] = epoch
                used.add(epoch)
        candidates = sorted(
            (e for e in range(1, num_epochs + 1) if e not in used),
            key=lambda e: (abs(e - t_star), e),
        )
        for offset in sorted(set(offsets) - set(assigned), key=abs):
            if candidates:
                epoch = candidates.pop(0)
            else:  # fewer epochs than slots: reuse the closest valid epoch
                epoch = min(max(t_star + offset, 1), num_epochs)
            assigned[offset] = epoch
        return [(offset, assigned[offset]) for offset in offsets]

    def _write_sidecar(
        self,
        checkpoint_path: Path,
        history: dict[str, list[float]],
        anchor: str,
        offset: int,
        anchor_epoch: int,
        epoch: int,
    ) -> None:
        index = epoch - 1
        sidecar = {
            "anchor": anchor,
            "slot": offset,
            "offset": epoch - anchor_epoch,
            "anchor_epoch": anchor_epoch,
            "source_epoch": epoch,
            "shifted": epoch != anchor_epoch + offset,
            "source_complexity": float(history["complexity"][index]),
            "source_entropy": float(history["entropy"][index]),
            "source_disequilibrium": float(history["disequilibrium"][index]),
            "source_val_loss": float(history["loss"][index]),
            "source_val_accuracy": float(history["accuracy"][index]),
        }
        sidecar_path = checkpoint_path.with_suffix(".json")
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        if mlflow.active_run():
            mlflow.log_artifact(str(sidecar_path), artifact_path="sweep")

    def _transfer_metadata(self) -> dict:
        """For target steps: copy the source-checkpoint sidecar into the results."""
        if not self.config.checkpoint_path:
            return {}
        sidecar_path = Path(self.config.checkpoint_path).with_suffix(".json")
        if not sidecar_path.exists():
            return {}
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return {f"transfer_{key}": value for key, value in sidecar.items()}

    def _save_history(self, history: dict[str, list[float]]) -> None:
        """Persist the per-epoch history (one file per run, not overwritten)."""
        output_path = Path(self.config.output_file)
        run_label = self.config.mlflow_run_name or "run"
        history_path = output_path.with_name(f"{output_path.stem}_{run_label}_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epochs": list(range(1, len(history["loss"]) + 1)),
            "val_accuracy": history["accuracy"],
            "val_loss": history["loss"],
            "complexity": history["complexity"],
            "entropy": history["entropy"],
            "disequilibrium": history["disequilibrium"],
        }
        history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
