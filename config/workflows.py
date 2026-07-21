from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

Strategy = Literal["normal", "complexity", "min_loss"]


@dataclass
class WorkflowStep:
    """A single step within a transfer learning workflow.

    Attributes:
        name: Identifier for the step (e.g. 'source', 'target_no_transfer').
        dataset_name: Dataset used in this step. Must be a key present in
            ``DATASET_REGISTRY``.
        num_epochs: Number of training epochs for this step.
        strategy: Training strategy to apply.
        checkpoint_path: Path to a checkpoint file for transfer learning.
            ``None`` means the model is trained from scratch.
        output_file: Path where this step's metrics will be saved.
        mlflow_run_name: Name of the corresponding MLflow run.
    """

    name: str
    dataset_name: str
    num_epochs: int
    strategy: Strategy
    checkpoint_path: str | None = None
    output_file: str | None = None
    mlflow_run_name: str | None = None


@dataclass
class Workflow:
    """A complete transfer learning workflow, composed of several steps.

    Attributes:
        name: Identifier for the workflow (e.g. 'medmnist', 'mnist').
        mlflow_experiment_name: Name of the MLflow experiment grouping all
            runs from this workflow.
        source_dataset: Dataset used to pretrain the source model.
        target_dataset: Dataset used to evaluate transfer learning.
        steps: Ordered list of steps that make up the workflow.
    """

    name: str
    mlflow_experiment_name: str
    source_dataset: str
    target_dataset: str
    steps: list[WorkflowStep] = field(default_factory=list)

    DEFAULT_TARGET_EPOCHS: ClassVar[int] = 10

    @classmethod
    def _build_transfer_workflow(
        cls,
        *,
        name: str,
        mlflow_experiment_name: str,
        source: str,
        target: str,
        source_epochs: int,
        metrics_prefix: str,
        models_dir: str,
        metrics_dir: str,
        target_epochs: int | None = None,
        use_windows: bool = False,
    ) -> Workflow:
        target_epochs = target_epochs or cls.DEFAULT_TARGET_EPOCHS

        steps = [
            WorkflowStep(
                name="source",
                dataset_name=source,
                num_epochs=source_epochs,
                strategy="normal",
                checkpoint_path=None,
                output_file=f"{metrics_dir}/{metrics_prefix}_{source}_source.json",
                mlflow_run_name=f"{source}_source_run",
            ),
            WorkflowStep(
                name="target_no_transfer",
                dataset_name=target,
                num_epochs=target_epochs,
                strategy="normal",
                checkpoint_path=None,
                output_file=f"{metrics_dir}/{metrics_prefix}_{target}_no_transfer.json",
                mlflow_run_name=f"{target}_no_transfer_run",
            ),
            WorkflowStep(
                name="target_min_loss",
                dataset_name=target,
                num_epochs=target_epochs,
                strategy="normal",
                checkpoint_path=f"{models_dir}/{source}_min_loss.pth",
                output_file=f"{metrics_dir}/{metrics_prefix}_{target}_min_loss.json",
                mlflow_run_name=f"{target}_min_loss_run",
            ),
            WorkflowStep(
                name="target_max_complexity",
                dataset_name=target,
                num_epochs=target_epochs,
                strategy="normal",
                checkpoint_path=f"{models_dir}/{source}_max_complexity.pth",
                output_file=f"{metrics_dir}/{metrics_prefix}_{target}_max_complexity.json",
                mlflow_run_name=f"{target}_max_complexity_run",
            ),
        ]

        if use_windows:
            for w in [0.2, 0.4, 0.6, 0.8]:
                w_str = f"w{int(w * 100)}"
                steps.append(
                    WorkflowStep(
                        name=f"target_max_complexity_{w_str}",
                        dataset_name=target,
                        num_epochs=target_epochs,
                        strategy="normal",
                        checkpoint_path=f"{models_dir}/{source}_max_complexity_{w_str}.pth",
                        output_file=f"{metrics_dir}/{metrics_prefix}_{target}_max_complexity_{w_str}.json",
                        mlflow_run_name=f"{target}_max_complexity_{w_str}_run",
                    )
                )

        return cls(
            name=name,
            mlflow_experiment_name=mlflow_experiment_name,
            source_dataset=source,
            target_dataset=target,
            steps=steps,
        )

    @classmethod
    def create_medmnist_workflow(
        cls, models_dir: str = "models", metrics_dir: str = "results/metrics", use_windows: bool = False,
    ) -> Workflow:
        """Create the TissueMNIST → BloodMNIST transfer learning workflow."""
        return cls._build_transfer_workflow(
            name="medmnist",
            mlflow_experiment_name="medmnist_transfer",
            source="TissueMNIST",
            target="BloodMNIST",
            source_epochs=50,
            metrics_prefix="medmnist",
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            use_windows=use_windows
        )

    @classmethod
    def create_mnist_workflow(
        cls, models_dir: str = "models", metrics_dir: str = "results/metrics", use_windows: bool = False
    ) -> Workflow:
        """Create the MNIST → FashionMNIST transfer learning workflow."""
        return cls._build_transfer_workflow(
            name="mnist",
            mlflow_experiment_name="mnist_transfer",
            source="MNIST",
            target="FashionMNIST",
            source_epochs=10,
            metrics_prefix="mnist",
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            use_windows=use_windows
        )

    @classmethod
    def create_tinyimagenet_catsdogs_workflow(
        cls, models_dir: str = "models", metrics_dir: str = "results/metrics", use_windows: bool = False
    ) -> Workflow:
        """Create the TinyImageNet -> CatsVsDogs transfer learning workflow."""
        return cls._build_transfer_workflow(
            name="tinyimagenet_catsdogs",
            mlflow_experiment_name="tinyimagenet_catsdogs_transfer",
            source="TinyImageNet",
            target="CatsVsDogs",
            source_epochs=50,
            metrics_prefix="tinyimagenet_catsdogs",
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            use_windows=use_windows
        )


WORKFLOWS = {
    "medmnist": Workflow.create_medmnist_workflow,
    "mnist": Workflow.create_mnist_workflow,
    "tinyimagenet_catsdogs": Workflow.create_tinyimagenet_catsdogs_workflow,
}
