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
    ) -> Workflow:
        """Build a standard source → target transfer learning workflow.

        Creates four steps: pretraining on the source dataset, training the
        target dataset from scratch, and two transfer learning runs using
        checkpoints selected by 'min_loss' and 'max_complexity' criteria.
        """
        target_epochs = target_epochs or cls.DEFAULT_TARGET_EPOCHS

        return cls(
            name=name,
            mlflow_experiment_name=mlflow_experiment_name,
            source_dataset=source,
            target_dataset=target,
            steps=[
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
            ],
        )

    @classmethod
    def create_medmnist_workflow(
        cls, models_dir: str = "models", metrics_dir: str = "results/metrics"
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
        )

    @classmethod
    def create_mnist_workflow(
        cls, models_dir: str = "models", metrics_dir: str = "results/metrics"
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
        )


WORKFLOWS = {
    "medmnist": Workflow.create_medmnist_workflow,
    "mnist": Workflow.create_mnist_workflow,
}
