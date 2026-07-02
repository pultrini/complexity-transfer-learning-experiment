from dataclasses import dataclass
from typing import ClassVar, Literal

from config.dataset_registry import DATASET_REGISTRY

Strategy = Literal["normal", "complexity", "min_loss"]
CheckpointType = Literal["loss", "complexity"]

@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment.

    Attributes:
        dataset_name: Name of the dataset to use. Must be a key present in
            ``DATASET_REGISTRY`` (e.g. 'BloodMNIST', 'TissueMNIST', 'MNIST',
            'FashionMNIST').
        num_epochs: Number of training epochs.
        device: Torch device identifier (e.g. 'cuda', 'cpu').
        output_file: Path where the experiment metrics will be saved.
        strategy: Training strategy to apply. One of 'normal', 'complexity',
            or 'min_loss'.
        checkpoint_type: Type of checkpoint to load for transfer learning
            ('loss' or 'complexity'), or None to train from scratch.
        checkpoint_path: Explicit path to a checkpoint file, overriding
            ``checkpoint_type`` resolution when set.
        models_dir: Directory where model checkpoints are stored.
        data_root: Root directory containing the datasets.
        mlflow_uri: URI of the MLflow tracking server.
        mlflow_experiment_id: MLflow experiment ID, if already known.
        mlflow_experiment_name: MLflow experiment name (used in workflows).
        mlflow_run_name: MLflow run name (used in workflows).
        workflow_name: Name of the parent workflow, if this config is part
            of one.
        step_name: Name of the step within the parent workflow.
    """
    dataset_name: str
    num_epochs: int
    device: str
    output_file: str
    strategy: Strategy
    checkpoint_type: CheckpointType | None = None
    checkpoint_path: str | None = None
    models_dir: str = "models"
    data_root: str = "data"
    mlflow_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_id: int | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    workflow_name: str | None = None
    step_name: str | None = None

    VALID_STRATEGIES: ClassVar[set[Strategy]] = {"normal", "complexity", "min_loss"}

    def __post_init__(self) -> None:
        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of {self.VALID_STRATEGIES}.")

    def _validate_dataset(self) -> None:
        if self.dataset_name not in DATASET_REGISTRY:
            valid = sorted(DATASET_REGISTRY.keys())
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name!r}. "
                f"Must be on of {valid}"
            )
