from dataclasses import dataclass
from typing import ClassVar, Literal

from config.dataset_registry import DATASET_REGISTRY

Strategy = Literal["normal", "complexity", "min_loss"]
CheckpointType = Literal["loss", "complexity"]
ModelArchitecture = Literal["resnet50", "efficientnet_v2_s"]
OptimizerName = Literal["adam", "adamw", "sgd"]

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
        model_architecture: Model architecture to use for training ('resnet50' or 'efficientnet_v2_s').
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
    model_architecture: ModelArchitecture = "resnet50"
    optimizer_name: OptimizerName = "adam"
    learning_rate: float = 0.001
    models_dir: str = "models"
    data_root: str = "data"
    complexity_window_fractions: list[float] | None = None
    mlflow_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_id: int | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    workflow_name: str | None = None
    step_name: str | None = None

    VALID_STRATEGIES: ClassVar[set[Strategy]] = {"normal", "complexity", "min_loss"}
    VALID_ARCHITECTURES: ClassVar[set[ModelArchitecture]] = {"resnet50", "efficientnet_v2_s"}
    VALID_OPTIMIZERS: ClassVar[set[OptimizerName]] = {"adam", "adamw", "sgd"}


    def __post_init__(self) -> None:
        self._validate_strategy()
        self._validate_dataset()
        self._validate_architecture()
        self._validate_optimizer()
        self._validate_complexity_windows()

    def _validate_strategy(self) -> None:
        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {self.strategy!r}. "
                f"Must be one of {sorted(self.VALID_STRATEGIES)}."
            )

    def _validate_dataset(self) -> None:
        if self.dataset_name not in DATASET_REGISTRY:
            valid = sorted(DATASET_REGISTRY.keys())
            raise ValueError(f"Unsupported dataset: {self.dataset_name!r}. Must be one of {valid}.")

    def _validate_architecture(self) -> None:
        if self.model_architecture not in self.VALID_ARCHITECTURES:
            raise ValueError(
                f"Invalid model_architecture: {self.model_architecture!r}. "
                f"Must be one of {sorted(self.VALID_ARCHITECTURES)}."
            )

    def _validate_optimizer(self) -> None:
        if self.optimizer_name not in self.VALID_OPTIMIZERS:
            raise ValueError(
                f"Invalid optimizer_name: {self.optimizer_name!r}. "
                f"Must be one of {sorted(self.VALID_OPTIMIZERS)}."
            )

    def _validate_complexity_windows(self) -> None:
        if self.complexity_window_fractions is None:
            return
        for fraction in self.complexity_window_fractions:
            if not (0 < fraction <= 1):
                raise ValueError(
                    f"complexity_window_fractions must all be in (0, 1], got {fraction!r}."
                )
