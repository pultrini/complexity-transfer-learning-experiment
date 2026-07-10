from dataclasses import dataclass
from typing import ClassVar, Literal

from config.lm_registry import LM_DATASET_REGISTRY

Strategy = Literal["normal", "complexity", "min_loss"]
CheckpointType = Literal["loss", "complexity"]


@dataclass
class LMExperimentConfig:
    """Configuration for a single language model training experiment."""

    dataset_name: str
    num_epochs: int
    device: str
    output_file: str
    strategy: Strategy
    checkpoint_type: CheckpointType | None = None
    checkpoint_path: str | None = None
    models_dir: str = "models"
    data_root: str = "data"

    tokenizer_name: str = "roberta-base"
    hidden_dim: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    seq_length: int = 32
    batch_size: int = 256
    learning_rate: float = 1e-4
    max_grad_norm: float | None = 1.0
    max_train_samples: int | None = None

    mlflow_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_id: int | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    workflow_name: str | None = None
    step_name: str | None = None

    VALID_STRATEGIES: ClassVar[set[Strategy]] = {"normal", "complexity", "min_loss"}

    def __post_init__(self) -> None:
        self._validate_strategy()
        self._validate_dataset()

    def _validate_strategy(self) -> None:
        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {self.strategy!r}. "
                f"Must be one of {sorted(self.VALID_STRATEGIES)}."
            )

    def _validate_dataset(self) -> None:
        if self.dataset_name not in LM_DATASET_REGISTRY:
            valid = sorted(LM_DATASET_REGISTRY.keys())
            raise ValueError(f"Unsupported dataset: {self.dataset_name!r}. Must be one of {valid}.")
