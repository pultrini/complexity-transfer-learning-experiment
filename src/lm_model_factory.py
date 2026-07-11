from pathlib import Path
from typing import ClassVar, Literal

import torch
from torch import nn

from src.language_model import TransformerLLM

CheckpointType = Literal["loss", "complexity"]


class LanguageModelFactory:
    """Creates and configures TransformerLLM instances, with checkpoint loading
    for transfer learning between text corpora.

    Mirrors the structure of ``ModelFactory`` (used for vision models) so the
    two follow the same checkpoint-selection conventions, but is kept as a
    separate class since the underlying architecture and construction
    parameters (vocab size, hidden dim, sequence length, ...) are unrelated to
    the vision models.
    """

    CHECKPOINT_MAP: ClassVar[dict[str, str]] = {
        "loss": "min_loss.pth",
        "complexity": "max_complexity.pth",
    }

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_attention_heads: int = 4,
        seq_length: int = 32,
        models_dir: str = "models",
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.models_dir = Path(models_dir)

    def create(
        self,
        checkpoint_type: CheckpointType | None = None,
        checkpoint_path: str | None = None,
    ) -> nn.Module:
        """Create a TransformerLLM, optionally loading a checkpoint.

        Args:
            checkpoint_type: Load a checkpoint from ``models_dir`` selected by
                type ('loss' or 'complexity'). Ignored if ``checkpoint_path``
                is also given.
            checkpoint_path: Explicit checkpoint path, used for transfer
                learning. Takes precedence over ``checkpoint_type``.
        """
        model = TransformerLLM(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            seq_length=self.seq_length,
        )

        if checkpoint_path is not None:
            model = self._load_checkpoint_from_path(model, Path(checkpoint_path))
        elif checkpoint_type is not None:
            model = self._load_checkpoint(model, checkpoint_type)

        return model

    def _load_checkpoint(self, model: nn.Module, checkpoint_type: CheckpointType) -> nn.Module:
        """Load weights from a checkpoint selected by type ('loss' or 'complexity')."""
        if checkpoint_type not in self.CHECKPOINT_MAP:
            valid = list(self.CHECKPOINT_MAP.keys())
            raise ValueError(f"Invalid checkpoint_type: {checkpoint_type!r}. Must be one of {valid}.")

        checkpoint_path = self.models_dir / self.CHECKPOINT_MAP[checkpoint_type]

        if not checkpoint_path.exists():
            print(f"Warning: checkpoint '{checkpoint_path}' not found. Using random weights.")
            return model

        state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict)
        return model

    def _load_checkpoint_from_path(self, model: nn.Module, checkpoint_path: Path) -> nn.Module:
        """Load a checkpoint from an explicit path for transfer learning.

        Uses ``strict=False`` since the embedding and lm_head layers may
        differ in size if source and target corpora use different tokenizers
        or vocabularies; only compatible layers are transferred.
        """
        if not checkpoint_path.exists():
            print(f"Warning: checkpoint '{checkpoint_path}' not found. Using random weights.")
            return model

        print(f"Loading checkpoint for transfer learning: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=True)
        incompatible = model.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print(f"  Layers missing from checkpoint (randomly initialized): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  Unused checkpoint layers: {incompatible.unexpected_keys}")

        return model
