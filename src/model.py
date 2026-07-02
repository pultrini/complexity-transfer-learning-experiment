from pathlib import Path
from typing import ClassVar

import torch
import torchvision
from torch import nn

CheckpointType = str  # 'loss' or 'complexity' — see CHECKPOINT_MAP


class ModelFactory:
    """Creates and configures ResNet50 models adapted for MedMNIST/torchvision datasets."""

    CHECKPOINT_MAP: ClassVar[dict[str, str]] = {
        "loss": "min_loss.pth",
        "complexity": "max_complexity.pth",
    }

    def __init__(self, dataset_name: str, n_classes: int, models_dir: str = "models"):
        self.dataset_name = dataset_name
        self.n_classes = n_classes
        self.models_dir = Path(models_dir)

    def create(
        self,
        checkpoint_type: CheckpointType | None = None,
        checkpoint_path: str | None = None,
    ) -> nn.Module:
        """Create an adapted ResNet50 model, optionally loading a checkpoint.

        Args:
            checkpoint_type: Load a checkpoint from ``models_dir`` selected by
                type ('loss' or 'complexity'). Ignored if ``checkpoint_path``
                is also given.
            checkpoint_path: Explicit checkpoint path, used for transfer
                learning. Takes precedence over ``checkpoint_type``.
        """
        model = torchvision.models.resnet50(weights=None)
        model = self._adapt_conv1(model)
        model = self._adapt_fc(model)

        if checkpoint_path is not None:
            model = self._load_checkpoint_from_path(model, Path(checkpoint_path))
        elif checkpoint_type is not None:
            model = self._load_checkpoint(model, checkpoint_type)

        return model

    def _adapt_conv1(self, model: nn.Module) -> nn.Module:
        """Adapt the first convolution layer to accept 1 channel (grayscale)."""
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        return model

    def _adapt_fc(self, model: nn.Module) -> nn.Module:
        """Adapt the fully connected layer to the dataset's number of classes."""
        original_num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=original_num_features, out_features=self.n_classes)
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

        Uses ``strict=False`` to allow loading weights across datasets with a
        different number of classes (e.g. TissueMNIST 8 classes → BloodMNIST
        8 classes, or MNIST 10 classes → FashionMNIST 10 classes), since the
        final fully connected layer may not match exactly.
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
