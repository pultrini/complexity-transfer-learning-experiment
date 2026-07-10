from pathlib import Path
from typing import ClassVar, Literal

import torch
import torchvision
from torch import nn

CheckpointType = Literal["loss", "complexity"]
ModelArchitecture = Literal["resnet50", "efficientnet_v2_s"]


class ModelFactory:
    """Creates and configures models adapted for MedMNIST/torchvision datasets."""

    CHECKPOINT_MAP: ClassVar[dict[str, str]] = {
        "loss": "min_loss.pth",
        "complexity": "max_complexity.pth",
    }

    def __init__(
        self,
        dataset_name: str,
        n_classes: int,
        models_dir: str = "models",
        architecture: ModelArchitecture = "resnet50",
    ):
        self.dataset_name = dataset_name
        self.n_classes = n_classes
        self.models_dir = Path(models_dir)
        self.architecture = architecture

    def create(
        self,
        checkpoint_type: CheckpointType | None = None,
        checkpoint_path: str | None = None,
    ) -> nn.Module:
        """Create an adapted model for the configured architecture."""
        model = self._build_base_model()

        if checkpoint_path is not None:
            model = self._load_checkpoint_from_path(model, Path(checkpoint_path))
        elif checkpoint_type is not None:
            model = self._load_checkpoint(model, checkpoint_type)

        return model

    def _build_base_model(self) -> nn.Module:
        """Dispatch to the builder matching the configured architecture."""
        builders = {
            "resnet50": self._build_resnet50,
            "efficientnet_v2_s": self._build_efficientnet_v2_s,
        }
        builder = builders.get(self.architecture)
        if builder is None:
            raise ValueError(
                f"Unsupported architecture: {self.architecture!r}. "
                f"Must be one of {list(builders.keys())}."
            )
        return builder()

    def _build_resnet50(self) -> nn.Module:
        """Build a ResNet50 adapted for 1-channel input and n_classes output."""
        model = torchvision.models.resnet50(weights=None)

        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=self.n_classes)
        return model

    def _build_efficientnet_v2_s(self) -> nn.Module:
        """Build an EfficientNetV2-S adapted for 1-channel input and n_classes output."""
        model = torchvision.models.efficientnet_v2_s(weights=None)

        # The first conv layer lives at features[0][0] (a Conv2dNormActivation block)
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        # The classification head is classifier[1] (a Linear layer)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=in_features, out_features=self.n_classes)
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

        Keys whose shape doesn't match the current model (e.g. the final
        classifier layer, when source and target datasets have a different
        number of classes) are skipped so the rest of the pretrained weights
        still load correctly.
        """
        if not checkpoint_path.exists():
            print(f"Warning: checkpoint '{checkpoint_path}' not found. Using random weights.")
            return model

        print(f"Loading checkpoint for transfer learning: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model_state = model.state_dict()

        compatible_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = sorted(set(state_dict) - set(compatible_state))

        incompatible = model.load_state_dict(compatible_state, strict=False)

        if skipped:
            print(f"  Skipped due to shape mismatch (e.g. classifier head): {skipped}")
        if incompatible.missing_keys:
            print(f"  Layers missing from checkpoint (randomly initialized): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  Unused checkpoint layers: {incompatible.unexpected_keys}")

        return model
