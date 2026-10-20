from collections import defaultdict
from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.experiment_config import ComplexityVariant
from utils.complexity import LMCComplexity


class EpochMetrics(TypedDict, total=False):
    loss: float
    acc: float
    entropy: float
    disequilibrium: float
    complexity: float


class Trainer:
    """Runs per-epoch training and validation for a model."""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        lmc: LMCComplexity | None = None,
        complexity_variant: ComplexityVariant = "adaptive_linear",
    ):
        self.model = model.to(device)
        self.device = device
        self.lmc = lmc or self._build_complexity(complexity_variant)
        self.mean_per_layer = complexity_variant == "mean_per_layer"
        self.exclude_bn_bias = complexity_variant == "no_bn_bias"

    def _build_complexity(self, variant: ComplexityVariant) -> LMCComplexity:
        settings = {
            "fixed_64": {"fixed_num_bins": 64},
            "fixed_256": {"fixed_num_bins": 256},
            "fixed_1024": {"fixed_num_bins": 1024},
            "normalized": {"normalized": True},
            "jensen_shannon": {"divergence": "jensen_shannon"},
            "hard_binning": {"hard_binning": True},
            "quantile_normalization": {"quantile_normalization": True},
        }
        return LMCComplexity(**settings.get(variant, {}))

    def _complexity_metrics(self) -> tuple[float, float, float]:
        batch_norm_parameters = {
            id(parameter)
            for module in self.model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for parameter in module.parameters(recurse=False)
        }
        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if not (
                self.exclude_bn_bias
                and (name.endswith(".bias") or id(parameter) in batch_norm_parameters)
            )
        ]
        if not named_parameters:
            raise ValueError("No parameters remain for complexity computation.")
        if self.mean_per_layer:
            layers: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
            for name, parameter in named_parameters:
                layer_name = name.rsplit(".", maxsplit=1)[0]
                layers[layer_name].append(parameter.detach())
            metrics = [
                self.lmc.compute(torch.nn.utils.parameters_to_vector(parameters))
                for parameters in layers.values()
            ]
            return tuple(
                float(sum(values) / len(values)) for values in zip(*metrics, strict=True)
            )
        flattened = torch.nn.utils.parameters_to_vector(
            [parameter for _, parameter in named_parameters]
        )
        return self.lmc.compute(flattened.detach())

    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> EpochMetrics:
        """Train for one epoch and return loss, accuracy, and LMC complexity metrics.

        Accuracy is returned as a fraction in [0, 1].
        """
        self.model.train()
        loss, correct, total = self._run_epoch(
            dataloader, criterion, optimizer=optimizer, train=True
        )

        entropy, disequilibrium, complexity = self._complexity_metrics()

        return {
            "loss": loss / total,
            "acc": correct / total,
            "entropy": entropy,
            "disequilibrium": disequilibrium,
            "complexity": complexity,
        }

    def validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> EpochMetrics:
        """Validate for one epoch and return loss and accuracy.

        Accuracy is returned as a fraction in [0, 1].
        """
        self.model.eval()
        loss, correct, total = self._run_epoch(dataloader, criterion, optimizer=None, train=False)

        return {
            "loss": loss / total,
            "acc": correct / total,
        }

    def _run_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        train: bool,
    ) -> tuple[float, int, int]:
        """Run a single pass over the dataloader, optionally updating weights.

        Returns:
            A tuple ``(total_loss, correct_predictions, total_samples)``, all
            unnormalized — division by ``total`` is left to the caller.
        """
        running_loss = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in dataloader:
                images_dev = images.to(self.device)
                labels_dev = labels.squeeze().long().to(self.device)

                if train:
                    optimizer.zero_grad()

                outputs = self.model(images_dev)
                loss = criterion(outputs, labels_dev)

                if train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * images_dev.size(0)
                predicted = torch.argmax(outputs, dim=1)
                total += labels_dev.size(0)
                correct += (predicted == labels_dev).sum().item()

        return running_loss, correct, total
