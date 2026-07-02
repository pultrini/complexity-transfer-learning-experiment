from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.complexity import LMCComplexity


class EpochMetrics(TypedDict, total=False):
    loss: float
    acc: float
    entropy: float
    disequilibrium: float
    complexity: float


class Trainer:
    """Runs per-epoch training and validation for a model."""

    def __init__(self, model: nn.Module, device: str, lmc: LMCComplexity | None = None):
        self.model = model.to(device)
        self.device = device
        self.lmc = lmc or LMCComplexity()

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

        params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        params = params.cpu().detach().numpy()
        entropy, disequilibrium, complexity = self.lmc.compute(params)

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
