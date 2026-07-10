from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.complexity import LMCComplexity


class LMEpochMetrics(TypedDict, total=False):
    loss: float
    entropy: float
    disequilibrium: float
    complexity: float


class LMTrainer:
    """Runs per-epoch training and validation for a causal language model.

    Trains with plain cross-entropy loss only -- LMC complexity is computed
    and logged as an observational metric for checkpoint selection, exactly
    as in the vision ``Trainer``, but never enters the training objective
    itself (unlike the original script this was adapted from, which blended
    a complexity term into the loss).
    """

    def __init__(self, model: nn.Module, device: str, lmc: LMCComplexity | None = None):
        self.model = model.to(device)
        self.device = device
        self.lmc = lmc or LMCComplexity()

    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        vocab_size: int,
        max_grad_norm: float | None = 1.0,
    ) -> LMEpochMetrics:
        """Train for one epoch and return loss and LMC complexity metrics."""
        self.model.train()
        total_loss, total_batches = self._run_epoch(
            dataloader, criterion, vocab_size, optimizer=optimizer, max_grad_norm=max_grad_norm, train=True
        )

        weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
        entropy, disequilibrium, complexity = self.lmc.compute(weights)

        return {
            "loss": total_loss / total_batches,
            "entropy": entropy,
            "disequilibrium": disequilibrium,
            "complexity": complexity,
        }

    def validate_epoch(
        self, dataloader: DataLoader, criterion: nn.Module, vocab_size: int
    ) -> LMEpochMetrics:
        """Validate for one epoch and return the average loss."""
        self.model.eval()
        total_loss, total_batches = self._run_epoch(
            dataloader, criterion, vocab_size, optimizer=None, max_grad_norm=None, train=False
        )
        return {"loss": total_loss / total_batches}

    def _run_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        vocab_size: int,
        optimizer: torch.optim.Optimizer | None,
        max_grad_norm: float | None,
        train: bool,
    ) -> tuple[float, int]:
        """Run a single pass over the dataloader, optionally updating weights.

        Returns:
            A tuple ``(total_loss, num_batches)`` -- unnormalized; averaging
            is left to the caller.
        """
        total_loss = 0.0
        total_batches = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                if train:
                    optimizer.zero_grad(set_to_none=True)

                logits = self.model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

                if train:
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                total_loss += loss.detach().item()
                total_batches += 1

        return total_loss, total_batches
