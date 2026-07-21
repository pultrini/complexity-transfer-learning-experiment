from typing import Literal

import torch

OptimizerName = Literal["adam", "adamw", "sgd"]


def build_optmizer(
    name: OptimizerName,
    parameters,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Build an optimizer by name, so experiments can vary it as a controlled variable.

    Args:
        name: Which optimizer to build.
        parameters: Model parameters to optimize (e.g. ``model.parameters()``).
        learning_rate: Learning rate.
        weight_decay: L2 penalty.
        momentum: Momentum coefficient, used only by SGD.

    Raises:
        ValueError: If ``name`` is not a recognized optimizer.
    """
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {name!r}. Must be one of 'adam', 'adamw', 'sgd'.")