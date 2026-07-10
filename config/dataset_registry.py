from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata about a suported dataset"""
    source: Literal['medmnist', 'torchvision']
    n_classes: int
    has_val_split: bool
    native_channels: int

DATASET_REGISTRY: dict[str, DatasetInfo] = {
    'BloodMNIST': DatasetInfo(
        source='medmnist',
        n_classes=8,
        has_val_split=True,
        native_channels=3
    ),
    'TissueMNIST': DatasetInfo(
        source='medmnist',
        n_classes=8,
        has_val_split=True,
        native_channels=3
    ),
    'MNIST': DatasetInfo(
        source='torchvision',
        n_classes=10,
        has_val_split=True,
        native_channels=1
    ),
    'FashionMNIST': DatasetInfo(
        source='torchvision',
        n_classes=10,
        has_val_split=True,
        native_channels=1
    ),
    'TinyImageNet': DatasetInfo(
        source='imagefolder',
        n_classes=200,
        has_val_split=True,
        native_channels=3
    ),
    'CatsVsDogs': DatasetInfo(
        source='imagefolder',
        n_classes=2,
        has_val_split=False,
        native_channels=3
    ),
}
