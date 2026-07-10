from pathlib import Path
from typing import ClassVar

import torch
from medmnist import BloodMNIST, TissueMNIST
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import v2

from config.dataset_registry import DATASET_REGISTRY

# Fraction of the torchvision training set reserved for train / validation.
# The remainder (10%) is used as-is; only train and val sizes are computed
# explicitly to avoid rounding drift in the split.
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

IMAGE_SIZE = (224, 224)


class DatasetManager:
    """Loads and prepares datasets from MedMNIST and torchvision sources."""

    MEDMNIST_CLASSES: ClassVar[dict] = {
        "BloodMNIST": BloodMNIST,
        "TissueMNIST": TissueMNIST,
    }

    TORCHVISION_CLASSES: ClassVar[dict] = {
        "MNIST": MNIST,
        "FashionMNIST": FashionMNIST,
    }

    def __init__(
        self,
        dataset_name: str,
        data_root: str = "data",
        batch_size_train: int = 128,
        batch_size_eval: int = 64,
    ):
        if dataset_name not in DATASET_REGISTRY:
            valid = list(DATASET_REGISTRY.keys())
            raise ValueError(
                f"Unsupported dataset: {dataset_name!r}. Must be one of {valid}."
            )

        self.dataset_name = dataset_name
        self.data_root = data_root
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval

        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None

    @property
    def _transform(self) -> v2.Compose:
        return v2.Compose(
            [
                v2.Resize(IMAGE_SIZE, antialias=True),
                v2.Grayscale(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    @property
    def n_classes(self) -> int:
        return DATASET_REGISTRY[self.dataset_name].n_classes

    def _make_loader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        """Wrap a dataset in a DataLoader with the manager's standard settings."""
        return DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    def _ensure_dataloaders(self) -> None:
        """Lazily create the train/val/test DataLoaders on first access."""
        if self._train_loader is not None:
            return

        Path(self.data_root).mkdir(parents=True, exist_ok=True)

        if self.dataset_name in self.TORCHVISION_CLASSES:
            self._train_loader, self._val_loader, self._test_loader = (
                self._build_torchvision_datasets()
            )
        else:
            self._train_loader, self._val_loader, self._test_loader = (
                self._build_medmnist_datasets()
            )

    def _build_medmnist_datasets(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build train/val/test DataLoaders for MedMNIST datasets.

        MedMNIST datasets ship with predefined train/val/test splits, so no
        manual splitting is required.
        """
        dataset_class = self.MEDMNIST_CLASSES[self.dataset_name]
        transform = self._transform

        splits = {
            split: dataset_class(
                split=split, download=True, root=self.data_root, transform=transform
            )
            for split in ("train", "val", "test")
        }

        return (
            self._make_loader(splits["train"], self.batch_size_train),
            self._make_loader(splits["val"], self.batch_size_eval),
            self._make_loader(splits["test"], self.batch_size_eval),
        )

    def _build_torchvision_datasets(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build train/val/test DataLoaders for torchvision datasets.

        Torchvision datasets only ship train/test splits, so the train split
        is further divided into train/val using an 80/10/10 ratio overall.
        """
        dataset_class = self.TORCHVISION_CLASSES[self.dataset_name]
        transform = self._transform

        full_train = dataset_class(
            root=self.data_root, train=True, download=True, transform=transform
        )
        test_dataset = dataset_class(
            root=self.data_root, train=False, download=True, transform=transform
        )

        train_size = int(TRAIN_SPLIT * len(full_train))
        val_size = int(VAL_SPLIT * len(full_train))
        remaining = len(full_train) - train_size - val_size

        train_dataset, val_dataset, _ = random_split(
            full_train, [train_size, val_size, remaining]
        )

        return (
            self._make_loader(train_dataset, self.batch_size_train),
            self._make_loader(val_dataset, self.batch_size_eval),
            self._make_loader(test_dataset, self.batch_size_eval),
        )

    @property
    def train_loader(self) -> DataLoader:
        self._ensure_dataloaders()
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        self._ensure_dataloaders()
        return self._val_loader

    @property
    def test_loader(self) -> DataLoader:
        self._ensure_dataloaders()
        return self._test_loader
