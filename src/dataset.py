from pathlib import Path
from typing import ClassVar, Literal

import torch
import torchvision
from medmnist import BloodMNIST, TissueMNIST
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import v2
from PIL import Image, UnidentifiedImageError

from config.dataset_registry import DATASET_REGISTRY

# Fraction of the torchvision training set reserved for train / validation.
# The remainder (10%) is used as-is; only train and val sizes are computed
# explicitly to avoid rounding drift in the split.
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

IMAGE_SIZE = (224, 224)


class _TinyImageNetDataset(Dataset):
    """
    Reads the native Tiny ImageNet layout, without reorganizing files.
    
    Train images live at ``train/<wnid>/images/*.JPEG``. The labeled
    validation split lives at ``val/images/*.JPEG`` with labels given in
    ``val/val_annotations.txt`` (tab-separated: filename, wnid, ...).
    """
    
    def __init__(self, root: Path, split: Literal["train", "val"], transform=None) ->None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        
        wnids = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        if split == "train":
            for wnid in wnids:
                img_dir = root / "train" / wnid / "images"
                for img_path in sorted(img_dir.glob("*.JPEG")):
                    self.samples.append((img_path, self.class_to_idx[wnid]))
        else:
            ann_path = root / "val" / "val_annotations.txt"
            img_dir = root / "val" / "images"
            with ann_path.open(encoding="utf-8") as f:
                for line in f:
                    filename, wnid = line.split("\t")[:2]
                    self.samples.append((img_dir / filename, self.class_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


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

        source = DATASET_REGISTRY[self.dataset_name].source
        builder ={
            "torchvision": self._build_torchvision_datasets,
            "medmnist": self._build_medmnist_datasets,
            "imagefolder": self._build_imagefolder_datasets,
        }
        self._train_loader, self._val_loader, self._test_loader = builder[source]()

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
    
    def _build_imagefolder_datasets(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        if self.dataset_name == "TinyImageNet":
            return self._build_tinyimagenet_datasets()
        if self.dataset_name == "CatsVsDogs":
            return self._build_catsvsdogs_datasets()
        raise ValueError(f"no imagefolder loader registred for dataset {self.dataset_name!r}")

    def _build_tinyimagenet_datasets(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Tiny ImageNet: reads the native layout directly (see _TinyImageNetDataset).

        The nativel labeled val/ split is used as our held-out test set; the
        train/ split is further divided into train/val (80/10), matching the
        torchvision branch's ratio.
        """
        root = Path(self.data_root) / "TinyImageNet"
        transform = self._transform

        full_train = _TinyImageNetDataset(root, split="train", transform=transform)
        test_dataset = _TinyImageNetDataset(root, split="val", transform=transform)

        train_size = int(TRAIN_SPLIT * len(full_train))
        val_size = int(VAL_SPLIT * len(full_train))
        remaining = len(full_train) - train_size - val_size

        train_dataset, val_dataset, _ = random_split(full_train, [train_size, val_size, remaining])

        return (
            self._make_loader(train_dataset, self.batch_size_train),
            self._make_loader(val_dataset, self.batch_size_eval),
            self._make_loader(test_dataset, self.batch_size_eval),
        )
    
    def _build_catsvsdogs_datasets(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Cats vs Dogs: expects data_root/CatsVsDogs/train/{cat,dog}/*.jpg
        (see scripts/reorganize_catsvsdogs.py). No native val/test split, so
        we do our own 80/10/10 split over the whole labeled pool.
        """
        root = Path(self.data_root) / "CatVsDogs"
        transform = self._transform

        full_dataset = torchvision.datasets.ImageFolder(
            root=str(root), transform=transform, loader=self._safe_pil_loader
        )
        full_dataset = self._filter_corrupt_samples(full_dataset)

        train_size = int(TRAIN_SPLIT * len(full_dataset))
        val_size = int(VAL_SPLIT * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        return (
            self._make_loader(train_dataset, self.batch_size_train),
            self._make_loader(val_dataset, self.batch_size_eval),
            self._make_loader(test_dataset, self.batch_size_eval),
        )
    @staticmethod
    def _safe_pil_loader(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")
    
    def _filter_corrupt_samples(
        self, dataset: torchvision.datasets.ImageFolder
    ) -> torchvision.datasets.ImageFolder:
        """Drop samples that fail to open as valid_images, in place
        Runs once at dataset-build time (a few thousand small files, so this
        is fast) rather than catching errors lazily inside __getitem__, which
        would surface as a crash deep inside a DataLoader worker.
        """
        valid_samples = []
        for path, label in dataset.samples:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_samples.append((path, label))
            except (UnidentifiedImageError, OSError):
                print(f"  Skipping corrupt image: {path}")

        n_skipped = len(dataset.samples) - len(valid_samples)
        if n_skipped:
            print(f"CatsVsDogs: filtered {n_skipped} corrupt file(s) out of {len(dataset.samples)}.")

        dataset.samples = valid_samples
        dataset.targets = [label for _, label in valid_samples]
        return dataset

    
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
