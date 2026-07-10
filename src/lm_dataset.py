from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config.lm_registry import LM_DATASET_REGISTRY
from src.language_model import TextDataset


class LMDatasetManager:
    """Loads and prepares text datasets for language model training.

    Mirrors ``DatasetManager`` (vision) in shape -- lazy DataLoader creation,
    train/val/test properties -- but is backed by pre-tokenized file splits
    rather than auto-downloaded image datasets.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str = "roberta-base",
        seq_length: int = 32,
        data_root: str = "data",
        batch_size: int = 256,
        max_train_samples: int | None = None,
    ):
        if dataset_name not in LM_DATASET_REGISTRY:
            valid = list(LM_DATASET_REGISTRY.keys())
            raise ValueError(f"Unsupported dataset: {dataset_name!r}. Must be one of {valid}.")

        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.seq_length = seq_length
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.max_train_samples = max_train_samples

        self._tokenizer = None
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def _ensure_dataloaders(self) -> None:
        """Lazily tokenize and wrap the train/val/test splits on first access."""
        if self._train_loader is not None:
            return

        info = LM_DATASET_REGISTRY[self.dataset_name]

        train_dataset = TextDataset(
            str(self.data_root / info.train_file), self.tokenizer, self.seq_length, self.max_train_samples
        )
        val_dataset = TextDataset(str(self.data_root / info.val_file), self.tokenizer, self.seq_length)
        test_dataset = TextDataset(str(self.data_root / info.test_file), self.tokenizer, self.seq_length)

        self._train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self._val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        self._test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True
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
