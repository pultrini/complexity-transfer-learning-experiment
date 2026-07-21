import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

csv.field_size_limit(sys.maxsize)
class TextDataset(Dataset):
    """Tokenizes a text or CSV file into fixed-length next-token-prediction windows."""

    def __init__(self, file_path: str, tokenizer, seq_length: int, max_samples: int | None = None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        tokenizer_id = getattr(tokenizer, "name_or_path", "default_tokenizer").replace("/", "_")
        cache_suffix = f".{tokenizer_id}_seq{seq_length}"
        if max_samples:
            cache_suffix += f"_max{max_samples}"
        cache_suffix += ".cache.pt"

        cache_path = Path(file_path).with_suffix(cache_suffix)

        if cache_path.exists():
            print(f"⚡ Loading cache dataset: {cache_path.name}")
            self.input_ids = torch.load(cache_path, weights_only=True)
        else:
            print(f"⏳ Processing and tokenizer the data: {Path(file_path).name}")
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                if file_path.endswith(".csv"):
                    reader = csv.DictReader(f)
                    text = "".join(row["text"] for row in reader)
                else:
                    text = f.read()

            encodings = tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False,
                max_length=None,
                verbose=False,
            )
            self.input_ids = encodings["input_ids"][0]

            if max_samples is not None:
                max_length = max_samples * seq_length
                self.input_ids = self.input_ids[:max_length]

            torch.save(self.input_ids, cache_path)
            print(f"✅ Cache salvo em: {cache_path.name}")

    def __len__(self) -> int:
        return max(0, len(self.input_ids) - self.seq_length)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = self.input_ids[idx : idx + self.seq_length]
        target_ids = self.input_ids[idx + 1 : idx + self.seq_length + 1]
        return {"input_ids": input_ids, "labels": target_ids}


class TransformerLLM(nn.Module):
    """A small causal Transformer language model.

    Architecturally a decoder-only stack (via ``nn.TransformerEncoder`` with a
    causal mask, following the same "encoder-as-decoder" pattern used by many
    minimal GPT-style implementations) with learned absolute position
    embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(seq_length, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        embeddings = self.embedding(input_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=input_ids.device), diagonal=1
        ).bool()

        transformer_out = self.transformer(embeddings, mask=causal_mask)
        return self.lm_head(transformer_out)
