"""
PyTorch Dataset and DataLoader utilities for ceol-gpt.

Each example is a single tune encoded as:
    <TYPE:reel> <KEY:Gmajor> <BOS> tok tok ... <EOS>

Long tunes are truncated to max_seq_len. Short tunes are left as-is
(collation handles padding per batch).

Train/val/test split is stratified by tune type so that each split
preserves the type distribution of the full dataset.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.tokenizer import ABCTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TuneDataset(Dataset):
    """Dataset of tokenised Irish tunes for autoregressive language modelling.

    Each item is a 1-D LongTensor of token IDs. The model is trained with a
    standard next-token prediction objective: input = ids[:-1], target = ids[1:].
    """

    def __init__(
        self,
        tunes: list[dict],
        tokenizer: ABCTokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._samples: list[torch.Tensor] = []

        for tune in tunes:
            ids = tokenizer.encode(tune["abc"], tune["type"], tune["mode"])
            # Truncate to max_seq_len
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            self._samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Collate (pad to longest in batch)
# ---------------------------------------------------------------------------

def collate_fn(batch: list[torch.Tensor], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a list of variable-length sequences to (B, T).

    Returns:
        input_ids: (B, T-1) — tokens fed into the model
        targets:   (B, T-1) — next-token labels
        attn_mask: (B, T-1) — 1 for real tokens, 0 for padding
    """
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)  # (B, T)
    input_ids = padded[:, :-1]
    targets = padded[:, 1:]
    attn_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "targets": targets, "attn_mask": attn_mask}


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    tunes: list[dict],
    train_frac: float = 0.85,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split tunes into train/val/test sets, stratified by tune type.

    The test fraction is implicitly 1 - train_frac - val_frac.
    """
    rng = random.Random(seed)

    # Group by type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for tune in tunes:
        by_type[tune["type"]].append(tune)

    train, val, test = [], [], []
    for group in by_type.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    # Shuffle each split so types aren't contiguous
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    data_path: str | Path,
    tokenizer: ABCTokenizer,
    batch_size: int = 64,
    max_seq_len: int = 512,
    train_frac: float = 0.85,
    val_frac: float = 0.10,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load tunes.json, split, build Datasets, return (train, val, test) DataLoaders."""
    with open(data_path) as f:
        tunes = json.load(f)

    train_tunes, val_tunes, test_tunes = stratified_split(
        tunes, train_frac=train_frac, val_frac=val_frac, seed=seed
    )

    pad_id = tokenizer.pad_id

    def _loader(split_tunes: list[dict], shuffle: bool) -> DataLoader:
        ds = TuneDataset(split_tunes, tokenizer, max_seq_len=max_seq_len)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, pad_id),
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader(train_tunes, shuffle=True),
        _loader(val_tunes, shuffle=False),
        _loader(test_tunes, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/tunes.json"

    print(f"Loading {data_path}...")
    with open(data_path) as f:
        tunes = json.load(f)

    print("Building tokenizer...")
    tokenizer = ABCTokenizer.from_tunes(tunes, min_freq=2)
    print(f"Vocab size: {len(tokenizer):,}")

    train_tunes, val_tunes, test_tunes = stratified_split(tunes)
    print(f"Split sizes — train: {len(train_tunes):,}  val: {len(val_tunes):,}  test: {len(test_tunes):,}")

    # Check type distribution is preserved
    def type_dist(split: list[dict]) -> dict[str, float]:
        from collections import Counter
        counts = Counter(t["type"] for t in split)
        total = sum(counts.values())
        return {k: round(v / total, 3) for k, v in sorted(counts.items())}

    print("\nType distribution:")
    print(f"  all:   {type_dist(tunes)}")
    print(f"  train: {type_dist(train_tunes)}")
    print(f"  val:   {type_dist(val_tunes)}")
    print(f"  test:  {type_dist(test_tunes)}")

    train_dl, val_dl, test_dl = make_dataloaders(
        data_path, tokenizer, batch_size=64, max_seq_len=512
    )
    batch = next(iter(train_dl))
    print(f"\nBatch shapes — input_ids: {batch['input_ids'].shape}  targets: {batch['targets'].shape}  attn_mask: {batch['attn_mask'].shape}")
    print(f"Pad fraction in batch: {(batch['input_ids'] == tokenizer.pad_id).float().mean():.3f}")
