from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LogDataset(Dataset):
    """Dataset from JSONL rows: {"features": [...], "label": int}."""

    def __init__(self, path: str | Path):
        path = Path(path)
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        if not rows:
            raise ValueError(f"No rows found in {path}")

        self.features = np.asarray([r["features"] for r in rows], dtype=np.float32)
        self.labels = np.asarray([r["label"] for r in rows], dtype=np.int64)

        if self.features.ndim != 2:
            raise ValueError("features must be a 2D array")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
