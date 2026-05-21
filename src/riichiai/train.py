from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import LogDataset
from .model import DiscardPolicyNet


def train_model(
    dataset_path: str,
    out_path: str,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_ratio: float = 0.1,
):
    dataset = LogDataset(dataset_path)
    input_dim = dataset.features.shape[1]

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiscardPolicyNet(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        train_loss = total_loss / max(1, train_size)
        val_acc = correct / max(1, total)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val:
            best_val = val_acc
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "best_val_acc": best_val,
                },
                out_path,
            )

    print(f"Saved best model to {out_path} (val_acc={best_val:.4f})")
