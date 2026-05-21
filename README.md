# riichiai

Starter project for a **riichi mahjong engine + machine learning training pipeline**.

## What this gives you

- A minimal engine-state encoder (`riichiai.engine`) you can replace with your full rules model.
- A PyTorch policy model (`riichiai.model`) that predicts which tile to discard (0-33).
- A training CLI (`riichiai train`) that learns from your extracted game logs.
- A prediction CLI (`riichiai predict`) for quick inference checks.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Log format

You said you'll source game logs. Convert each decision point to a JSONL row:

```json
{"features": [0.1, 0.2, ...], "label": 17}
```

- `features`: encoded state vector (float list)
- `label`: discard tile index (`0..33`)

## Train executable

After `pip install -e .`, the `riichiai` command is available:

```bash
riichiai train --dataset data/train.jsonl --output artifacts/discard_policy.pt --epochs 30
```

## Quick checks

```bash
riichiai encode-example
```

This prints the expected feature dimension and a sample slice.

## Predict from saved model

`features.json`:

```json
{"features": [0.0, 0.2, ...]}
```

Run:

```bash
riichiai predict --model artifacts/discard_policy.pt --features features.json
```

## Next steps you should build

1. Full legal action generation + shanten/ukeire features.
2. Richer labels: not only discard, also riichi/call/pass decisions.
3. Better log converter from Tenhou/Majsoul formats.
4. Self-play loop and value model for expected placement.
