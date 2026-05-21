from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .engine import GameState, PlayerState, encode_state
from .model import DiscardPolicyNet
from .train import train_model


def cmd_train(args: argparse.Namespace):
    train_model(
        dataset_path=args.dataset,
        out_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
    )


def cmd_encode_example(args: argparse.Namespace):
    # Tiny self-check that engine feature encoding works.
    state = GameState(
        wall_remaining=60,
        turn=14,
        players=[
            PlayerState(hand=[0, 0, 1, 2, 3, 3, 9, 10, 11, 20, 21, 22, 33], discards=[5, 6]),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ],
    )
    vec = encode_state(state, 0)
    print(json.dumps({"feature_dim": len(vec), "features": vec[:10]}))


def cmd_predict(args: argparse.Namespace):
    ckpt = torch.load(args.model, map_location="cpu")
    model = DiscardPolicyNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = json.loads(Path(args.features).read_text(encoding="utf-8"))
    x = torch.tensor(data["features"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, k=5)
    result = [
        {"tile": int(idx), "prob": float(prob)}
        for prob, idx in zip(top.values.tolist(), top.indices.tolist())
    ]
    print(json.dumps({"top5_discards": result}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="riichiai")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a discard policy model from JSONL features")
    p_train.add_argument("--dataset", required=True, help="Path to JSONL with features+label")
    p_train.add_argument("--output", required=True, help="Output .pt file")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--val-ratio", type=float, default=0.1)
    p_train.set_defaults(func=cmd_train)

    p_enc = sub.add_parser("encode-example", help="Print example encoded features")
    p_enc.set_defaults(func=cmd_encode_example)

    p_pred = sub.add_parser("predict", help="Predict discard probabilities from a feature file")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--features", required=True, help="JSON file with {features: [...]} ")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
