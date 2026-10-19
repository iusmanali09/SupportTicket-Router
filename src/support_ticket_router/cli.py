"""Command-line interface for training, evaluation, and interactive predictions."""

import argparse
import json
from pathlib import Path
import sys

from .data import load_dataset, split_dataset
from .model import evaluate, load_model, predict, save_model, train

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="support-ticket-router")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Train model and evaluate on held-out test data")
    fit.add_argument("--data", type=Path, default=ROOT / "data/raw/tickets.csv")
    fit.add_argument("--model", type=Path, default=ROOT / "models/router.joblib")
    fit.add_argument("--seed", type=int, default=42)
    infer = commands.add_parser("predict", help="Predict a ticket route")
    infer.add_argument("--model", type=Path, default=ROOT / "models/router.joblib")
    infer.add_argument("--subject", required=True)
    infer.add_argument("--description", required=True)
    demo = commands.add_parser("demo", help="Read ticket subject and description interactively")
    demo.add_argument("--model", type=Path, default=ROOT / "models/router.joblib")
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            texts, labels = load_dataset(args.data)
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = split_dataset(texts, labels, args.seed)
            model = train(x_train, y_train, args.seed)
            validation = evaluate(model, x_valid, y_valid)
            model = train(x_train + x_valid, y_train + y_valid, args.seed)
            test = evaluate(model, x_test, y_test)
            save_model(model, args.model)
            print(json.dumps({"dataset_rows": len(texts), "split_sizes": {"train": len(x_train), "validation": len(x_valid), "test": len(x_test)}, "validation": validation, "test": test, "model_path": str(args.model)}, indent=2))
        elif args.command == "predict":
            print(json.dumps(predict(load_model(args.model), args.subject, args.description), indent=2))
        else:
            model = load_model(args.model)
            subject = input("Subject: ")
            description = input("Description: ")
            print(json.dumps(predict(model, subject, description), indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
