"""The four public pipeline commands."""

import argparse
import json
from pathlib import Path

from robot_policy_lab.data import InvalidEpisode, generate_data, validate_dataset
from robot_policy_lab.evaluate import evaluate
from robot_policy_lab.policy import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless robot policy lab")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-data")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--episodes", type=int, default=128)
    generate.add_argument("--start-seed", type=int, default=0)
    validate = commands.add_parser("validate-data")
    validate.add_argument("dataset", type=Path)
    training = commands.add_parser("train")
    training.add_argument("dataset", type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--chunk-size", type=int, default=4)
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--one-step", type=Path)
    evaluation.add_argument("--chunked", type=Path)
    evaluation.add_argument("--episodes", type=int, default=128)
    evaluation.add_argument("--start-seed", type=int, default=10000)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-data":
            print(generate_data(args.output, args.episodes, args.start_seed))
        elif args.command == "validate-data":
            tables = validate_dataset(args.dataset)
            print(json.dumps(dict(episodes=len(tables), rows=sum(t.num_rows for t in tables))))
        elif args.command == "train":
            result = train(args.dataset, args.output, args.chunk_size, args.epochs, args.seed)
            print(json.dumps({key: value for key, value in result.items() if key != "history"}))
        else:
            result = evaluate(
                args.one_step, args.chunked, args.episodes, args.start_seed, args.output
            )
            print(json.dumps({name: {key: val for key, val in item.items() if key != "outcomes"}
                              for name, item in result["policies"].items()}, indent=2))
    except (InvalidEpisode, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
