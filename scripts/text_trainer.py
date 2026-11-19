
#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path


CACHE_DIR = Path("/cache")
CHECKPOINTS_DIR = Path("/app/checkpoints")

SUPPORTED_TASKS = {
    "InstructTextTask": "instruct",
    "DpoTask": "dpo",
    "GrpoTask": "grpo",
    "ChatTask": "chat",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradients Text Tournament Trainer"
    )

    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-type", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--file-format", required=True)
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True, type=float)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task type: {args.task_type}")

    if args.hours_to_complete <= 0:
        raise ValueError("hours-to-complete must be greater than zero")

    try:
        dataset_config = json.loads(args.dataset_type)
    except json.JSONDecodeError as exc:
        raise ValueError("dataset-type must contain valid JSON") from exc

    model_path = CACHE_DIR / "models" / args.model.replace("/", "--")
    dataset_path = CACHE_DIR / "datasets" / f"{args.task_id}_train_data.json"

    output_path = (
        CHECKPOINTS_DIR
        / args.task_id
        / args.expected_repo_name
    )

    print("=== Gradients Text Trainer ===")
    print(f"Task ID:       {args.task_id}")
    print(f"Task type:     {args.task_type}")
    print(f"Model:         {args.model}")
    print(f"Dataset:       {args.dataset}")
    print(f"File format:   {args.file_format}")
    print(f"Model path:    {model_path}")
    print(f"Dataset path:  {dataset_path}")
    print(f"Output path:   {output_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Cached model not found: {model_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Cached dataset not found: {dataset_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    trainer_name = SUPPORTED_TASKS[args.task_type]

    print(f"Dataset configuration loaded: {type(dataset_config).__name__}")
    print(f"Selected trainer: {trainer_name}")

    from src.trainers.dispatcher import train

    train(
        task_type=args.task_type,
        model_path=model_path,
        dataset_path=dataset_path,
        dataset_config=dataset_config,
        output_path=output_path,
        hours_to_complete=args.hours_to_complete,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TRAINING ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
