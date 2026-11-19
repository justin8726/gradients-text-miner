
import json
import logging
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("gradients-text-miner")


def load_training_data(dataset_path: Path):
    logger.info("Loading training dataset: %s", dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
    )


def load_model_and_tokenizer(model_path: Path):
    logger.info("Loading model: %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


def save_checkpoint(model, tokenizer, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving final checkpoint: %s", output_path)

    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
    )

    tokenizer.save_pretrained(str(output_path))

    logger.info("Checkpoint saved successfully.")


def remaining_time(start_time: float, hours_to_complete: float) -> float:
    allowed_seconds = hours_to_complete * 3600
    elapsed_seconds = time.monotonic() - start_time
    return max(0.0, allowed_seconds - elapsed_seconds)


def load_json_config(dataset_config):
    if isinstance(dataset_config, str):
        return json.loads(dataset_config)

    if isinstance(dataset_config, dict):
        return dataset_config

    raise ValueError("Dataset configuration must be a JSON object.")
