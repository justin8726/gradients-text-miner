
import logging
import time

from transformers import TrainingArguments
from trl import SFTTrainer

from src.trainers.common import (
    load_model_and_tokenizer,
    load_training_data,
    save_checkpoint,
)

logger = logging.getLogger("gradients-text-miner")


def train_instruct(
    model_path,
    dataset_path,
    dataset_config,
    output_path,
    hours_to_complete,
):
    start_time = time.monotonic()

    logger.info("Starting Instruct training")

    model, tokenizer = load_model_and_tokenizer(model_path)
    dataset = load_training_data(dataset_path)

    if "text" not in dataset.column_names:
        raise ValueError(
            "Instruct dataset must contain a 'text' column. "
            f"Found columns: {dataset.column_names}"
        )

    remaining_seconds = max(
        60,
        int(hours_to_complete * 3600 - (time.monotonic() - start_time)),
    )

    training_args = TrainingArguments(
        output_dir=str(output_path / "training"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=False,
        bf16=False,
        max_steps=-1,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    save_checkpoint(model, tokenizer, output_path)

    logger.info(
        "Instruct training completed in %.2f seconds",
        time.monotonic() - start_time,
    )
