
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


def train_chat(
    model_path,
    dataset_path,
    dataset_config,
    output_path,
    hours_to_complete,
):
    start_time = time.monotonic()

    logger.info("Starting Chat training")

    model, tokenizer = load_model_and_tokenizer(model_path)
    dataset = load_training_data(dataset_path)

    if "messages" not in dataset.column_names:
        raise ValueError(
            "Chat dataset must contain a 'messages' column. "
            f"Found columns: {dataset.column_names}"
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
        "Chat training completed in %.2f seconds",
        time.monotonic() - start_time,
    )
