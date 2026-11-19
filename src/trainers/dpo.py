
import logging
import time

from trl import DPOConfig, DPOTrainer

from src.trainers.common import (
    load_model_and_tokenizer,
    load_training_data,
    save_checkpoint,
)

logger = logging.getLogger("gradients-text-miner")


def train_dpo(
    model_path,
    dataset_path,
    dataset_config,
    output_path,
    hours_to_complete,
):
    start_time = time.monotonic()

    logger.info("Starting DPO training")

    model, tokenizer = load_model_and_tokenizer(model_path)
    dataset = load_training_data(dataset_path)

    required_columns = {"prompt", "chosen", "rejected"}
    missing = required_columns - set(dataset.column_names)

    if missing:
        raise ValueError(
            f"DPO dataset is missing columns: {sorted(missing)}. "
            f"Available columns: {dataset.column_names}"
        )

    training_args = DPOConfig(
        output_dir=str(output_path / "training"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    save_checkpoint(model, tokenizer, output_path)

    logger.info(
        "DPO training completed in %.2f seconds",
        time.monotonic() - start_time,
    )
