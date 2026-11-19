
import logging
import time

from trl import GRPOConfig, GRPOTrainer

from src.trainers.common import (
    load_model_and_tokenizer,
    load_training_data,
    save_checkpoint,
)

logger = logging.getLogger("gradients-text-miner")


def train_grpo(
    model_path,
    dataset_path,
    dataset_config,
    output_path,
    hours_to_complete,
):
    start_time = time.monotonic()

    logger.info("Starting GRPO training")

    model, tokenizer = load_model_and_tokenizer(model_path)
    dataset = load_training_data(dataset_path)

    if "prompt" not in dataset.column_names:
        raise ValueError(
            "GRPO dataset must contain a 'prompt' column. "
            f"Found columns: {dataset.column_names}"
        )

    def reward_function(completions, **kwargs):
        rewards = []
        for completion in completions:
            if isinstance(completion, list):
                text = " ".join(
                    item.get("content", "")
                    for item in completion
                    if isinstance(item, dict)
                )
            else:
                text = str(completion)

            rewards.append(1.0 if text.strip() else 0.0)

        return rewards

    training_args = GRPOConfig(
        output_dir=str(output_path / "training"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_completion_length=256,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    save_checkpoint(model, tokenizer, output_path)

    logger.info(
        "GRPO training completed in %.2f seconds",
        time.monotonic() - start_time,
    )
