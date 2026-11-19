
from pathlib import Path


def train(
    task_type: str,
    model_path: Path,
    dataset_path: Path,
    dataset_config: dict,
    output_path: Path,
    hours_to_complete: float,
):
    common_args = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "dataset_config": dataset_config,
        "output_path": output_path,
        "hours_to_complete": hours_to_complete,
    }

    if task_type == "InstructTextTask":
        from src.trainers.instruct import train_instruct
        return train_instruct(**common_args)

    elif task_type == "DpoTask":
        from src.trainers.dpo import train_dpo
        return train_dpo(**common_args)

    elif task_type == "GrpoTask":
        from src.trainers.grpo import train_grpo
        return train_grpo(**common_args)

    elif task_type == "ChatTask":
        from src.trainers.chat import train_chat
        return train_chat(**common_args)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")
