"""
Module with utility functions for training the model.
"""
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb.sdk.wandb_run import Run

load_dotenv()


def tokens_init(cfg: DictConfig) -> Run:
    """
    Initialize Weights & Biases logging
    and configure authentication using environment variables.

    This function reads Hugging Face and Wandb tokens from environment variables:
        - HF_TOKEN for Hugging Face
        - WANDB_API_KEY for Weights & Biases

    Args:
        cfg (DictConfig): Configuration object with training parameters.

    Returns:
        Run: Initialized Weights & Biases run object.
    """
    if cfg.other.hf_login:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            raise EnvironmentError("Environment variable 'HF_TOKEN' is not set.")
        login(token=hf_token)

    # Weights & Biases login
    wb_token = os.getenv("WANB_API")
    if wb_token is None:
        raise EnvironmentError("Environment variable 'WANB_API' is not set.")
    # wandb.login(key=wb_token)

    print(wb_token)
    run = wandb.init(
        project=cfg.wandb.project_name,
        job_type="training",
        config=OmegaConf.to_container(cfg, resolve=True),
        anonymous=cfg.wandb.anonymous,
    )
    return run


def dataset_to_json(dataset: Dict[str, Any], filename: str) -> List[Dict[str, str]]:
    """
    Convert dataset to JSON lines format and save to a file.

    Args:
        dataset (Dict[str, Any]): Source dataset containing:
            - 'system': System prompt template
            - 'examples': Dictionary of conversation examples
        filename (str): Output file path where JSON lines are written.

    Returns:
        List[Dict[str, str]]: List of JSON objects representing each example.
    """
    json_objects: List[Dict[str, str]] = []
    system_template = dataset.get("system", "")
    examples = dataset.get("examples", {})

    # Initialize (or clear) the output file
    with open(filename, "w", encoding="utf-8"):
        pass

    for _, example in examples.items():
        system_message = system_template
        # Import here to avoid circular import
        from evaluation.model_evaluation import get_user_prompt

        user_message = get_user_prompt(example.get("prompt", {}))
        bot_message = str(example.get("answer", ""))

        json_object = {
            "system": system_message,
            "user": user_message,
            "bot": bot_message,
        }
        json_objects.append(json_object)

        # Append JSON object per line
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_object, ensure_ascii=False) + "\n")

    return json_objects
