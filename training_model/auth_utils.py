"""
Module with utility functions for training the model.
"""
import os
from typing import TYPE_CHECKING

from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb.sdk.wandb_run import Run

if TYPE_CHECKING:
    pass


def _load_environment_if_needed(cfg: DictConfig) -> None:
    """Load environment variables from .env if configured to do so."""
    if cfg.get("environment", {}).get("use_dotenv", False):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass


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
    # Load environment variables if configured to do so
    _load_environment_if_needed(cfg)

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
