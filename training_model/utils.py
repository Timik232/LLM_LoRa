"""
Module with utility functions for training the model.
"""

import logging
import os

from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import wandb
from wandb.sdk.wandb_run import Run

from training_model.exceptions import ConfigurationError


def get_generation_config(
    cfg: DictConfig, tokenizer: AutoTokenizer | PreTrainedTokenizerBase, method: str = "global"
) -> dict:
    """Get generation configuration from config with method-specific overrides.

    Currently used for GRPO training and general evaluation/testing purposes.
    SFT and DPO don't need generation during training.

    Args:
        cfg: Configuration object.
        tokenizer: Tokenizer for setting pad_token_id.
        method: Training method ("global", "grpo").

    Returns:
        Dictionary with generation parameters.

    Raises:
        ConfigurationError: If generation configuration is invalid.
    """
    try:
        # Start with global defaults
        generation_config = {
            "do_sample": getattr(cfg.generation, "do_sample", True),
            "temperature": getattr(cfg.generation, "temperature", 0.7),
            "top_k": getattr(cfg.generation, "top_k", 50),
            "top_p": getattr(cfg.generation, "top_p", 0.95),
            "max_new_tokens": getattr(cfg.generation, "max_new_tokens", 256),
            "repetition_penalty": getattr(cfg.generation, "repetition_penalty", 1.1),
            "pad_token_id": (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        }

        # Apply GRPO-specific overrides for backward compatibility
        if method == "grpo" and hasattr(cfg, "grpo"):
            grpo_config = {
                "do_sample": getattr(cfg.grpo, "do_sample", generation_config["do_sample"]),
                "temperature": getattr(
                    cfg.grpo, "temperature", generation_config["temperature"]
                ),
                "top_k": getattr(cfg.grpo, "top_k", generation_config["top_k"]),
                "top_p": getattr(cfg.grpo, "top_p", generation_config["top_p"]),
                "max_new_tokens": getattr(
                    cfg.grpo, "response_length", generation_config["max_new_tokens"]
                ),
                "repetition_penalty": generation_config["repetition_penalty"],
                "pad_token_id": generation_config["pad_token_id"],
            }
            generation_config.update(grpo_config)

        # Remove None values and ensure pad_token_id is set
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        # Ensure pad_token_id is always set (required for generation)
        if (
            "pad_token_id" not in generation_config
            or generation_config.get("pad_token_id") is None
        ):
            # Try multiple fallback options
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            if pad_id is None:
                pad_id = 0
            generation_config["pad_token_id"] = pad_id
            logging.info(f"Set pad_token_id to {pad_id}")

        logging.info(f"Generation config for {method}: {generation_config}")
        return generation_config

    except Exception as e:
        raise ConfigurationError(
            f"Failed to create generation config for {method}: {e}"
        ) from e


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
    import logging

    logger = logging.getLogger(__name__)

    # Load environment variables if configured to do so
    _load_environment_if_needed(cfg)

    if cfg.other.hf_login:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            raise OSError("Environment variable 'HF_TOKEN' is not set.")
        login(token=hf_token)

    # Weights & Biases login
    wb_token = os.getenv("WANB_API")
    if wb_token is None:
        raise OSError("Environment variable 'WANB_API' is not set.")
    # wandb.login(key=wb_token)

    logger.debug(
        f"Using Weights & Biases token: {wb_token[:8]}..." if wb_token else "No token found",
    )
    run = wandb.init(
        project=cfg.wandb.project_name,
        job_type="training",
        config=OmegaConf.to_container(cfg, resolve=True),
        anonymous=cfg.wandb.anonymous,
    )
    return run
