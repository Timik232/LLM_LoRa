"""
Comprehensive configuration validation for LLM LoRa training pipeline.

This module provides centralized validation for all configuration parameters,
ensuring consistency and catching errors early in the training process.
"""

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def validate_model_config(cfg: DictConfig) -> bool:
    logger = logging.getLogger(__name__)

    required_params = ["model_name"]
    missing_params = [p for p in required_params if not hasattr(cfg.model, p)]
    if missing_params:
        logger.error(f"Missing required model parameters: {missing_params}")
        return False

    model_name = getattr(cfg.model, "model_name", None)
    if not model_name or not isinstance(model_name, str):
        logger.error("Model name must be a non-empty string")
        return False

    train_steps = getattr(cfg.model, "train_steps", None)
    if train_steps is not None and train_steps <= 0:
        logger.error("Training steps must be positive")
        return False

    if hasattr(cfg.model, "lora"):
        lora_cfg = cfg.model.lora
        r = getattr(lora_cfg, "r", None)
        if r is not None and r <= 0:
            logger.error("LoRa rank (r) must be positive")
            return False
        alpha = getattr(lora_cfg, "alpha", None)
        if alpha is not None and alpha <= 0:
            logger.error("LoRa alpha must be positive")
            return False
        dropout = getattr(lora_cfg, "dropout", None)
        if dropout is not None and (dropout < 0 or dropout > 1):
            logger.error("LoRa dropout must be between 0 and 1")
            return False

    if hasattr(cfg.model, "quantization"):
        quant_cfg = cfg.model.quantization
        load_4bit = getattr(quant_cfg, "load_in_4bit", False)
        load_8bit = getattr(quant_cfg, "load_in_8bit", False)
        if load_4bit and load_8bit:
            logger.error("Cannot enable both 4-bit and 8-bit quantization")
            return False

    if hasattr(cfg.model, "rkllm"):
        rkllm_cfg = cfg.model.rkllm
        enabled = getattr(rkllm_cfg, "enabled", False)
        if enabled:
            valid_platforms = ["rk3588", "rk3576"]
            target = getattr(rkllm_cfg, "target_platform", None)
            if target is not None and target not in valid_platforms:
                logger.error(
                    "Invalid RKLLM target platform: %s. Valid: %s",
                    target,
                    valid_platforms,
                )
                return False

            valid_quant = ["w8a8", "w4a16", "w4a16_g128"]
            quant = getattr(rkllm_cfg, "quantization", None)
            if quant is not None and quant not in valid_quant:
                logger.error(
                    "Invalid RKLLM quantization: %s. Valid: %s",
                    quant,
                    valid_quant,
                )
                return False

            num_core = getattr(rkllm_cfg, "num_npu_core", None)
            if num_core is not None and not (1 <= num_core <= 3):
                logger.error("RKLLM num_npu_core must be between 1 and 3")
                return False

    logger.info("Model configuration validation passed")
    return True


def validate_training_config(cfg: DictConfig) -> bool:
    logger = logging.getLogger(__name__)

    if not hasattr(cfg, "training"):
        logger.error("Missing training configuration section")
        return False

    training_cfg = cfg.training

    lr = getattr(training_cfg, "learning_rate", None)
    if lr is not None:
        if lr <= 0:
            logger.error("Learning rate must be positive")
            return False
        if lr > 1.0:
            logger.warning(f"Learning rate {lr} is unusually high")

    for step_param in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
    ):
        value = getattr(training_cfg, step_param, None)
        if value is not None and value <= 0:
            logger.error(f"{step_param} must be positive")
            return False

    epochs = getattr(training_cfg, "num_train_epochs", None)
    if epochs is not None and epochs <= 0:
        logger.error("Number of training epochs must be positive")
        return False

    if getattr(training_cfg, "fp16", False) and getattr(training_cfg, "bf16", False):
        logger.error("Cannot enable both fp16 and bf16 precision")
        return False

    for step_param in ["logging_steps", "eval_steps", "save_steps"]:
        value = getattr(training_cfg, step_param, None)
        if value is not None and value <= 0:
            logger.error(f"{step_param} must be positive")
            return False

    logger.info("Training configuration validation passed")
    return True


def validate_paths_config(cfg: DictConfig) -> bool:
    """Validate paths configuration and file existence.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if configuration is valid
    """
    logger = logging.getLogger(__name__)

    if not hasattr(cfg, "paths"):
        logger.error("Missing paths configuration section")
        return False

    paths_cfg = cfg.paths

    required_paths = ["data_dir", "output_dir"]
    for path_param in required_paths:
        if not hasattr(paths_cfg, path_param):
            logger.error(f"Missing required path parameter: {path_param}")
            return False

    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            from hydra.utils import get_original_cwd

            work_dir = get_original_cwd()
        else:
            work_dir = Path.cwd()
    except Exception:
        work_dir = Path.cwd()

    data_dir = Path(work_dir) / paths_cfg.data_dir
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")

    output_dir = Path(work_dir) / paths_cfg.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")
    except Exception as e:
        logger.exception(f"Cannot create output directory {output_dir}: {e}")
        return False

    if hasattr(paths_cfg, "train_data"):
        train_file = data_dir / paths_cfg.train_data
        if not train_file.exists():
            logger.warning(f"Training data file not found: {train_file}")

    logger.info("Paths configuration validation passed")
    return True


def validate_dpo_config(cfg: DictConfig) -> bool:
    """Validate DPO configuration parameters.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if configuration is valid
    """
    logger = logging.getLogger(__name__)

    if not hasattr(cfg, "dpo"):
        logger.info("DPO configuration not found - skipping DPO validation")
        return True

    required_dpo_params = ["val_data", "train_data"]

    missing_params = [p for p in required_dpo_params if not hasattr(cfg.dpo, p)]
    if missing_params:
        logger.error(f"Missing required DPO parameters: {missing_params}")
        return False

    try:
        from hydra.utils import get_original_cwd

        work_dir = Path(get_original_cwd())
    except Exception:
        work_dir = Path.cwd()

    data_dir = Path(work_dir) / cfg.paths.data_dir
    train_file = data_dir / cfg.dpo.train_data
    val_file = data_dir / cfg.dpo.val_data

    if not train_file.exists():
        logger.error(f"DPO training data file not found: {train_file}")
        return False

    if not val_file.exists():
        logger.error(f"DPO validation data file not found: {val_file}")
        return False

    beta = getattr(cfg.dpo, "beta", None)
    if beta is not None and beta <= 0:
        logger.error("DPO beta parameter must be positive")
        return False

    max_length = getattr(cfg.dpo, "max_length", None)
    if max_length is not None and max_length <= 0:
        logger.error("DPO max_length must be positive")
        return False

    logger.info("DPO configuration validation passed")
    return True


def validate_grpo_config(cfg: DictConfig) -> bool:
    """Validate GRPO configuration parameters.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if configuration is valid
    """
    logger = logging.getLogger(__name__)

    if not hasattr(cfg, "grpo"):
        logger.info("GRPO configuration not found - skipping GRPO validation")
        return True

    if not hasattr(cfg, "paths") or cfg.paths is None:
        logger.error("Missing required GRPO parameters or cfg.paths: [paths section missing]")
        return False

    # Check GRPO-specific parameters in cfg.grpo
    grpo_params = ["num_generations"]
    missing_grpo_params = [p for p in grpo_params if not hasattr(cfg.grpo, p)]

    # Check path-based parameters in cfg.paths
    path_params = ["val_data", "train_data", "data_dir"]
    missing_path_params = [p for p in path_params if getattr(cfg.paths, p, None) is None]

    all_missing_params = []
    if missing_grpo_params:
        all_missing_params.extend([f"grpo.{p}" for p in missing_grpo_params])
    if missing_path_params:
        all_missing_params.extend([f"paths.{p}" for p in missing_path_params])

    if all_missing_params:
        logger.error(f"Missing required GRPO parameters or cfg.paths: {all_missing_params}")
        return False

    try:
        from hydra.utils import get_original_cwd

        work_dir = Path(get_original_cwd())
    except Exception:
        work_dir = Path.cwd()

    data_dir = Path(work_dir) / cfg.paths.data_dir
    train_file = data_dir / cfg.paths.train_data
    val_file = data_dir / cfg.paths.val_data

    if not train_file.exists():
        logger.error(f"GRPO training data file not found: {train_file}")
        return False

    if not val_file.exists():
        logger.error(f"GRPO validation data file not found: {val_file}")
        return False

    num_generations = getattr(cfg.grpo, "num_generations", None)
    if num_generations is not None and num_generations <= 0:
        logger.error("GRPO num_generations must be positive")
        return False

    epsilon = getattr(cfg.grpo, "epsilon", None)
    if epsilon is not None and (epsilon <= 0 or epsilon > 1):
        logger.error("GRPO epsilon must be between 0 and 1")
        return False

    temperature = getattr(cfg.grpo, "temperature", None)
    if temperature is not None and temperature <= 0:
        logger.error("GRPO temperature must be positive")
        return False

    logger.info("GRPO configuration validation passed")
    return True


def validate_logging_config(cfg: DictConfig) -> bool:
    """Validate logging configuration parameters.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if configuration is valid
    """
    logger = logging.getLogger(__name__)

    if not hasattr(cfg, "logging"):
        logger.info("Logging configuration not found - using defaults")
        return True

    logging_cfg = cfg.logging

    # Validate logging backend
    if hasattr(logging_cfg, "logging_backend"):
        valid_backends = ["none", "wandb", "mlflow"]
        if logging_cfg.logging_backend not in valid_backends:
            logger.error(
                f"Invalid logging backend: {logging_cfg.logging_backend}. "
                f"Valid: {valid_backends}",
            )
            return False

    # Validate MLflow configuration if used
    if hasattr(logging_cfg, "logging_backend") and logging_cfg.logging_backend == "mlflow":
        if hasattr(logging_cfg, "mlflow"):
            mlflow_cfg = logging_cfg.mlflow
            if hasattr(mlflow_cfg, "tracking_uri") and not mlflow_cfg.tracking_uri:
                logger.warning("MLflow tracking URI is empty")
        else:
            logger.error("MLflow backend selected but no mlflow configuration found")
            return False

    # Validate Wandb configuration if used
    if hasattr(logging_cfg, "logging_backend") and logging_cfg.logging_backend == "wandb":
        if hasattr(cfg, "wandb"):
            wandb_cfg = cfg.wandb
            if not hasattr(wandb_cfg, "project_name") or not wandb_cfg.project_name:
                logger.error("Wandb project name is required")
                return False
        else:
            logger.error("Wandb backend selected but no wandb configuration found")
            return False

    logger.info("Logging configuration validation passed")
    return True


def validate_environment_config(cfg: DictConfig) -> bool:
    logger = logging.getLogger(__name__)

    if hasattr(cfg, "environment") and cfg.environment.get("use_dotenv", False):
        try:
            from dotenv import load_dotenv

            load_dotenv()
            logger.info("Environment variables loaded from .env file")
        except ImportError:
            logger.warning("dotenv package not available - cannot load .env file")

    if hasattr(cfg, "other") and cfg.other.get("hf_login", False):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN environment variable is required for Hugging Face login")
            return False

    if (hasattr(cfg, "logging") and cfg.logging.get("logging_backend") == "wandb") or hasattr(
        cfg,
        "wandb",
    ):
        wandb_token = os.getenv("WANB_API")
        if not wandb_token:
            logger.warning("WANB_API environment variable not set - Wandb logging may fail")

    logger.info("Environment configuration validation passed")
    return True


def validate_complete_config(cfg: DictConfig) -> bool:
    """Run complete configuration validation.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if all validations pass
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive configuration validation...")

    validation_functions = [
        ("Model", validate_model_config),
        ("Training", validate_training_config),
        ("Paths", validate_paths_config),
        ("DPO", validate_dpo_config),
        ("GRPO", validate_grpo_config),
        ("Logging", validate_logging_config),
        ("Environment", validate_environment_config),
    ]

    failed_validations = []

    for name, validation_func in validation_functions:
        try:
            if not validation_func(cfg):
                failed_validations.append(name)
                logger.error(f"{name} configuration validation failed")
        except Exception as e:
            failed_validations.append(name)
            logger.exception(f"{name} configuration validation error: {e}")

    if failed_validations:
        logger.error(f"Configuration validation failed for: {', '.join(failed_validations)}")
        return False

    logger.info("✅ All configuration validations passed successfully!")
    return True


def get_validation_summary(cfg: DictConfig) -> dict[str, Any]:
    """Get a summary of configuration validation results.

    Args:
        cfg: Configuration object

    Returns:
        Dict containing validation results and warnings
    """
    logging.getLogger(__name__)

    summary = {"valid": True, "warnings": [], "errors": [], "sections_validated": []}

    validation_functions = [
        ("model", validate_model_config),
        ("training", validate_training_config),
        ("paths", validate_paths_config),
        ("dpo", validate_dpo_config),
        ("grpo", validate_grpo_config),
        ("logging", validate_logging_config),
        ("environment", validate_environment_config),
    ]

    for section_name, validation_func in validation_functions:
        try:
            result = validation_func(cfg)
            summary["sections_validated"].append(section_name)
            if not result:
                summary["valid"] = False
                summary["errors"].append(f"{section_name} validation failed")
        except Exception as e:
            summary["valid"] = False
            summary["errors"].append(f"{section_name} validation error: {e!s}")

    return summary
