"""
Memory management utilities for efficient training pipeline cleanup.
"""

import gc
import logging
import os
from typing import Any

import psutil
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage statistics."""
    memory_stats = {}

    # System RAM
    process = psutil.Process(os.getpid())
    memory_stats["ram_mb"] = process.memory_info().rss / 1024 / 1024

    # GPU memory if available
    if torch.cuda.is_available():
        memory_stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_stats["gpu_free_mb"] = (
            (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())
            / 1024
            / 1024
        )

    return memory_stats


def log_memory_usage(
    prefix: str = "", log_level: int = logging.INFO, cfg: OmegaConf | None = None
) -> None:
    """Log current memory usage with optional prefix."""
    # Check config to see if logging is enabled
    if (
        cfg
        and hasattr(cfg, "memory_management")
        and not getattr(cfg.memory_management, "log_memory_usage", True)
    ):
        return

    stats = get_memory_usage()
    msg = f"{prefix}Memory usage - RAM: {stats['ram_mb']:.1f}MB"

    if "gpu_allocated_mb" in stats:
        msg += f", GPU allocated: {stats['gpu_allocated_mb']:.1f}MB"
        msg += f", GPU reserved: {stats['gpu_reserved_mb']:.1f}MB"
        msg += f", GPU free: {stats['gpu_free_mb']:.1f}MB"

    logger.log(log_level, msg)


def comprehensive_memory_cleanup(
    aggressive: bool = False, cfg: OmegaConf | None = None
) -> None:
    """
    Perform comprehensive memory cleanup.

    Args:
        aggressive: If True, performs more thorough cleanup that may impact performance
        cfg: Configuration object with memory management settings
    """
    # Use config to determine aggressiveness if provided
    if cfg and hasattr(cfg, "memory_management"):
        aggressive = getattr(cfg.memory_management, "aggressive_cleanup", aggressive)

    logger.debug("Starting memory cleanup")

    # Standard cleanup
    gc.collect()

    if torch.cuda.is_available():
        # Clear CUDA cache
        torch.cuda.empty_cache()

        if aggressive:
            # More aggressive CUDA cleanup
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()

    if aggressive:
        # Force multiple garbage collection passes
        for _ in range(3):
            gc.collect()

    logger.debug("Memory cleanup completed")


def cleanup_model(
    model: Any | None, model_name: str = "model", cfg: OmegaConf | None = None
) -> None:
    """
    Safely cleanup a model and free its memory.

    Args:
        model: The model to cleanup (can be None)
        model_name: Name for logging purposes
        cfg: Configuration object with memory management settings
    """
    if model is not None:
        logger.debug(f"Cleaning up {model_name}")

        # Move to CPU if on CUDA
        if hasattr(model, "cpu"):
            try:
                model.cpu()
            except Exception as e:
                logger.warning(f"Failed to move {model_name} to CPU: {e}")

        # Delete the model
        del model

        # Cleanup memory with config-aware settings
        comprehensive_memory_cleanup(cfg=cfg)
        logger.debug(f"{model_name} cleanup completed")


def cleanup_tokenizer(tokenizer: Any | None, tokenizer_name: str = "tokenizer") -> None:
    """
    Safely cleanup a tokenizer and free its memory.

    Args:
        tokenizer: The tokenizer to cleanup (can be None)
        tokenizer_name: Name for logging purposes
    """
    if tokenizer is not None:
        logger.debug(f"Cleaning up {tokenizer_name}")
        del tokenizer
        gc.collect()
        logger.debug(f"{tokenizer_name} cleanup completed")


def cleanup_dataset(dataset: Any | None, dataset_name: str = "dataset") -> None:
    """
    Safely cleanup a dataset and free its memory.

    Args:
        dataset: The dataset to cleanup (can be None)
        dataset_name: Name for logging purposes
    """
    if dataset is not None:
        logger.debug(f"Cleaning up {dataset_name}")

        # Clear dataset cache if available
        if hasattr(dataset, "cleanup_cache_files"):
            try:
                dataset.cleanup_cache_files()
            except Exception as e:
                logger.warning(f"Failed to cleanup {dataset_name} cache: {e}")

        del dataset
        gc.collect()
        logger.debug(f"{dataset_name} cleanup completed")


def cleanup_trainer(trainer: Any | None, trainer_name: str = "trainer") -> None:
    """
    Safely cleanup a trainer and free its memory.

    Args:
        trainer: The trainer to cleanup (can be None)
        trainer_name: Name for logging purposes
    """
    if trainer is not None:
        logger.debug(f"Cleaning up {trainer_name}")

        # Cleanup trainer state
        if hasattr(trainer, "model"):
            trainer.model = None
        if hasattr(trainer, "tokenizer"):
            trainer.tokenizer = None
        if hasattr(trainer, "train_dataset"):
            trainer.train_dataset = None
        if hasattr(trainer, "eval_dataset"):
            trainer.eval_dataset = None

        del trainer
        comprehensive_memory_cleanup()
        logger.debug(f"{trainer_name} cleanup completed")
