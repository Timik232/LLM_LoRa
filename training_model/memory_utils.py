"""
Memory management utilities for efficient training pipeline cleanup.
"""

import gc
import logging
import os
from typing import Any

import psutil
import torch
from omegaconf import DictConfig

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
    prefix: str = "", log_level: int = logging.INFO, cfg: DictConfig | None = None
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
    aggressive: bool = False, cfg: DictConfig | None = None
) -> None:
    """
    Perform comprehensive memory cleanup.
    Args:
        aggressive: If True, performs more thorough cleanup.
        cfg: Configuration object with memory management settings.
    """
    # Use config to determine aggressiveness if provided
    if cfg and hasattr(cfg, "memory_management"):
        aggressive = getattr(cfg.memory_management, "aggressive_cleanup", aggressive)

    logger.debug("Starting memory cleanup")
    # Standard cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset peak stats for better monitoring
        if aggressive:
            torch.cuda.synchronize()  # Ensure all operations are complete
            torch.cuda.ipc_collect()  # Collect inter-process communication memory
            gc.collect()  # One extra GC pass in aggressive mode (avoid loops)
    logger.debug("Memory cleanup completed")


def cleanup_object(obj: Any, obj_name: str = "object", cfg: DictConfig | None = None) -> None:
    """
    Safely cleanup any object and free its memory.
    Args:
        obj: The object to cleanup (can be None).
        obj_name: Name for logging purposes.
        cfg: Configuration object with memory management settings.
    """
    if obj is None:
        return
    logger.debug(f"Cleaning up {obj_name}")
    try:
        # Move to CPU if it's a torch module/model
        if isinstance(obj, torch.nn.Module) and hasattr(obj, "cpu"):
            obj.cpu()
        # Clear specific attributes if it's a trainer-like object
        if hasattr(obj, "model"):
            obj.model = None
        if hasattr(obj, "tokenizer"):
            obj.tokenizer = None
        if hasattr(obj, "train_dataset"):
            obj.train_dataset = None
        if hasattr(obj, "eval_dataset"):
            obj.eval_dataset = None
        # Cleanup cache if it's a dataset
        if hasattr(obj, "cleanup_cache_files"):
            obj.cleanup_cache_files()
    except Exception as e:
        logger.warning(f"Partial failure during {obj_name} cleanup: {e}")
    finally:
        del obj
        comprehensive_memory_cleanup(cfg=cfg)
    logger.debug(f"{obj_name} cleanup completed")


# Specialized wrappers (for backward compatibility or specific use)
def cleanup_model(
    model: Any | None, model_name: str = "model", cfg: DictConfig | None = None
) -> None:
    cleanup_object(model, model_name, cfg)


def cleanup_tokenizer(
    tokenizer: Any | None, tokenizer_name: str = "tokenizer", cfg: DictConfig | None = None
) -> None:
    cleanup_object(tokenizer, tokenizer_name, cfg)


def cleanup_dataset(
    dataset: Any | None, dataset_name: str = "dataset", cfg: DictConfig | None = None
) -> None:
    cleanup_object(dataset, dataset_name, cfg)


def cleanup_trainer(
    trainer: Any | None, trainer_name: str = "trainer", cfg: DictConfig | None = None
) -> None:
    cleanup_object(trainer, trainer_name, cfg)
