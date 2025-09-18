"""Optimizer factory for creating custom optimizers including Adam-mini.

This module provides factory functions to create optimizers based on configuration,
following the exact implementation patterns from official repositories.
"""

import logging
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer

if TYPE_CHECKING:
    from adam_mini import Adam_mini

try:
    from adam_mini import Adam_mini

    ADAM_MINI_AVAILABLE = True
except ImportError:
    ADAM_MINI_AVAILABLE = False
    Adam_mini = None
    logging.warning(
        "adam-mini package not available. Adam-mini" " optimizer will not be supported.",
    )


def create_adam_mini_optimizer(model: Module, cfg: DictConfig) -> "Adam_mini":
    """Create Adam-mini optimizer following GitHub implementation.

    Implementation based on: https://github.com/zyushun/Adam-mini

    Args:
        model: The model to optimize
        cfg: Configuration object containing Adam-mini parameters

    Returns:
        Adam_mini optimizer instance

    Raises:
        ImportError: If adam-mini package is not available
        ValueError: If required model configuration is missing
    """
    if not ADAM_MINI_AVAILABLE:
        raise ImportError(
            "adam-mini package is not available. Install it with: pip install adam-mini",
        )

    # Get model configuration for auto-detection of transformer parameters
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise ValueError("Model must have a 'config' attribute for Adam-mini optimization")

    # Auto-detect or use configured transformer parameters
    dim = cfg.training.adam_mini.dim
    if dim is None:
        dim = getattr(model_config, "hidden_size", None)
        if dim is None:
            raise ValueError(
                "Could not auto-detect model dimension. "
                "Please set training.adam_mini.dim explicitly",
            )
        logging.info(f"Auto-detected model dimension: {dim}")

    n_heads = cfg.training.adam_mini.n_heads
    if n_heads is None:
        n_heads = getattr(model_config, "num_attention_heads", None)
        if n_heads is None:
            raise ValueError(
                "Could not auto-detect number of attention heads. "
                "Please set training.adam_mini.n_heads explicitly",
            )
        logging.info(f"Auto-detected number of attention heads: {n_heads}")

    n_kv_heads = cfg.training.adam_mini.n_kv_heads
    if n_kv_heads is None:
        # Try to auto-detect, but this is optional
        n_kv_heads = getattr(model_config, "num_key_value_heads", None)
        if n_kv_heads is not None:
            logging.info(f"Auto-detected number of key-value heads: {n_kv_heads}")
        else:
            logging.info("No key-value heads specified, using default (same as n_heads)")

    # Create optimizer with GitHub-specified API
    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),
        lr=cfg.training.adam_mini.learning_rate,
        betas=(cfg.training.adam_mini.beta1, cfg.training.adam_mini.beta2),
        eps=cfg.training.adam_mini.eps,
        weight_decay=cfg.training.adam_mini.weight_decay,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )
    optimizer.wqk_names.add("q_proj")  # For Query
    optimizer.wqk_names.add("k_proj")  # For Key
    optimizer.wv_names.add("v_proj")  # For Value
    optimizer.attn_proj_names.add("o_proj")  # For attention output projection
    optimizer.mlp_names.add("up_proj")  # For MLP (up projection)
    optimizer.mlp_names.add("down_proj")  # For MLP (down projection)
    optimizer.mlp_names.add("gate_proj")  # For MLP (gate projection)

    # Apply single lr for values optimization for small training runs
    if cfg.training.adam_mini.use_single_lr_for_values:
        optimizer.wv_names.clear()
        logging.info("Applied single lr for values optimization for small training runs")
    logging.info(
        f"Created Adam-mini optimizer with: lr={cfg.training.adam_mini.learning_rate}, "
        f"weight_decay={cfg.training.adam_mini.weight_decay}, "
        f"betas=({cfg.training.adam_mini.beta1}, {cfg.training.adam_mini.beta2}), "
        f"eps={cfg.training.adam_mini.eps}, dim={dim}, n_heads={n_heads},"
        f" n_kv_heads={n_kv_heads}",
    )

    return optimizer


def create_optimizer(model: Module, cfg: DictConfig) -> Optimizer | None:
    """Create optimizer based on configuration.

    Args:
        model: The model to optimize
        cfg: Configuration object

    Returns:
        Optimizer instance if custom optimizer is enabled, None for default behavior

    Raises:
        ImportError: If required optimizer package is not available
        ValueError: If optimizer configuration is invalid
    """
    # Check if Adam-mini is enabled
    if hasattr(cfg.training, "adam_mini") and cfg.training.adam_mini.enabled:
        return create_adam_mini_optimizer(model, cfg)

    # Return None to use default optimizer behavior
    return None


def get_optimizer_config_updates(cfg: DictConfig) -> dict[str, Any]:
    """Get configuration updates needed for custom optimizers.

    When using custom optimizers, some training configuration parameters
    may need to be adjusted to work properly with the training framework.

    Args:
        cfg: Configuration object

    Returns:
        Dictionary of configuration updates to apply
    """
    config_updates = {}

    # When using custom optimizers, we need to use a standard optimizer name
    # that the training framework recognizes, but we'll override it with our custom optimizer
    if hasattr(cfg.training, "adam_mini") and cfg.training.adam_mini.enabled:
        # Use adamw_torch as fallback since it's widely supported
        config_updates["optim"] = "adamw_torch"
        logging.info("Using adamw_torch as fallback optimizer name for Adam-mini integration")

    return config_updates
