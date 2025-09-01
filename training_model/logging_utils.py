"""MLflow logging utilities for training pipeline."""

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def get_report_to_backend(cfg: DictConfig) -> str:
    """Get the appropriate report_to parameter based on configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        String indicating which backend to report to: "mlflow", "wandb", or "none".
    """
    log_dict = getattr(cfg, "logging", {})
    backend = getattr(log_dict, "logging_backend", None)

    if backend is None:
        backend = "wandb" if getattr(cfg, "wandb", None) else "none"

    backend = str(backend).lower()

    # Map our backend names to HuggingFace trainer expected values
    if backend == "mlflow":
        return "mlflow"
    if backend == "wandb":
        return "wandb"
    return "none"


def log_training_config(cfg: DictConfig) -> None:
    """Log training configuration parameters to MLflow if enabled.

    Args:
        cfg: Hydra configuration object.
    """
    try:
        import mlflow

        # Check if MLflow is active
        if mlflow.active_run() is None:
            return

        # Check if parameters are already logged to prevent conflicts
        run = mlflow.active_run()
        if run is None:
            return

        # Get already logged parameters
        try:
            run_data = mlflow.get_run(run.info.run_id)
            existing_params = set(run_data.data.params.keys())
        except Exception:
            # If we can't get existing params, continue with empty set
            existing_params = set()

        # Log model parameters
        model_params = {
            "base_model": cfg.model.model_name,
            "new_model_name": cfg.model.new_model,
            "torch_dtype": cfg.model.torch_dtype,
            "attn_implementation": cfg.model.attn_implementation,
            "train_steps": cfg.model.train_steps,
            "model_type": cfg.model.model_type,
        }

        # Log LoRA parameters
        lora_params = {
            "lora_r": cfg.model.lora.r,
            "lora_alpha": cfg.model.lora.alpha,
            "lora_dropout": cfg.model.lora.dropout,
        }

        # Log training parameters
        training_params = {
            "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
            "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "num_train_epochs": cfg.training.num_train_epochs,
            "learning_rate": cfg.training.learning_rate,
            "max_seq_length": cfg.training.max_seq_length,
            "fp16": cfg.training.fp16,
            "bf16": cfg.training.bf16,
            "weight_decay": cfg.training.weight_decay,
            "optim": cfg.training.optim,
            "gradient_checkpointing": cfg.training.gradient_checkpointing,
            "warmup_steps": cfg.training.warmup_steps,
            "eval_steps": cfg.training.eval_steps,
            "logging_steps": cfg.training.logging_steps,
            "neftune_noise_alpha": cfg.training.neftune_noise_alpha,
            "seed": cfg.training.seed,
        }

        # Log training methods
        training_methods = {
            "use_sft": cfg.training.use_sft,
            "use_grpo": cfg.training.use_grpo,
            "use_dpo": cfg.training.use_dpo,
        }

        # Log quantization settings
        quant_params = {
            "gguf_conversion_enabled": cfg.model.quant.enabled,
            "quantization_type": cfg.model.quant.qtype,
            "use_8bit": cfg.model.quant.use_8bit,
        }

        # Log RKLLM settings if enabled
        rkllm_params = {}
        if getattr(cfg.model, "rkllm", {}).get("enabled", False):
            rkllm_params = {
                "rkllm_enabled": cfg.model.rkllm.enabled,
                "rkllm_target_platform": cfg.model.rkllm.target_platform,
                "rkllm_quantization": cfg.model.rkllm.quantization,
                "rkllm_do_parallelize": cfg.model.rkllm.do_parallelize,
                "rkllm_hybrid_quantization": cfg.model.rkllm.hybrid_quantization,
                "rkllm_num_npu_core": cfg.model.rkllm.num_npu_core,
            }

        # Log all parameter groups, but filter out already logged parameters
        for params in [
            model_params,
            lora_params,
            training_params,
            training_methods,
            quant_params,
            rkllm_params,
        ]:
            # Filter out parameters that are already logged
            filtered_params = {
                key: value for key, value in params.items() if key not in existing_params
            }

            # Only log if there are new parameters
            if filtered_params:
                mlflow.log_params(filtered_params)
                # Update the set of existing parameters
                existing_params.update(filtered_params.keys())

        logging.info("Logged training configuration to MLflow")

    except Exception as e:
        logging.warning(f"Failed to log configuration to MLflow: {e}")


def log_training_artifacts(cfg: DictConfig, model_path: str | Path, global_steps: int) -> None:
    """Log training artifacts to MLflow if enabled.

    Args:
        cfg: Hydra configuration object.
        model_path: Path to the trained model.
        global_steps: Number of training steps completed.
    """
    try:
        import mlflow

        # Check if MLflow is active
        if mlflow.active_run() is None:
            return

        # Log final model checkpoint
        checkpoint_path = Path(cfg.model.new_model) / f"checkpoint-{global_steps}"
        if checkpoint_path.exists():
            mlflow.log_artifacts(checkpoint_path, "model_checkpoint")
            logging.info(f"Logged model checkpoint to MLflow: {checkpoint_path}")

        # Log merged model if it exists
        if Path(model_path).exists():
            mlflow.log_artifacts(model_path, "merged_model")
            logging.info(f"Logged merged model to MLflow: {model_path}")

        # Log GGUF model if it exists
        if cfg.model.quant.get("enabled", True):
            gguf_path = Path(cfg.paths.final_weights_path) / cfg.model.quant.gguf_dir
            if gguf_path.exists():
                mlflow.log_artifacts(gguf_path, "gguf_model")
                logging.info(f"Logged GGUF model to MLflow: {gguf_path}")

        # Log RKLLM model if it exists
        if getattr(cfg.model, "rkllm", {}).get("enabled", False):
            rkllm_path = Path(cfg.paths.output_dir) / cfg.model.rkllm.output_dir
            if rkllm_path.exists():
                mlflow.log_artifacts(rkllm_path, "rkllm_model")
                logging.info(f"Logged RKLLM model to MLflow: {rkllm_path}")

    except Exception as e:
        logging.warning(f"Failed to log artifacts to MLflow: {e}")


def log_evaluation_metrics(metrics: dict[str, Any]) -> None:
    """Log evaluation metrics to MLflow if enabled.

    Args:
        metrics: Dictionary of metric names to values.
    """
    try:
        import mlflow

        # Check if MLflow is active
        if mlflow.active_run() is None:
            return

        # Filter out non-numeric metrics and log them
        numeric_metrics = {}
        for key, value in metrics.items():
            try:
                # Try to convert to float, skip if not possible
                numeric_value = float(value)
                numeric_metrics[key] = numeric_value
            except (ValueError, TypeError):
                logging.debug(f"Skipping non-numeric metric: {key}={value}")
                continue

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
            logging.info(
                f"Logged evaluation metrics to MLflow: {list(numeric_metrics.keys())}"
            )

    except Exception as e:
        logging.warning(f"Failed to log evaluation metrics to MLflow: {e}")


def validate_mlflow_connection(cfg: DictConfig) -> bool:
    """Validate MLflow connection and configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        True if MLflow is properly configured and accessible, False otherwise.
    """
    try:
        import mlflow

        log_dict = getattr(cfg, "logging", {})
        backend = getattr(log_dict, "logging_backend", "none")

        if backend.lower() != "mlflow":
            return False

        ml_cfg = getattr(log_dict, "mlflow", None)
        if not ml_cfg:
            logging.warning("MLflow backend selected but no mlflow configuration found")
            return False

        tracking_uri = getattr(ml_cfg, "tracking_uri", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Try to list experiments to test connection
        experiments = mlflow.search_experiments()
        logging.info(f"MLflow connection validated. Found {len(experiments)} experiments.")
        return True

    except Exception as e:
        logging.exception(f"MLflow connection validation failed: {e}")
        return False
