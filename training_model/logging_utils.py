"""MLflow logging utilities for training pipeline."""

import logging
from contextlib import contextmanager
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


@contextmanager
def mlflow_phase_run(phase_name: str, enabled: bool = True) -> None:
    """Context manager for MLflow nested runs per training phase.

    This prevents parameter conflicts when different training phases (SFT, GRPO, DPO)
    have different trainer configurations. Each phase gets its own nested run.

    Args:
        phase_name: Name of the training phase (e.g., "sft", "grpo", "dpo")
        enabled: Whether MLflow is enabled (if False, no-op context manager)

    Yields:
        None

    Example:
        >>> with mlflow_phase_run("sft", enabled=True):
        ...     trainer.train()
    """
    if not enabled:
        yield
        return

    try:
        import mlflow

        if mlflow.active_run() is None:
            yield
            return

        mlflow.start_run(nested=True, run_name=phase_name)
        logging.info(f"Started nested MLflow run for {phase_name} phase")

        try:
            yield
        except Exception:
            raise
        finally:
            mlflow.end_run()
            logging.info(f"Ended nested MLflow run for {phase_name} phase")

    except ImportError:
        logging.debug("MLflow not available, skipping nested run")
        yield


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

        # Log training methods
        training_methods = {
            "use_sft": cfg.training.use_sft,
            "use_grpo": cfg.training.use_grpo,
            "use_dpo": cfg.training.use_dpo,
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
            training_methods,
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
                f"Logged evaluation metrics to MLflow: {list(numeric_metrics.keys())}",
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


def log_dataset_samples(
    dataset: list[dict] | Any,
    cfg: DictConfig,
    dataset_name: str = "dataset",
    phase: str = "unknown",
) -> None:
    """Log sample entries from a dataset for inspection before training.

    Args:
        dataset: The dataset to sample from (list of dicts, HuggingFace dataset, etc.)
        cfg: Hydra configuration object containing dataset_logging settings
        dataset_name: Name of the dataset for logging context (e.g., "train", "validation")
        phase: Processing phase (e.g., "raw", "processed")
    """
    try:
        # Check if dataset logging is enabled
        dataset_logging_cfg = getattr(cfg.logging, "dataset_logging", {})
        if not getattr(dataset_logging_cfg, "enabled", False):
            return

        # Get logging configuration parameters
        log_level = getattr(dataset_logging_cfg, "log_level", "INFO").upper()
        num_samples = getattr(dataset_logging_cfg, "num_samples", 5)
        include_metadata = getattr(dataset_logging_cfg, "include_metadata", True)
        truncate_long_text = getattr(dataset_logging_cfg, "truncate_long_text", True)
        max_text_length = getattr(dataset_logging_cfg, "max_text_length", 500)

        # Skip if wrong phase
        log_raw = getattr(dataset_logging_cfg, "log_raw_data", True)
        log_processed = getattr(dataset_logging_cfg, "log_processed_data", True)

        if phase == "raw" and not log_raw:
            return
        if phase == "processed" and not log_processed:
            return

        # Convert log level string to logging constant
        log_level_int = getattr(logging, log_level, logging.INFO)

        # Get dataset size and convert to list if needed
        dataset_size = 0
        dataset_list = []

        # Handle different dataset types
        if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            # HuggingFace dataset or similar
            dataset_size = len(dataset)
            dataset_list = [dataset[i] for i in range(min(num_samples, dataset_size))]
        elif isinstance(dataset, list):
            # List of dictionaries
            dataset_size = len(dataset)
            dataset_list = dataset[:num_samples]
        else:
            # Unknown dataset type
            logging.warning(f"Unknown dataset type for {dataset_name}: {type(dataset)}")
            return

        # Log metadata if enabled
        if include_metadata:
            logging.log(
                log_level_int,
                f"Dataset '{dataset_name}' ({phase}): {dataset_size} total samples, "
                f"showing first {min(num_samples, dataset_size)} samples",
            )

        # Log individual samples
        for i, sample in enumerate(dataset_list):
            # Convert sample to string representation
            if isinstance(sample, dict):
                sample_str = _format_sample_dict(sample, truncate_long_text, max_text_length)
            else:
                sample_str = str(sample)
                if truncate_long_text and len(sample_str) > max_text_length:
                    sample_str = sample_str[:max_text_length] + "..."

            logging.log(
                log_level_int,
                f"Dataset '{dataset_name}' ({phase}) Sample {i+1}:\n{sample_str}",
            )

        logging.log(
            log_level_int,
            f"Finished logging {len(dataset_list)} samples from "
            f"dataset '{dataset_name}' ({phase})",
        )

    except Exception as e:
        logging.warning(f"Failed to log dataset samples for {dataset_name}: {e}")


def _format_sample_dict(
    sample: dict[str, Any], truncate_long_text: bool = True, max_text_length: int = 500
) -> str:
    """Format a sample dictionary for logging display.

    Args:
        sample: Dictionary representing a dataset sample
        truncate_long_text: Whether to truncate long text fields
        max_text_length: Maximum length for text fields when truncating

    Returns:
        Formatted string representation of the sample
    """
    formatted_lines = []

    for key, value in sample.items():
        # Convert value to string
        value_str = str(value)

        # Truncate if needed
        if truncate_long_text and len(value_str) > max_text_length:
            value_str = (
                value_str[:max_text_length]
                + f"... (truncated, original length: {len(str(value))})"
            )

        # Format the key-value pair
        formatted_lines.append(f"  {key}: {value_str}")

    return "{\n" + "\n".join(formatted_lines) + "\n}"
