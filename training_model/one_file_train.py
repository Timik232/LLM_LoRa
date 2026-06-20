"""Main file for model training."""

import contextlib
import functools
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import requests
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, get_peft_model
from requests.auth import HTTPBasicAuth
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer

from .data_preparation import dataset_to_json
from .dpo_train import dpo_train
from .exceptions import (
    ConfigurationError,
    ConversionError,
    DataProcessingError,
    ExperimentTrackingError,
    ModelLoadingError,
    TrainingError,
)
from .grpo_train import grpo_train
from .logging_config import configure_logging
from .logging_utils import (
    get_report_to_backend,
    log_dataset_samples,
    log_evaluation_metrics,
    log_training_artifacts,
    log_training_config,
    mlflow_phase_run,
    validate_mlflow_connection,
)
from .memory_utils import (
    cleanup_dataset,
    cleanup_model,
    cleanup_tokenizer,
    cleanup_trainer,
    comprehensive_memory_cleanup,
    log_memory_usage,
)
from .optimizer_factory import create_optimizer, get_optimizer_config_updates
from .types import ModelType, TrainingResult
from .utils import _load_environment_if_needed

_LOGGING_BACKEND: str | None = None


logger = logging.getLogger(__name__)


def init_logging_backend(cfg: DictConfig) -> None:
    """Initialize experiment logging backend according to configuration.

    Supported backends:
        - "wandb": Uses Weights & Biases (requires `wandb` import).
        - "mlflow": Uses MLflow (optional dependency).
        - "none": No external experiment logging.

    Selection precedence:
        1. If `cfg.logging_backend` is present, its (lowercased) value is used.
        2. Otherwise if a `cfg.wandb` node exists, "wandb" is chosen.
        3. Otherwise defaults to "none".

    This function swallows initialization errors and falls back to "none".

    Args:
        cfg: Hydra config object.

    Returns:
        None
    """
    global _LOGGING_BACKEND
    log_dict = getattr(cfg, "logging", {})
    backend = getattr(log_dict, "logging_backend", None)
    if backend is None:
        backend = "wandb" if getattr(cfg, "wandb", None) else "none"
    backend = str(backend).lower()
    _LOGGING_BACKEND = backend

    if backend == "wandb":
        try:
            wb_cfg = getattr(log_dict, "wandb", None) or {}
            project = getattr(wb_cfg, "project_name", None)
            anonymous = getattr(wb_cfg, "anonymous", None)
            init_kwargs: dict[str, Any] = {}
            if project:
                init_kwargs["project"] = project
            if anonymous is not None:
                init_kwargs["anonymous"] = anonymous
            wandb.init(**init_kwargs)
            logging.info("Initialized wandb logging backend")
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}. Falling back to 'none'.")
            _LOGGING_BACKEND = "none"

    elif backend == "mlflow":
        try:
            import mlflow  # type: ignore[import]

            ml_cfg = getattr(log_dict, "mlflow", None) or {}
            experiment = getattr(ml_cfg, "experiment_name", "default")
            tracking_uri = getattr(ml_cfg, "tracking_uri", None)
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            mlflow.enable_system_metrics_logging()
            mlflow.start_run()
            logging.info("Initialized mlflow logging backend")
        except Exception as e:
            logging.warning(f"Failed to initialize mlflow: {e}. Falling back to 'none'.")
            _LOGGING_BACKEND = "none"

    else:
        logging.info("No external logging backend initialized (using 'none').")


def finish_logging_backend() -> None:
    """Finish/cleanup the selected logging backend (if any).

    Args:
        None

    Returns:
        None
    """
    global _LOGGING_BACKEND
    if _LOGGING_BACKEND == "wandb":
        try:
            wandb.finish()
            logging.info("wandb finished")
        except Exception as e:
            logging.warning(f"wandb.finish() failed: {e}")
    elif _LOGGING_BACKEND == "mlflow":
        try:
            import mlflow  # type: ignore[import]

            mlflow.end_run()
            logging.info("mlflow run ended")
        except Exception as e:
            logging.warning(f"mlflow.end_run() failed: {e}")
    _LOGGING_BACKEND = None


@contextmanager
def change_dir(destination: str) -> Generator[None, None, None]:
    """Context manager for temporarily changing the working directory.

    Args:
        destination: Path to the target directory.

    Yields:
        None: Enters the target directory during context execution.
    """
    current_dir = Path.cwd()
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(str(target_dir))
    try:
        yield
    finally:
        os.chdir(str(current_dir))


def generate_prompt(tokenizer: PreTrainedTokenizerBase, data_point: dict[str, str]) -> str:
    """Generate a chat template prompt for the model.

    Args:
        tokenizer: Hugging Face tokenizer.
        data_point: Dictionary containing system, user and bot messages.

    Returns:
        Formatted chat prompt.
    """
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        res = apply_fn(
            [
                {"role": "system", "content": data_point["system"]},
                {"role": "user", "content": data_point["user"]},
                {"role": "assistant", "content": data_point["bot"]},
            ],
            tokenize=False,
        )
        return str(res)
    # Fallback: join messages into a single prompt string
    parts = [
        data_point.get("system", ""),
        data_point.get("user", ""),
        data_point.get("bot", ""),
    ]
    return "\n".join([p for p in parts if p])


def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    cutoff_len: int,
    prompt: str,
) -> BatchEncoding | dict[str, list]:
    """Tokenize text with specified length constraints.

    Args:
        tokenizer: Hugging Face tokenizer.
        cutoff_len: Maximum sequence length.
        prompt: Text to tokenize.

    Returns:
        Tokenized output dictionary.
    """
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    return result


def generate_and_tokenize_prompt(
    data_point: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    cutoff: int,
    should_add_prompt: bool = False,
) -> dict[str, Any] | BatchEncoding:
    """Generate and tokenize a complete prompt.

    Args:
        data_point: Dictionary containing conversation data.
        tokenizer: Hugging Face tokenizer.
        cutoff: Maximum sequence length.
        should_add_prompt: Used for GRPO, when needed dict with keyword "prompt" returned.

    Returns:
        Tokenized prompt dictionary or a dict containing "prompt" when should_add_prompt=True.
    """

    if should_add_prompt:
        # Enhanced prompt for GRPO training with JSON output instruction
        enhanced_prompt = generate_grpo_prompt(tokenizer, data_point)
        return {"prompt": enhanced_prompt, "correct_answer": data_point["bot"]}
    # Standard SFT prompt
    full_prompt = generate_prompt(tokenizer, data_point)
    tokenized_full_prompt = tokenize(
        tokenizer,
        cutoff,
        full_prompt,
    )
    if (
        tokenized_full_prompt["input_ids"][-1] != tokenizer.eos_token_id
        and len(tokenized_full_prompt["input_ids"]) < cutoff
    ):
        tokenized_full_prompt["input_ids"].append(tokenizer.eos_token_id)
        if "attention_mask" in tokenized_full_prompt:
            tokenized_full_prompt["attention_mask"].append(1)
    return tokenized_full_prompt


def generate_grpo_prompt(
    tokenizer: PreTrainedTokenizerBase,
    data_point: dict[str, str],
) -> str:
    """Generate an enhanced prompt for GRPO training with JSON output instruction.

    Args:
        tokenizer: Hugging Face tokenizer.
        data_point: Dictionary containing system, user and bot messages.

    Returns:
        Enhanced prompt string with JSON instruction.
    """
    # Start with the standard chat template
    messages = [
        {"role": "system", "content": data_point["system"]},
        {"role": "user", "content": data_point["user"]},
    ]

    # Add JSON instruction to the user message
    enhanced_user_content = (
        f"{data_point['user']}\n\n"
        "Please respond with a valid JSON object in the following format: "
        '{"Content": {"Action": "<your_action>"}}. '
        "Choose the most appropriate action based on the context."
    )

    messages[1]["content"] = enhanced_user_content

    # Apply chat template without the assistant response (for GRPO generation)
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        res = apply_fn(messages, tokenize=False, add_generation_prompt=True)
        return str(res)
    # Fallback: join messages into a single prompt string
    return "\n".join([m.get("content", "") for m in messages if m.get("content")])


def _load_json_file(file_path: Path, description: str) -> dict:
    """Load and parse a JSON file with standardized error handling.

    Args:
        file_path: Path to the JSON file to load.
        description: Human-readable description of the file for error messages.

    Returns:
        Loaded JSON data as a dictionary.

    Raises:
        DataProcessingError: If file loading or JSON parsing fails.
    """
    if not file_path.exists():
        raise DataProcessingError(f"{description} file not found: {file_path}")

    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DataProcessingError(f"Invalid JSON format " f"in {file_path}: {e}") from e
    except Exception as e:
        raise DataProcessingError(
            f"Failed to read {description.lower()} from {file_path}: {e}",
        ) from e


def _validate_dataset_structure(data: dict, file_path: Path) -> None:
    """Validate that dataset has required structure with 'system' and 'examples' keys.

    Args:
        data: The loaded dataset dictionary.
        file_path: Path to the file (for error messages).

    Raises:
        DataProcessingError: If dataset structure is invalid.
    """
    if not isinstance(data, dict) or "examples" not in data:
        raise DataProcessingError(
            f"Unrecognized JSON structure in {file_path}. "
            "Expected dict with 'examples' key.",
        )


def _process_auto_split_mode(base: Path, cfg: DictConfig) -> tuple[dict, dict]:
    """Process auto_split mode: load single file and split automatically.

    Args:
        base: Base data directory path.
        cfg: Configuration object.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        DataProcessingError: If data processing fails.
    """
    single_path = base / cfg.paths.train_data
    raw = _load_json_file(single_path, "Training data")
    _validate_dataset_structure(raw, single_path)

    # Split the data automatically
    all_items: list[dict] = raw["examples"]
    try:
        list_train, list_test = train_test_split(
            all_items,
            test_size=cfg.testing.test_split_ratio,
            shuffle=True,
            random_state=cfg.training.seed,
        )
    except Exception as e:
        raise DataProcessingError(f"Failed to split dataset: {e}") from e

    train_dataset = {"system": raw["system"], "examples": list_train}
    test_dataset = {"system": raw["system"], "examples": list_test}
    return train_dataset, test_dataset


def _process_separate_validation_mode(base: Path, cfg: DictConfig) -> tuple[dict, dict]:
    """Process separate_validation mode: load train file + separate validation file.

    Args:
        base: Base data directory path.
        cfg: Configuration object.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        DataProcessingError: If data processing fails.
    """
    # Load training file
    train_path = base / cfg.paths.train_data
    train_raw = _load_json_file(train_path, "Training data")
    _validate_dataset_structure(train_raw, train_path)

    # Load separate validation file
    val_path = base / cfg.testing.val_data_file
    val_raw = _load_json_file(val_path, "Validation data")
    _validate_dataset_structure(val_raw, val_path)

    return train_raw, val_raw


def _process_separate_files_mode(base: Path, cfg: DictConfig) -> tuple[dict, dict]:
    """Process separate_files mode: load separate train and test files.

    Args:
        base: Base data directory path.
        cfg: Configuration object.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        DataProcessingError: If data processing fails.
    """
    train_path = base / cfg.paths.train_data
    test_path = base / cfg.paths.test_data

    # Load both files (no structure validation - more flexible)
    train_dataset = _load_json_file(train_path, "Training file")
    test_dataset = _load_json_file(test_path, "Test file")

    return train_dataset, test_dataset


def data_preparation(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    should_add_prompt: bool = False,
) -> tuple[Dataset, Dataset]:
    """Prepare and preprocess training and validation datasets.

    This function supports three data source modes:
      - "auto_split": Single file with automatic train/test splitting
      - "separate_validation": Single train file + separate validation file
      - "separate_files": Separate train and test files

    Args:
        cfg: Configuration object.
        tokenizer: Hugging Face tokenizer.
        should_add_prompt: If True, returns dict with "prompt" key for grpo training.

    Returns:
        Tuple containing train and validation datasets.

    Raises:
        DataProcessingError: If data loading or preprocessing fails.
        ConfigurationError: If configuration parameters are invalid.
    """
    try:
        # Use current working directory or the configured data directory
        base = Path.cwd() / cfg.paths.data_dir

        # Get data source mode and process accordingly
        data_mode = cfg.testing.data_source_mode

        if data_mode == "auto_split":
            train_dataset, test_dataset = _process_auto_split_mode(base, cfg)
        elif data_mode == "separate_validation":
            train_dataset, test_dataset = _process_separate_validation_mode(base, cfg)
        elif data_mode == "separate_files":
            train_dataset, test_dataset = _process_separate_files_mode(base, cfg)
        else:
            raise ConfigurationError(
                f"Invalid data_source_mode: '{data_mode}'. "
                "Valid options: 'auto_split', 'separate_validation', 'separate_files'",
            )

        # Log raw datasets before processing
        log_dataset_samples(train_dataset, cfg, "train", "raw")
        log_dataset_samples(test_dataset, cfg, "validation", "raw")

        # Use temporary directory for JSON files
        try:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                train_json = str(temp_path / "train.json")
                test_json = str(temp_path / "test.json")

                try:
                    dataset_to_json(train_dataset, train_json, cfg.data_preparation.method)
                    dataset_to_json(test_dataset, test_json, cfg.data_preparation.method)
                except Exception as e:
                    raise DataProcessingError(
                        f"Failed to convert dataset to JSON format: {e}",
                    ) from e

                from datasets import load_dataset

                try:
                    dataset = load_dataset(
                        "json",
                        data_files={"train": train_json, "test": test_json},
                    )

                    tokenize_partial = functools.partial(
                        generate_and_tokenize_prompt,
                        tokenizer=tokenizer,
                        cutoff=cfg.other.cutoff_len,
                        should_add_prompt=should_add_prompt,
                    )

                    ds_train = cast(Any, dataset["train"])
                    ds_test = cast(Any, dataset["test"])

                    # Prefer using the datasets library `.map` when available.
                    # Otherwise, map manually and build a Dataset.
                    if hasattr(ds_train, "map"):
                        train_data = ds_train.map(tokenize_partial)
                    else:
                        # ds_train might be a list of examples
                        train_list = [tokenize_partial(cast(dict, x)) for x in ds_train]
                        train_data = Dataset.from_list(train_list)

                    if hasattr(ds_test, "map"):
                        val_data = ds_test.map(tokenize_partial)
                    else:
                        val_list = [tokenize_partial(cast(dict, x)) for x in ds_test]
                        val_data = Dataset.from_list(val_list)

                    # Log processed datasets after tokenization
                    log_dataset_samples(train_data, cfg, "train", "processed")
                    log_dataset_samples(val_data, cfg, "validation", "processed")

                except Exception as e:
                    raise DataProcessingError(
                        f"Failed to load or tokenize datasets: {e}",
                    ) from e
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Failed during dataset processing: {e}") from e

        return train_data, val_data
    except (DataProcessingError, ConfigurationError):
        raise
    except Exception as e:
        raise DataProcessingError(f"Unexpected error during data preparation: {e}") from e


def model_merge_for_converting(cfg: DictConfig, steps: int, save_path: str) -> None:
    """Merge base model with adapter weights and save the result.

    Args:
        cfg: Configuration object.
        steps: Training step number for checkpoint selection.
        save_path: Path to save merged model.

    Raises:
        ModelLoadingError: If model loading fails.
        ConversionError: If model merging or saving fails.
    """
    try:
        model_path = cfg.model.model_name
        adapter_path = f"{cfg.model.new_model}/checkpoint-{steps}"

        # Load base model
        try:
            if cfg.model.model_type == "gemma":
                base_model = Gemma3ForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype="auto",
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype="auto",
                )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load base model {model_path}: {e}") from e

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            raise ModelLoadingError(f"Failed to load tokenizer for {model_path}: {e}") from e

        # Load and merge adapter
        try:
            peft_model = PeftModel.from_pretrained(cast(Any, base_model), adapter_path)
            merged_model = cast(nn.Module, peft_model.merge_and_unload())  # type: ignore[assignment]
        except Exception as e:
            raise ConversionError(
                f"Failed to load and merge adapter from {adapter_path}: {e}",
            ) from e

        # Cast to target dtype (PEFT adapters are FP32 by default, which causes
        # merge_and_unload to upcast the entire model to FP32 on save)
        try:
            target_dtype = getattr(torch, cfg.model.torch_dtype, torch.bfloat16)
            merged_model = cast(nn.Module, merged_model.to(target_dtype))  # type: ignore[assignment]
            logging.info(f"Merged model cast to {cfg.model.torch_dtype}")
        except Exception as e:
            logging.warning(f"Failed to cast merged model to {cfg.model.torch_dtype}: {e}")

        # Save merged model
        try:
            merged_model.save_pretrained(save_path, safe_serialization=True)  # type: ignore[attr-defined]
            tokenizer.save_pretrained(save_path)
        except Exception as e:
            raise ConversionError(f"Failed to save merged model to {save_path}: {e}") from e
        finally:
            # Clean up resources with comprehensive memory management
            log_memory_usage("Before model merge cleanup: ")
            if "merged_model" in locals():
                cleanup_model(merged_model, "merged_model")
            if "peft_model" in locals():
                cleanup_model(peft_model, "peft_model")
            if "base_model" in locals():
                cleanup_model(base_model, "base_model")
            if "tokenizer" in locals():
                cleanup_tokenizer(tokenizer, "tokenizer")
            log_memory_usage("After model merge cleanup: ")

        logging.info("Model merged")
    except (ModelLoadingError, ConversionError):
        raise
    except Exception as e:
        raise ConversionError(f"Unexpected error during model merging: {e}") from e


def setup_model_and_tokenizer(cfg: DictConfig) -> tuple[ModelType, AutoTokenizer]:
    """Set up model and tokenizer with quantization config.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ModelLoadingError: If model or tokenizer loading fails.
        ConfigurationError: If model configuration is invalid.
    """
    try:
        torch_dtype = (
            getattr(torch, cfg.model.torch_dtype)
            if isinstance(cfg.model.torch_dtype, str)
            else cfg.model.torch_dtype
        )
    except AttributeError as e:
        raise ConfigurationError(
            f"Invalid torch_dtype: {cfg.model.torch_dtype}. "
            f"Must be one of: float16, bfloat16, float32",
        ) from e

    try:
        if cfg.model.quant.enabled:
            if not cfg.model.quant.use_8bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    torch_dtype=torch_dtype,
                )
        else:
            bnb_config = None

        if cfg.model.model_type == "gemma":
            model = Gemma3ForCausalLM.from_pretrained(
                cfg.model.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation=cfg.model.attn_implementation,
                use_cache=False,
            )
        elif cfg.model.model_type == "gemma3n":
            # TODO: Implement Gemma3n support when available
            raise ConfigurationError("Gemma3n model type not yet implemented")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation=cfg.model.attn_implementation,
                use_cache=False,
            )
    except Exception as e:
        raise ModelLoadingError(f"Failed to load model {cfg.model.model_name}: {e}") from e

    logging.info("Model loaded")

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            tokenizer.pad_token = "<|pad|>"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load tokenizer for {cfg.model.model_name}: {e}",
        ) from e

    return model, tokenizer


def attach_lora_adapters(model: ModelType, cfg: DictConfig) -> ModelType:
    """Attach LoRA adapters to a quantized model for fine-tuning.

    This function creates and applies LoRA adapters to a quantized model,
    enabling fine-tuning on quantized weights (which are normally frozen).
    This is essential when running GRPO or other training methods without SFT,
    where the model is loaded in quantized form but needs trainable adapters.

    Args:
        model: The quantized model to attach LoRA adapters to.
        cfg: Configuration object containing LoRA parameters.

    Returns:
        The model with LoRA adapters attached (PeftModel).

    Raises:
        ConfigurationError: If LoRA configuration creation fails.
    """
    try:
        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "up_proj",
                "down_proj",
                "gate_proj",
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
            ],
            inference_mode=False,
        )
        logging.info("LoRA configuration created")

        model = get_peft_model(model, peft_config)
        logging.info("LoRA adapters attached to model")

        # CRITICAL: Ensure use_cache=False is set on the wrapped model
        # This prevents incompatibility with gradient checkpointing
        # Must be done AFTER get_peft_model() to ensure it persists
        if hasattr(model, "config"):
            model.config.use_cache = False
            logging.info("Set model.config.use_cache = False after LoRA attachment")

        if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
            model.base_model.config.use_cache = False
            logging.info("Set model.base_model.config.use_cache = False after LoRA attachment")

        return model
    except Exception as e:
        raise ConfigurationError(f"Failed to attach LoRA adapters: {e}") from e


def run_sft_training(
    model: ModelType,
    tokenizer: AutoTokenizer,
    cfg: DictConfig,
    train_data: Dataset,
    val_data: Dataset,
) -> tuple[int, float, Any]:
    """Run SFT training phase.

    Args:
        model: The model to train.
        tokenizer: The tokenizer.
        cfg: Configuration object.
        train_data: Training dataset.
        val_data: Validation dataset.

    Returns:
        Tuple of (global_steps, eval_loss, trained_model).
        The trained_model is the PeftModel with LoRA adapters attached.

    Raises:
        TrainingError: If SFT training fails.
        ConfigurationError: If training configuration is invalid.
    """
    try:
        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "up_proj",
                "down_proj",
                "gate_proj",
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
            ],
            inference_mode=False,
        )
    except Exception as e:
        raise ConfigurationError(f"Failed to create LoRA configuration: {e}") from e

    logging.info("Starting SFT training phase...")

    # Get the appropriate report_to backend based on configuration
    try:
        report_to_backend = get_report_to_backend(cfg)
    except Exception as e:
        raise ConfigurationError(f"Failed to configure logging backend: {e}") from e

    # Create custom optimizer if adam-mini is enabled
    custom_optimizer = None
    config_updates = {}
    try:
        custom_optimizer = create_optimizer(model, cfg)
        if custom_optimizer is not None:
            config_updates = get_optimizer_config_updates(cfg)
            logging.info("Using custom adam-mini optimizer for SFT training")
    except Exception as e:
        raise ConfigurationError(f"Failed to create custom optimizer: {e}") from e

    try:
        # Apply optimizer config updates if custom optimizer is being used
        optim_setting = config_updates.get("optim", cfg.training.optim)

        sft_config = SFTConfig(
            output_dir=cfg.model.new_model,
            max_length=cfg.training.max_seq_length,
            dataset_kwargs={"skip_prepare_dataset": True},
            packing=False,
            run_name=cfg.model.new_model,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            optim=optim_setting,
            num_train_epochs=cfg.training.num_train_epochs,
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            logging_steps=cfg.training.logging_steps,
            warmup_steps=cfg.training.warmup_steps,
            logging_strategy="steps",
            learning_rate=cfg.training.learning_rate,
            fp16=cfg.training.fp16,
            bf16=cfg.training.bf16,
            weight_decay=cfg.training.weight_decay,
            neftune_noise_alpha=cfg.training.neftune_noise_alpha,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to=report_to_backend,  # Use dynamic backend selection
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=cfg.training.load_best,
        )
        if cfg.optuna.enabled:
            sft_config.run_name = f"{sft_config.run_name}_optuna"
    except Exception as e:
        raise ConfigurationError(f"Failed to create SFT configuration: {e}") from e

    # Run SFT training inside its own try/except/finally so resources are cleaned
    global_steps = 0
    eval_loss = float("nan")
    trainer = None
    trained_model = None

    # Check if MLflow is enabled for nested run management
    mlflow_enabled = report_to_backend == "mlflow"

    try:
        # Create SFTTrainer with custom optimizer if available
        trainer_kwargs = {
            "model": cast(nn.Module, model),
            "train_dataset": train_data,
            "eval_dataset": val_data,
            "peft_config": peft_config,
            "processing_class": cast(PreTrainedTokenizerBase, tokenizer),
            "args": sft_config,
        }

        # Add custom optimizer if adam-mini is enabled
        if custom_optimizer is not None:
            trainer_kwargs["optimizers"] = (custom_optimizer, None)

        trainer = SFTTrainer(**trainer_kwargs)

        # Use nested MLflow run for SFT phase to avoid parameter conflicts
        with mlflow_phase_run("sft", enabled=mlflow_enabled):
            trainer.train()

        global_steps = trainer.state.global_step
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss", float("nan"))
        logging.info(
            f"SFT completed. Global steps: {global_steps}, Evaluation loss: {eval_loss}",
        )

        # Extract the trained PeftModel before cleanup
        trained_model = trainer.model
        logging.info(f"Extracted trained model from SFTTrainer: {type(trained_model)}")
    except Exception as e:
        raise TrainingError(f"SFT training failed: {e}") from e
    finally:
        # Clean up SFT trainer to free memory before next training phase
        log_memory_usage("Before SFT cleanup: ")
        with contextlib.suppress(NameError):
            if trainer is not None:
                # Set trainer.model to None to avoid cleanup destroying our reference
                trainer.model = None
                cleanup_trainer(trainer, "SFT trainer")
        log_memory_usage("After SFT cleanup: ")

    return global_steps, eval_loss, trained_model


def train(cfg: DictConfig) -> TrainingResult:
    """Execute full training pipeline.

    Args:
        cfg: Configuration object.

    Returns:
        A TrainingResult dict with 'global_steps' and 'eval_loss'.

    Raises:
        TrainingError: If any training phase fails.
        ConfigurationError: If training configuration is invalid.
        ModelLoadingError: If model setup fails.
        DataProcessingError: If data preparation fails.
    """
    from .exceptions import ConfigurationError, TrainingError
    from .types import TrainingResult

    try:
        model, tokenizer = setup_model_and_tokenizer(cfg)
        train_data, val_data = data_preparation(cfg, tokenizer)
        logging.info("Data prepared")

        # Initialize training state tracking
        global_steps = 0
        eval_loss = 0.0
        training_completed = False

        # Validate training configuration
        if not cfg.training.use_grpo and not cfg.training.use_sft and not cfg.training.use_dpo:
            raise ConfigurationError(
                "No training method enabled. Please set at least one of: "
                "training.use_sft, training.use_grpo, or training.use_dpo",
            )

        # Phase 1: Supervised Fine-Tuning (SFT)
        if cfg.training.use_sft:
            try:
                global_steps, eval_loss, model = run_sft_training(
                    model,
                    tokenizer,
                    cfg,
                    train_data,
                    val_data,
                )
                training_completed = True

                # Clean up SFT datasets after completion (GRPO/DPO will prepare their own)
                if cfg.training.use_grpo or cfg.training.use_dpo:
                    log_memory_usage("Before SFT dataset cleanup: ", cfg=cfg)
                    cleanup_dataset(train_data, "SFT train dataset")
                    cleanup_dataset(val_data, "SFT val dataset")
                    log_memory_usage("After SFT dataset cleanup: ", cfg=cfg)

                    # Model is now a PeftModel with trained LoRA adapters
                    if isinstance(model, PeftModel):
                        logging.info(
                            "SFT completed with LoRA adapters. "
                            "Subsequent phases will continue training the same adapters."
                        )
                    else:
                        logging.warning(f"Expected PeftModel after SFT, but got {type(model)}")
            except Exception as e:
                raise TrainingError(f"SFT training phase failed: {e}") from e

        # Phase 2: Group Relative Policy Optimization (GRPO)
        if cfg.training.use_grpo:
            logging.info("Starting GRPO training phase...")

            try:
                # If GRPO runs standalone (without SFT), attach LoRA adapters
                # to quantized model
                if not cfg.training.use_sft and cfg.model.quant.enabled:
                    logging.info(
                        "GRPO running standalone with quantized model. "
                        "Attaching LoRA adapters for fine-tuning..."
                    )
                    log_memory_usage("Before attaching LoRA for GRPO: ")
                    model = attach_lora_adapters(model, cfg)
                    log_memory_usage("After attaching LoRA for GRPO: ")
                else:
                    # SFT already attached LoRA adapters, continue training them
                    logging.info(
                        "GRPO running after SFT. Continuing to train existing LoRA adapters."
                    )

                grpo_steps = grpo_train(
                    model=cast(Any, model),
                    tokenizer=tokenizer,
                    cfg=cfg,
                    data_preparing_func=None,
                )

                # Update global steps from GRPO
                global_steps = grpo_steps
                logging.info(f"GRPO completed. Final global steps: {global_steps}")
                training_completed = True

                # Clean up GRPO datasets after completion if DPO is next
                if cfg.training.use_dpo:
                    log_memory_usage("Before GRPO dataset cleanup: ", cfg=cfg)
                    comprehensive_memory_cleanup(cfg=cfg)
                    log_memory_usage("After GRPO dataset cleanup: ", cfg=cfg)
            except Exception as e:
                raise TrainingError(f"GRPO training phase failed: {e}") from e

        # Phase 3: Direct Preference Optimization (DPO)
        if cfg.training.use_dpo:
            logging.info("Starting DPO training phase...")

            try:
                # If DPO runs standalone (without SFT or GRPO), attach LoRA adapters
                if (
                    not cfg.training.use_sft
                    and not cfg.training.use_grpo
                    and cfg.model.quant.enabled
                ):
                    logging.info(
                        "DPO running standalone with quantized model. "
                        "Attaching LoRA adapters for fine-tuning..."
                    )
                    log_memory_usage("Before attaching LoRA for DPO: ")
                    model = attach_lora_adapters(model, cfg)
                    log_memory_usage("After attaching LoRA for DPO: ")
                else:
                    # Prior training phase(s) already attached
                    # LoRA adapters, continue training them
                    logging.info(
                        "DPO running after prior training phases. "
                        "Continuing to train existing LoRA adapters."
                    )

                # Prepare reference model if specified
                ref_model = None
                if getattr(cfg.dpo, "use_ref_model", False):
                    ref_model_name = getattr(cfg.dpo, "ref_model_name", cfg.model.model_name)
                    logging.info(f"Loading reference model: {ref_model_name}")
                    try:
                        torch_dtype = (
                            getattr(torch, cfg.model.torch_dtype)
                            if isinstance(cfg.model.torch_dtype, str)
                            else cfg.model.torch_dtype
                        )
                        if cfg.model.model_type == "gemma":
                            ref_model = Gemma3ForCausalLM.from_pretrained(
                                ref_model_name,
                                device_map="auto",
                                torch_dtype=torch_dtype,
                            )
                        else:
                            ref_model = AutoModelForCausalLM.from_pretrained(
                                ref_model_name,
                                device_map="auto",
                                torch_dtype=torch_dtype,
                            )
                    except Exception as e:
                        raise TrainingError(
                            f"Failed to load reference model {ref_model_name}: {e}",
                        ) from e

                dpo_steps = dpo_train(
                    model=cast(Any, model),
                    tokenizer=tokenizer,
                    cfg=cfg,
                    data_preparing_func=None,
                    ref_model=ref_model,
                )

                # Update global steps - DPO is the final training phase
                global_steps = dpo_steps
                logging.info(f"DPO completed. Final global steps: {global_steps}")
                training_completed = True

                # Clean up reference model if it was loaded
                if ref_model is not None:
                    cleanup_model(ref_model, "DPO reference model")

                # Clean up DPO datasets after completion
                log_memory_usage("Before DPO dataset cleanup: ", cfg=cfg)
                comprehensive_memory_cleanup(cfg=cfg)
                log_memory_usage("After DPO dataset cleanup: ", cfg=cfg)
            except Exception as e:
                raise TrainingError(f"DPO training phase failed: {e}") from e

        # Validate training completion
        if not training_completed:
            raise TrainingError("Training was configured but did not complete successfully")

        logging.info("Model training completed successfully")

        # Merge adapter weights from the final checkpoint
        try:
            final_checkpoint_path = Path(cfg.model.new_model) / f"checkpoint-{global_steps}"
            if final_checkpoint_path.exists():
                merge_adapter_from_checkpoint(
                    base_model_name=cfg.model.model_name,
                    adapter_dir=str(final_checkpoint_path),
                    save_path=cfg.paths.output_dir,
                    device="cpu",
                )
                logging.info(f"Model merged from checkpoint: {final_checkpoint_path}")
            else:
                logging.error(f"Final checkpoint not found at {final_checkpoint_path}")
                merge_adapter_from_checkpoint(
                    base_model_name=cfg.model.model_name,
                    adapter_dir=cfg.model.new_model,
                    save_path=cfg.paths.output_dir,
                    device="cpu",
                )
        except Exception as e:
            raise TrainingError(f"Failed to merge and save final model: {e}") from e
        finally:
            # Final comprehensive cleanup after training pipeline
            logging.info("Model saved")
            log_memory_usage("Before final training cleanup: ", cfg=cfg)

            # Clean up model
            if "model" in locals():
                cleanup_model(model, "final trained model")

            # Clean up datasets (if they still exist from single-phase training)
            if "train_data" in locals():
                cleanup_dataset(train_data, "final train dataset")
            if "val_data" in locals():
                cleanup_dataset(val_data, "final val dataset")

            # Clean up tokenizer
            if "tokenizer" in locals():
                cleanup_tokenizer(tokenizer, "final tokenizer")

            log_memory_usage("After final training cleanup: ", cfg=cfg)

        return TrainingResult(global_steps=global_steps, eval_loss=eval_loss)

    except (TrainingError, ConfigurationError, ModelLoadingError, DataProcessingError):
        raise
    except Exception as e:
        raise TrainingError(f"Unexpected error during training pipeline: {e}") from e


def merge_adapter_from_checkpoint(
    base_model_name: str,
    adapter_dir: str | Path,
    save_path: str | Path,
    device: str = "cpu",
) -> None:
    """Merge a LoRA adapter checkpoint into the base model and save merged HF model.

    Args:
        base_model_name: HF model identifier or local path to base model.
        adapter_dir: Directory containing the PEFT adapter (e.g. checkpoints).
        save_path: Directory where merged model will be saved.
        device: Device to load model on; use 'cpu' to conserve GPU memory.

    Raises:
        ModelLoadingError: If model or tokenizer loading fails.
        ConversionError: If adapter merging or model saving fails.
    """
    try:
        # Load base model on CPU to avoid GPU OOM, then attach adapter
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map={"": device} if device != "auto" else "auto",
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load base model {base_model_name}: {e}") from e

        try:
            # Try loading tokenizer from checkpoint first (includes custom tokens)
            try:
                tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
                logging.info(f"Loaded tokenizer from checkpoint: {adapter_dir}")
            except Exception:
                # Fallback: load from base model and apply same modifications as training
                logging.info(f"Loading tokenizer from base model: {base_model_name}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                tokenizer.padding_side = "right"
                # Apply same tokenizer modifications as setup_model_and_tokenizer
                if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                    tokenizer.pad_token = "<|pad|>"
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                            tokenizer.pad_token
                        )
                    logging.info("Added pad token to tokenizer")

            # Resize model embeddings to match tokenizer vocabulary
            base_model.resize_token_embeddings(len(tokenizer))
            logging.info(f"Resized model embeddings to match tokenizer: {len(tokenizer)}")
        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load tokenizer for {base_model_name}: {e}",
            ) from e

        try:
            peft_model = PeftModel.from_pretrained(
                cast(nn.Module, base_model),
                adapter_dir,
                device_map={"": device},
            )
        except Exception as e:
            raise ConversionError(
                f"Failed to load PEFT adapter from {adapter_dir}: {e}",
            ) from e

        try:
            # Merge LoRA weights into base weights and free adapter memory
            # Cast peft_model to Any before calling merge_and_unload so the
            # Static analyzer does not confuse the return type with a Tensor.
            merged_model = cast(nn.Module, peft_model.merge_and_unload())
            # Cast to FP16 to avoid FP32 upcast (PEFT adapters default to FP32)
            merged_model = cast(nn.Module, merged_model.half())  # type: ignore[assignment]
            # merged_model is a model instance; save_pretrained is expected on
            # PreTrainedModel-like objects. Use a suppress block for optional
            # save behavior if the merged model doesn't implement it.
            try:
                merged_model.save_pretrained(save_path, safe_serialization=True)  # type: ignore[attr-defined]
            except AttributeError as e:
                logger.warning(f"merged_model does not implement save_pretrained: {e}")
            tokenizer.save_pretrained(save_path)
        except Exception as e:
            raise ConversionError(f"Failed to merge and save model to {save_path}: {e}") from e

    except (ModelLoadingError, ConversionError):
        raise
    except Exception as e:
        raise ConversionError(
            f"Unexpected error during adapter checkpoint merging: {e}",
        ) from e


def convert_to_gguf(
    model_path: str | Path,
    outfile: str | Path,
    python_exe: str | Path,
    outtype: str,
    cfg: DictConfig,
) -> None:
    """Convert Hugging Face model to GGUF format.

    Args:
        model_path: Path to input model directory.
        outfile: Output file path.
        python_exe: Python executable path.
        outtype: Output type specification.
        cfg: Configuration object.

    Raises:
        FileNotFoundError: If required paths are missing.
        ConversionError: If GGUF conversion fails.
    """
    try:
        llama_cpp_dir = Path(cfg.paths.llama_cpp_dir).resolve()
        conversion_script = llama_cpp_dir / "convert_hf_to_gguf.py"

        if not llama_cpp_dir.is_dir():
            raise FileNotFoundError(f"llama.cpp directory not found: {llama_cpp_dir}")
        if not conversion_script.exists():
            raise FileNotFoundError(f"Conversion script missing: {conversion_script}")

        model_path_str = str(Path(model_path).resolve())
        outfile_str = str(Path(outfile).resolve())
        python_exe_str = str(Path(cfg.paths.venv_python_path))

        subprocess.run(
            [
                python_exe_str,
                str(conversion_script),
                model_path_str,
                "--outfile",
                outfile_str,
                "--outtype",
                outtype,
            ],
            check=True,
            cwd=str(llama_cpp_dir),
        )
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"GGUF conversion failed. Check:\n"
            f"- llama.cpp exists at {cfg.paths.llama_cpp_dir}\n"
            f"- Conversion script exists: "
            f"{Path(cfg.paths.llama_cpp_dir) / 'convert_hf_to_gguf.py'}\n"
            f"- Python executable: {python_exe}\n"
            f"- Model path: {model_path}"
        )
        logging.exception(error_msg)
        raise ConversionError(
            f"GGUF conversion subprocess failed with exit code {e.returncode}",
        ) from e
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ConversionError(f"Unexpected error during GGUF conversion: {e}") from e
    finally:
        comprehensive_memory_cleanup(aggressive=True, cfg=cfg)
        log_memory_usage("After GGUF conversion cleanup: ", cfg=cfg)


def convert_to_rkllm(
    model_path: str | Path,
    output_dir: str | Path,
    target_platform: str = "rk3588",
    quantization: str = "w8a8",
    do_parallelize: bool = False,
    hybrid_quantization: bool = False,
    num_npu_core: int = 1,
    max_context: int = 4096,
) -> str:
    """Convert Hugging Face model to RKLLM format using dedicated RKLLM container.

    Args:
        model_path: Path to input Hugging Face model directory or GGUF file.
        output_dir: Directory to save RKLLM model.
        target_platform: Target Rockchip platform (rk3588, rk3576, etc.).
        quantization: Quantization type (w8a8, w4a16, w4a16_g128).
        do_parallelize: Enable model parallelization for larger models.
        hybrid_quantization: Enable hybrid quantization.
        num_npu_core: Number of NPU cores to use (1-3).
        max_context: Maximum context length.

    Returns:
        Path to the generated RKLLM model file.

    Raises:
        RuntimeError: If RKLLM conversion fails.
        ImportError: If RKLLM container is not available.
    """
    import subprocess
    from pathlib import Path

    # Validate configuration before proceeding
    validate_rkllm_config_params(target_platform, quantization, num_npu_core)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename based on configuration
    model_name = Path(model_path).name
    output_filename = f"{model_name}_{target_platform}_{quantization}.rkllm"
    output_file_path = output_path / output_filename

    # Convert paths to absolute paths for Docker mounting
    abs_model_path = Path(model_path).resolve()
    abs_output_dir = output_path.resolve()

    logging.info("Converting model to RKLLM format using dedicated container...")
    logging.info(f"  Source: {abs_model_path}")
    logging.info(f"  Target platform: {target_platform}")
    logging.info(f"  Quantization: {quantization}")
    logging.info(f"  Output: {output_file_path}")

    try:
        # Check if RKLLM container is available
        check_cmd = ["docker", "images", "-q", "rkllm_converter"]
        result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)

        if not result.stdout.strip():
            # Try to build the RKLLM container if it doesn't exist
            logging.warning("RKLLM container not found, attempting to build...")

            # Build from the rkllm_files directory context
            rkllm_files_path = Path("rkllm_files").resolve()
            build_cmd = [
                "docker",
                "build",
                "-f",
                str(rkllm_files_path / "Dockerfile.rkllm"),
                "-t",
                "rkllm_converter",
                str(rkllm_files_path),
            ]

            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if build_result.returncode != 0:
                raise ImportError(
                    f"RKLLM container build failed. "
                    f"Please build the RKLLM container first:\n"
                    f"docker build -f rkllm_files/Dockerfile.rkllm -t "
                    f"rkllm_converter rkllm_files/\n\n"
                    f"Build error: {build_result.stderr}",
                )

            logging.info("RKLLM container built successfully")

        # Auto-detect model format
        model_format = "auto"
        if abs_model_path.suffix.lower() == ".gguf":
            model_format = "gguf"
        elif abs_model_path.is_dir() and (abs_model_path / "config.json").exists():
            model_format = "huggingface"

        # Prepare Docker command for RKLLM conversion
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{abs_model_path}:/input",
            "-v",
            f"{abs_output_dir}:/output",
            "rkllm_converter",
            "convert",
            "--model-path",
            "/input",
            "--output-path",
            f"/output/{output_filename}",
            "--target-platform",
            target_platform,
            "--quantization",
            quantization,
            "--num-npu-core",
            str(num_npu_core),
            "--model-format",
            model_format,
            "--max-context",
            str(max_context),
        ]

        # Add optional parameters
        if do_parallelize:
            docker_cmd.append("--do-parallelize")
        if hybrid_quantization:
            docker_cmd.append("--hybrid-quantization")

        logging.info(f"Running RKLLM conversion command: {' '.join(docker_cmd)}")

        # Execute the conversion
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=False,
        )

        # Log the output
        if result.stdout:
            logging.info(f"RKLLM conversion output:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"RKLLM conversion stderr:\n{result.stderr}")

        if result.returncode != 0:
            raise RuntimeError(
                f"RKLLM conversion failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(docker_cmd)}\n"
                f"Error output: {result.stderr}\n"
                f"Standard output: {result.stdout}",
            )

        # Verify the output file exists
        if not output_file_path.exists():
            raise RuntimeError(f"RKLLM model file was not created: {output_file_path}")

        file_size = output_file_path.stat().st_size / (1024 * 1024)  # Size in MB
        logging.info("RKLLM conversion completed successfully")
        logging.info(f"  Output file: {output_file_path}")
        logging.info(f"  File size: {file_size:.2f} MB")

        return str(output_file_path)

    except subprocess.TimeoutExpired:
        logging.error("RKLLM conversion timed out after 1 hour")
        raise RuntimeError("RKLLM conversion timed out") from None
    except Exception as e:
        logging.exception(f"RKLLM conversion failed: {e}")
        # Clean up partial files
        if output_file_path.exists():
            output_file_path.unlink()
        raise RuntimeError(f"RKLLM conversion failed: {e}") from e


def safe_import_rkllm() -> None:
    """Safely import RKLLM with detailed diagnostics."""
    diagnostic_info = []
    import importlib
    import os
    import subprocess
    import sys
    import tempfile

    # Attempt dynamic import to avoid static analysis false-positives
    try:
        rkllm = importlib.import_module("rkllm")
        path = getattr(rkllm, "__file__", "<unknown>")
        diagnostic_info.append(f"✓ RKLLM package found at: {path}")
    except Exception as e:  # ImportError or other import-time errors
        diagnostic_info.append(f"✗ RKLLM package import failed: {e}")

        # Check if rknn-toolkit2 is available
        try:
            rknn_toolkit2 = importlib.import_module("rknn_toolkit2")
            diagnostic_info.append(
                "✓ rknn-toolkit2 available: "
                + str(getattr(rknn_toolkit2, "__version__", "<unknown>")),
            )
        except Exception:
            diagnostic_info.append("✗ rknn-toolkit2 not found")

        # Check pip-installed package presence using current interpreter
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "show",
                    "rkllm-toolkit",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                diagnostic_info.append("✓ rkllm-toolkit package installed")
                diagnostic_info.append(result.stdout[:500])  # Limit output
            else:
                diagnostic_info.append("✗ rkllm-toolkit package not found")
        except Exception as subprocess_error:
            diagnostic_info.append(f"✗ Failed to check installation: {subprocess_error}")

        # Check if installation log exists in a portable temp directory
        log_file = os.path.join(tempfile.gettempdir(), "rkllm_install.log")
        if os.path.exists(log_file):
            try:
                with open(log_file) as f:
                    log_content = f.read()[-1000:]  # Last 1000 characters
                diagnostic_info.append("Recent installation log:")
                diagnostic_info.append(log_content)
            except Exception:
                diagnostic_info.append("✗ Failed to read installation log")

        msg = (
            "RKLLM toolkit not available.\n"
            "DIAGNOSTIC INFORMATION:\n"
            + "\n".join(diagnostic_info)
            + "\n\nSOLUTIONS:\n"
            + "1. For Docker: Rebuild container with 'docker-compose build --no-cache'\n"
            + "2. Check installation logs at "
            + tempfile.gettempdir()
            + os.sep
            + "rkllm_install.log\n"
            + "3. Verify network connectivity for downloading RKLLM toolkit\n"
            + "4. For persistent issues, check GitHub "
            "releases at https://github.com/airockchip/rknn-llm/releases"
        )
        raise ImportError(msg) from e

    # Import RKLLM API dynamically
    try:
        rkllm_api = importlib.import_module("rkllm.api")
        rkllm_class = rkllm_api.RKLLM
        diagnostic_info.append("✓ RKLLM.api import successful")
        return rkllm_class
    except Exception as e:
        diagnostic_info.append(f"✗ RKLLM.api import failed: {e}")
        raise ImportError(
            "RKLLM API not available.\n"
            "DIAGNOSTIC INFORMATION:\n" + "\n".join(diagnostic_info),
        ) from e


def validate_rkllm_config_params(
    target_platform: str,
    quantization: str,
    num_npu_core: int,
) -> None:
    """Validate RKLLM configuration parameters."""
    # Validate platform
    valid_platforms = ["rk3588", "rk3576", "rk3566", "rk3568"]
    if target_platform not in valid_platforms:
        raise ValueError(
            f"Invalid target_platform: {target_platform}. "
            f"Valid options: {valid_platforms}",
        )

    # Validate quantization
    valid_quantizations = ["w8a8", "w4a16", "w4a16_g128"]
    if quantization not in valid_quantizations:
        raise ValueError(
            f"Invalid quantization: {quantization}. " f"Valid options: {valid_quantizations}",
        )

    # Validate NPU core count
    if not (1 <= num_npu_core <= 3):
        raise ValueError(f"Invalid num_npu_core: {num_npu_core}. " "Must be between 1 and 3")

    logging.info("✓ RKLLM configuration parameters validated successfully")


def rkllm_quantize(
    model_path: str | Path,
    output_path: str | Path,
    quantization: str = "w8a8",
    target_platform: str = "rk3588",
    **kwargs: dict,
) -> bool:
    """Quantize model for RKLLM format (integrated within convert_to_rkllm).

    This function is a wrapper that calls convert_to_rkllm with quantization.
    The actual quantization is performed during the RKLLM conversion process.

    Args:
        model_path: Path to input model.
        output_path: Path for quantized output.
        quantization: Quantization type (w8a8, w4a16, w4a16_g128).
        target_platform: Target Rockchip platform.
        **kwargs: Additional parameters for convert_to_rkllm.

    Returns:
        True if quantization succeeded, False otherwise.

    Raises:
        ConversionError: If RKLLM quantization fails.
    """
    try:
        output_dir = Path(output_path).parent
        result_path = convert_to_rkllm(
            model_path=model_path,
            output_dir=str(output_dir),
            target_platform=target_platform,
            quantization=quantization,
            **kwargs,
        )

        # If output filename is different, rename the file
        result_path_obj = Path(result_path)
        output_path_obj = Path(output_path)
        if result_path_obj != output_path_obj and result_path_obj.exists():
            shutil.move(str(result_path_obj), str(output_path_obj))
            logging.info(f"RKLLM model moved to: {output_path}")

        return True

    except Exception as e:
        logging.exception(f"RKLLM quantization failed: {e}")
        raise ConversionError(f"RKLLM quantization failed: {e}") from e


def quantize_model(
    model_path: str | Path,
    outfile: str | Path,
    qtype: str = "q4_0",
    llama_cpp_path: str | Path = ".",
    quantized_path: str | Path = "llama-quantize.exe",
) -> bool:
    """Quantize GGUF model using llama.cpp quantizer.

    Args:
        model_path: Path to input GGUF model.
        outfile: Path for quantized output.
        qtype: Quantization type (default: q4_0).
        llama_cpp_path: Path to llama.cpp directory.
        quantized_path: Name of quantizer executable.

    Returns:
        True if quantization succeeded, False otherwise.

    Raises:
        ConversionError: If quantization process fails.
    """
    try:
        llama_cpp_dir = Path(llama_cpp_path).resolve()
        llama_quantize_path = llama_cpp_dir / quantized_path
        model_path_abs = Path(model_path).resolve()
        outfile_abs = Path(outfile).resolve()

        if not llama_quantize_path.exists():
            error_msg = f"Error: llama-quantize.exe not found at {llama_quantize_path}"
            logging.error(error_msg)
            return False

        logging.info("Trying to quantize model...")
        command = [str(llama_quantize_path), str(model_path_abs), str(outfile_abs), qtype]
        logging.info(f"Running command: {command}")

        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                cwd=str(llama_cpp_dir),
            )
            logging.info("Model quantized")
            logging.info(f"Command output: {process.stdout.decode()}")
            logging.info(
                f"Command stderr: {process.stderr.decode() if process.stderr else 'None'}",
            )
            return True
        except subprocess.CalledProcessError as e:
            logging.exception(f"Command failed with exit code {e.returncode}")
            return False
    except Exception as e:
        logging.exception(f"Quantization process failed: {e}")
        return False


def copy_data(
    file: str,
    gguf_directory: str = "custom-model",
    destination: str = r"T:\lm-studio\models\game-model",
) -> None:
    """Move file to destination directory with versioning.

    Args:
        file: Source file name.
        gguf_directory: Version subdirectory.
        destination: Root destination directory.

    Raises:
        ConversionError: If file copy operation fails.
    """
    try:
        destination_path = Path(destination) / gguf_directory / file
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = Path.cwd() / file

        if not source_path.exists():
            raise ConversionError(f"Source file not found: {source_path}")

        shutil.move(str(source_path), str(destination_path))
        logging.info(f"File moved from {source_path} to {destination_path}")
    except Exception as e:
        raise ConversionError(f"Failed to copy file {file} to destination: {e}") from e


def train_pipeline(cfg: DictConfig) -> dict[str, Any]:
    """Execute complete training pipeline including conversion and quantization.

    Args:
        cfg: Configuration object.

    Returns:
        A dict with 'eval_loss'.

    Raises:
        TrainingError: If training pipeline fails.
        ConversionError: If model conversion or quantization fails.
        ExperimentTrackingError: If experiment logging fails.
    """
    try:
        # Initialize optional experiment logging backend (wandb / mlflow / none)
        init_logging_backend(cfg)

        # Validate MLflow connection if MLflow backend is selected
        if not validate_mlflow_connection(cfg):
            logging.warning(
                "MLflow connection validation failed, but continuing with training",
            )

        # Log training configuration to MLflow if enabled
        try:
            log_training_config(cfg)
        except Exception as e:
            raise ExperimentTrackingError(f"Failed to log training configuration: {e}") from e

        try:
            result = train(cfg)
            merge_adapter_from_checkpoint(
                base_model_name=cfg.model.model_name,
                adapter_dir=str(
                    Path(cfg.model.new_model) / f"checkpoint-{result['global_steps']}",
                ),  # where trainer saved checkpoint-<steps>
                save_path=cfg.paths.output_dir,
                device="cpu",  # merge on CPU to avoid OOM
            )
            steps = result["global_steps"]
            eval_loss = result["eval_loss"]
        except Exception as e:
            raise TrainingError(f"Training phase failed: {e}") from e

        try:
            with TemporaryDirectory() as merged_model_dir:
                model_merge_for_converting(cfg, steps, merged_model_dir)

                # GGUF conversion (optional based on config)
                if cfg.model.quant.get(
                    "convert_to_gguf",
                    True,
                ):  # Default to True for backward compatibility
                    outfile = cfg.model.outfile

                    try:
                        convert_to_gguf(
                            model_path=merged_model_dir,
                            outfile=str(Path(merged_model_dir) / outfile),
                            python_exe=cfg.paths.venv_python_path,
                            outtype="f16",
                            cfg=cfg,
                        )
                        logging.info(f"Converted to GGUF: {outfile}")
                    except Exception as e:
                        raise ConversionError(f"GGUF conversion failed: {e}") from e

                    quantized_file = outfile
                    if not quantize_model(
                        model_path=str(Path(merged_model_dir) / outfile),
                        outfile=quantized_file,
                        qtype=cfg.model.quant.qtype,
                        llama_cpp_path=str(Path(cfg.paths.llama_cpp_dir).resolve()),
                        quantized_path=cfg.paths.quantized_path,
                    ):
                        raise ConversionError("Model quantization failed")

                    try:
                        copy_data(
                            quantized_file,
                            cfg.model.quant.gguf_dir,
                            cfg.paths.final_weights_path,
                        )
                        quantized_file_path = Path(quantized_file)
                        if quantized_file_path.exists():
                            quantized_file_path.unlink()
                            logging.info(f"Removed intermediate file: {quantized_file}")
                    except Exception as e:
                        raise ConversionError(f"Failed to copy quantized model: {e}") from e
                else:
                    # Alternative flow: copy merged HuggingFace model directly to output
                    try:
                        logging.info("GGUF conversion disabled, copying merged model directly")
                        merged_output_dir = Path(cfg.paths.output_dir) / "merged_model"
                        if merged_output_dir.exists():
                            shutil.rmtree(merged_output_dir)
                        shutil.copytree(merged_model_dir, merged_output_dir)
                        logging.info(f"Merged model copied to: {merged_output_dir}")
                    except Exception as e:
                        raise ConversionError(f"Failed to copy merged model: {e}") from e

                # RKLLM conversion (if enabled)
                if getattr(cfg.model, "rkllm", {}).get("enabled", False):
                    logging.info("Starting RKLLM conversion...")
                    try:
                        rkllm_config = cfg.model.rkllm
                        rkllm_output_dir = Path(cfg.paths.output_dir) / rkllm_config.output_dir

                        rkllm_model_path = convert_to_rkllm(
                            model_path=cfg.paths.output_dir,  # Use the merged model directory
                            output_dir=str(rkllm_output_dir),
                            target_platform=rkllm_config.target_platform,
                            quantization=rkllm_config.quantization,
                            do_parallelize=rkllm_config.get("do_parallelize", False),
                            hybrid_quantization=rkllm_config.get("hybrid_quantization", False),
                            num_npu_core=rkllm_config.get("num_npu_core", 1),
                        )

                        logging.info(f"RKLLM conversion completed: {rkllm_model_path}")

                    except Exception as e:
                        logging.exception(f"RKLLM conversion failed: {e}")
                        # Don't raise error - RKLLM conversion is optional
                        logging.info("Continuing without RKLLM conversion...")
                else:
                    logging.info("RKLLM conversion disabled in configuration")
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(f"Model processing pipeline failed: {e}") from e

        # Log training artifacts to MLflow if enabled
        if cfg.logging.logging_backend == "mlflow":
            try:
                if cfg.logging.mlflow.log_artifacts:
                    logging.info("Logging training artifacts...")
                    log_training_artifacts(cfg, cfg.paths.output_dir, steps)
                # Log evaluation metrics to MLflow if enabled
                log_evaluation_metrics({"eval_loss": eval_loss, "global_steps": steps})
            except Exception as e:
                raise ExperimentTrackingError(
                    f"Failed to log training artifacts or metrics: {e}",
                ) from e

        logging.info("Training pipeline completed")
        return {"eval_loss": eval_loss}

    except (TrainingError, ConversionError, ExperimentTrackingError):
        raise
    except Exception as e:
        raise TrainingError(f"Unexpected error in training pipeline: {e}") from e
    finally:
        # Finish/cleanup configured logging backend (if any) and comprehensive memory cleanup
        try:
            finish_logging_backend()
        except Exception as e:
            logging.warning(f"Failed to cleanup logging backend: {e}")

        # Final pipeline cleanup
        log_memory_usage("Before pipeline completion cleanup: ")
        from .memory_utils import comprehensive_memory_cleanup

        comprehensive_memory_cleanup(aggressive=True)
        log_memory_usage("After pipeline completion cleanup: ")


def main_train(data_dir: str, cfg: DictConfig) -> dict[str, Any]:
    """Main training entry point with dataset processing.

    Args:
        data_dir: Directory containing training data.
        cfg: Configuration object.

    Returns:
        Result dictionary from training pipeline.

    Raises:
        TrainingError: If training pipeline fails.
        DataProcessingError: If dataset processing fails.
    """
    try:
        result = train_pipeline(cfg)
        comprehensive_memory_cleanup(aggressive=cfg.memory_management.aggressive_cleanup)
        log_memory_usage("Final script cleanup: ", cfg=cfg)

        try:
            test_file_path = Path(data_dir) / "test_ru.json"
            if not test_file_path.exists():
                raise DataProcessingError(f"Test dataset file not found: {test_file_path}")

            with test_file_path.open(encoding="utf-8") as file:
                test_dataset = json.load(file)

            dataset_to_json(
                test_dataset,
                cfg.testing.output_test_file,
                cfg.data_preparation.method,
            )
        except json.JSONDecodeError as e:
            raise DataProcessingError(f"Invalid JSON format in test dataset: {e}") from e
        except Exception as e:
            raise DataProcessingError(f"Failed to process test dataset: {e}") from e

        return result

    except (TrainingError, DataProcessingError):
        raise
    except Exception as e:
        raise TrainingError(f"Unexpected error in main training function: {e}") from e


def post_new_dataset() -> None:
    """Upload new dataset version to remote server.

    Raises:
        DataProcessingError: If dataset upload fails.
    """
    _load_environment_if_needed()
    password = os.getenv("PASSWORD_BOT")
    try:
        url = "https://dataset.ser13volk.me/dataset_ru"
        dataset_path = Path("../data") / "dataset_ru.json"

        if not dataset_path.exists():
            raise DataProcessingError(f"Dataset file not found: {dataset_path}")

        with dataset_path.open("rb") as f:
            files = {"file": f}
            response = requests.post(
                url,
                files=files,
                auth=HTTPBasicAuth("admin", password),
                timeout=30,
            )

        response.raise_for_status()  # Raises HTTPError for bad HTTP status codes
        logging.info(response.json())

    except requests.RequestException as e:
        raise DataProcessingError(f"Failed to upload dataset to remote server: {e}") from e
    except Exception as e:
        raise DataProcessingError(f"Unexpected error during dataset upload: {e}") from e


if __name__ == "__main__":
    configure_logging(logging.DEBUG)
    post_new_dataset()
