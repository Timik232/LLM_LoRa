"""Main file for model training."""

import functools
import gc
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import requests
import torch
from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel
from requests.auth import HTTPBasicAuth
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
)
from trl import SFTConfig, SFTTrainer

import wandb

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
    log_evaluation_metrics,
    log_training_artifacts,
    log_training_config,
    validate_mlflow_connection,
)
from .types import ModelType, TrainingResult

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


def generate_prompt(tokenizer: AutoTokenizer, data_point: dict[str, str]) -> str:
    """Generate a chat template prompt for the model.

    Args:
        tokenizer: Hugging Face tokenizer.
        data_point: Dictionary containing system, user and bot messages.

    Returns:
        Formatted chat prompt.
    """
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": data_point["system"]},
            {"role": "user", "content": data_point["user"]},
            {"role": "assistant", "content": data_point["bot"]},
        ],
        tokenize=False,
    )


def tokenize(
    tokenizer: AutoTokenizer, cutoff_len: int, prompt: str
) -> dict[str, torch.Tensor]:
    """Tokenize text with specified length constraints.

    Args:
        tokenizer: Hugging Face tokenizer.
        cutoff_len: Maximum sequence length.
        prompt: Text to tokenize.

    Returns:
        Tokenized output dictionary.
    """
    return tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors=None,
        add_special_tokens=True,
    )


def generate_and_tokenize_prompt(
    data_point: dict[str, str],
    tokenizer: AutoTokenizer,
    cutoff: int,
    should_add_prompt: bool = False,
) -> dict[str, Any]:
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
    return tokenized_full_prompt


def generate_grpo_prompt(tokenizer: AutoTokenizer, data_point: dict[str, str]) -> str:
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # This adds the assistant prompt without response
    )


def data_preparation(
    cfg: DictConfig, tokenizer: AutoTokenizer, should_add_prompt: bool = False
) -> tuple[Dataset, Dataset]:
    """Prepare and preprocess training and validation datasets.

    This function accepts either:
      - a JSON file containing a dict with keys: "system" and "examples" (list), or
      - separate train/test files when cfg.testing.use_separate_files is True.

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
        try:
            base = Path(get_original_cwd()) / cfg.paths.data_dir
        except ValueError:
            # Fallback if Hydra context is not available
            base = Path.cwd() / cfg.paths.data_dir

        if not cfg.testing.use_separate_files:
            single_path = base / cfg.paths.train_data
            if not single_path.exists():
                raise DataProcessingError(f"Training data file not found: {single_path}")

            try:
                raw = json.loads(single_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise DataProcessingError(f"Invalid JSON format in {single_path}: {e}") from e
            except Exception as e:
                raise DataProcessingError(
                    f"Failed to read training data from {single_path}: {e}"
                ) from e

            all_items: list[dict]
            if isinstance(raw, dict) and "examples" in raw:
                all_items = raw["examples"]
            else:
                raise DataProcessingError(
                    f"Unrecognized JSON structure in {single_path}. "
                    "Expected dict with 'examples' key."
                )

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

        else:
            train_path = base / cfg.paths.train_file
            test_path = base / cfg.paths.test_file

            if not train_path.exists():
                raise DataProcessingError(f"Training file not found: {train_path}")
            if not test_path.exists():
                raise DataProcessingError(f"Test file not found: {test_path}")

            try:
                train_dataset = json.loads(train_path.read_text(encoding="utf-8"))
                test_dataset = json.loads(test_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise DataProcessingError(f"Invalid JSON format in data files: {e}") from e
            except Exception as e:
                raise DataProcessingError(f"Failed to read data files: {e}") from e

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
                        f"Failed to convert dataset to JSON format: {e}"
                    ) from e

                from datasets import load_dataset

                try:
                    dataset = load_dataset(
                        "json", data_files={"train": train_json, "test": test_json}
                    )
                except Exception as e:
                    raise DataProcessingError(
                        f"Failed to load dataset with HuggingFace datasets library: {e}"
                    ) from e

                try:
                    tokenize_partial = functools.partial(
                        generate_and_tokenize_prompt,
                        tokenizer=tokenizer,
                        cutoff=cfg.other.cutoff_len,
                        should_add_prompt=should_add_prompt,
                    )
                    train_data = dataset["train"].map(tokenize_partial)
                    val_data = dataset["test"].map(tokenize_partial)
                except Exception as e:
                    raise DataProcessingError(f"Failed to tokenize datasets: {e}") from e
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
                model = Gemma3ForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype="auto",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype="auto",
                )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load base model {model_path}: {e}") from e

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            raise ModelLoadingError(f"Failed to load tokenizer for {model_path}: {e}") from e

        # Load and merge adapter
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
        except Exception as e:
            raise ConversionError(
                f"Failed to load and merge adapter from {adapter_path}: {e}"
            ) from e

        # Save merged model
        try:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        except Exception as e:
            raise ConversionError(f"Failed to save merged model to {save_path}: {e}") from e
        finally:
            # Clean up resources
            if "model" in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()

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
            f"Must be one of: float16, bfloat16, float32"
        ) from e

    try:
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
            f"Failed to load tokenizer for {cfg.model.model_name}: {e}"
        ) from e

    return model, tokenizer


def run_sft_training(
    model: ModelType,
    tokenizer: AutoTokenizer,
    cfg: DictConfig,
    train_data: Dataset,
    val_data: Dataset,
) -> tuple[int, float]:
    """Run SFT training phase.

    Args:
        model: The model to train.
        tokenizer: The tokenizer.
        cfg: Configuration object.
        train_data: Training dataset.
        val_data: Validation dataset.

    Returns:
        Tuple of (global_steps, eval_loss).

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
            modules_to_save=["lm_head"],
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

    try:
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
            optim=cfg.training.optim,
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
            group_by_length=True,
            report_to=report_to_backend,  # Use dynamic backend selection
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=cfg.training.load_best,
        )
        if cfg.training.use_optuna_optimize:
            sft_config.run_name = f"{sft_config.run_name}_optuna"
    except Exception as e:
        raise ConfigurationError(f"Failed to create SFT configuration: {e}") from e

    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=sft_config,
        )
        trainer.train()
        global_steps = trainer.state.global_step
        eval_results = trainer.evaluate()
        eval_loss = eval_results["eval_loss"]
        logging.info(
            f"SFT completed. Global steps: {global_steps}, Evaluation loss: {eval_loss}"
        )
    except Exception as e:
        raise TrainingError(f"SFT training failed: {e}") from e
    finally:
        # Clean up SFT trainer to free memory before next training phase
        if "trainer" in locals():
            del trainer
        gc.collect()
        torch.cuda.empty_cache()

    return global_steps, eval_loss


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
                "training.use_sft, training.use_grpo, or training.use_dpo"
            )

        # Phase 1: Supervised Fine-Tuning (SFT)
        if cfg.training.use_sft:
            try:
                global_steps, eval_loss = run_sft_training(
                    model, tokenizer, cfg, train_data, val_data
                )
                training_completed = True
            except Exception as e:
                raise TrainingError(f"SFT training phase failed: {e}") from e

        # Phase 2: Group Relative Policy Optimization (GRPO)
        if cfg.training.use_grpo:
            logging.info("Starting GRPO training phase...")

            try:
                # If both SFT and GRPO are enabled, load the best checkpoint from SFT
                if cfg.training.use_sft and training_completed:
                    logging.info("Loading SFT checkpoint for GRPO training...")
                    checkpoint_path = Path(cfg.model.new_model) / f"checkpoint-{global_steps}"
                    if checkpoint_path.exists():
                        # Load the adapter weights into the model
                        model = PeftModel.from_pretrained(model, str(checkpoint_path))
                        logging.info(f"Loaded SFT checkpoint from {checkpoint_path}")
                    else:
                        logging.warning(
                            f"SFT checkpoint not found at {checkpoint_path}, "
                            "continuing with current model state"
                        )

                grpo_steps = grpo_train(
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    data_preparing_func=None,
                )

                # Update global steps if GRPO was the only training method or ran after SFT
                if not cfg.training.use_sft:
                    global_steps = grpo_steps
                else:
                    # If both SFT and GRPO ran, use GRPO steps as the final checkpoint
                    global_steps = grpo_steps

                logging.info(f"GRPO completed. Final global steps: {global_steps}")
                training_completed = True
            except Exception as e:
                raise TrainingError(f"GRPO training phase failed: {e}") from e

        # Phase 3: Direct Preference Optimization (DPO)
        if cfg.training.use_dpo:
            logging.info("Starting DPO training phase...")

            try:
                # If previous training phases completed, load the best checkpoint
                if training_completed:
                    logging.info("Loading previous checkpoint for DPO training...")
                    checkpoint_path = Path(cfg.model.new_model) / f"checkpoint-{global_steps}"
                    if checkpoint_path.exists():
                        # Load the adapter weights into the model
                        model = PeftModel.from_pretrained(model, str(checkpoint_path))
                        logging.info(f"Loaded checkpoint from {checkpoint_path}")
                    else:
                        logging.warning(
                            f"Checkpoint not found at {checkpoint_path}, "
                            "continuing with current model state"
                        )

                # Prepare reference model if specified
                ref_model = None
                if getattr(cfg.dpo, "use_ref_model", False):
                    ref_model_name = getattr(cfg.dpo, "ref_model_name", cfg.model.model_name)
                    logging.info(f"Loading reference model: {ref_model_name}")
                    # Load reference model with same configuration as the main model
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
                            f"Failed to load reference model {ref_model_name}: {e}"
                        ) from e

                dpo_steps = dpo_train(
                    model=model,
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
                    del ref_model
                    gc.collect()
                    torch.cuda.empty_cache()
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
                # Still try to save the model in its current state
                merge_adapter_from_checkpoint(
                    base_model_name=cfg.model.model_name,
                    adapter_dir=cfg.model.new_model,  # Fallback to the output directory
                    save_path=cfg.paths.output_dir,
                    device="cpu",
                )
        except Exception as e:
            raise TrainingError(f"Failed to merge and save final model: {e}") from e
        finally:
            # Clean up model resources
            logging.info("Model saved")
            if "model" in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()

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
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load tokenizer for {base_model_name}: {e}"
            ) from e

        try:
            peft_model = PeftModel.from_pretrained(
                base_model, adapter_dir, device_map={"": device}
            )
        except Exception as e:
            raise ConversionError(
                f"Failed to load PEFT adapter from {adapter_dir}: {e}"
            ) from e

        try:
            # Merge LoRA weights into base weights and free adapter memory
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        except Exception as e:
            raise ConversionError(f"Failed to merge and save model to {save_path}: {e}") from e

    except (ModelLoadingError, ConversionError):
        raise
    except Exception as e:
        raise ConversionError(
            f"Unexpected error during adapter checkpoint merging: {e}"
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
            f"GGUF conversion subprocess failed with exit code {e.returncode}"
        ) from e
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ConversionError(f"Unexpected error during GGUF conversion: {e}") from e


def convert_to_rkllm(
    model_path: str | Path,
    output_dir: str | Path,
    target_platform: str = "rk3588",
    quantization: str = "w8a8",
    do_parallelize: bool = False,
    hybrid_quantization: bool = False,
    num_npu_core: int = 1,
) -> str:
    """Convert Hugging Face model to RKLLM format for Rockchip NPU.

    Args:
        model_path: Path to input Hugging Face model directory.
        output_dir: Directory to save RKLLM model.
        target_platform: Target Rockchip platform (rk3588, rk3576, etc.).
        quantization: Quantization type (w8a8, w4a16, w4a16_g128).
        do_parallelize: Enable model parallelization for larger models.
        hybrid_quantization: Enable hybrid quantization.
        num_npu_core: Number of NPU cores to use (1-3).

    Returns:
        Path to the generated RKLLM model file.

    Raises:
        RuntimeError: If RKLLM conversion fails.
        ImportError: If rkllm-toolkit is not available.
    """
    try:
        from rkllm.api import RKLLM
    except ImportError as e:
        raise ImportError(
            "RKLLM toolkit not found. RKLLM conversion requires the rkllm-toolkit package.\n"
            "For Docker: Rebuild container with 'docker-compose build --no-cache'\n"
            "For local development: RKLLM is not supported due to PyTorch version conflicts\n"
            "The updated Dockerfile now includes improved RKLLM installation with fallbacks.\n"
            f"Original error: {e}"
        ) from e

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename based on configuration
    model_name = Path(model_path).name
    output_filename = f"{model_name}_{target_platform}_{quantization}.rkllm"
    output_file_path = output_path / output_filename

    logging.info("Converting model to RKLLM format...")
    logging.info(f"  Source: {model_path}")
    logging.info(f"  Target platform: {target_platform}")
    logging.info(f"  Quantization: {quantization}")
    logging.info(f"  Output: {output_file_path}")

    try:
        # Initialize RKLLM converter
        rkllm = RKLLM()

        # Load model configuration
        ret = rkllm.load_huggingface(
            model=str(model_path), target_platform=target_platform, num_npu_core=num_npu_core
        )

        if ret != 0:
            raise RuntimeError(f"Failed to load Hugging Face model: {model_path}")

        # Configure quantization and optimization settings
        ret = rkllm.build(
            do_quantization=True,
            optimization_level=1,
            quantized_dtype=quantization,
            target_platform=target_platform,
            num_npu_core=num_npu_core,
            do_parallelize=do_parallelize,
            hybrid_quantization=hybrid_quantization,
        )

        if ret != 0:
            raise RuntimeError(
                f"Failed to build RKLLM model with quantization: {quantization}"
            )

        # Export the model
        ret = rkllm.export_rkllm(str(output_file_path))

        if ret != 0:
            raise RuntimeError(f"Failed to export RKLLM model to: {output_file_path}")

        # Verify the output file exists
        if not output_file_path.exists():
            raise RuntimeError(f"RKLLM model file was not created: {output_file_path}")

        file_size = output_file_path.stat().st_size / (1024 * 1024)  # Size in MB
        logging.info("RKLLM conversion completed successfully")
        logging.info(f"  Output file: {output_file_path}")
        logging.info(f"  File size: {file_size:.2f} MB")

        return str(output_file_path)

    except Exception as e:
        logging.exception(f"RKLLM conversion failed: {e}")
        # Clean up partial files
        if output_file_path.exists():
            output_file_path.unlink()
        raise RuntimeError(f"RKLLM conversion failed: {e}") from e

    finally:
        # Ensure RKLLM resources are cleaned up
        try:
            if "rkllm" in locals():
                del rkllm
        except Exception as cleanup_error:
            logging.warning(f"Failed to cleanup RKLLM resources: {cleanup_error}")


def rkllm_quantize(
    model_path: str | Path,
    output_path: str | Path,
    quantization: str = "w8a8",
    target_platform: str = "rk3588",
    **kwargs: Any,
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
                command, check=True, capture_output=True, cwd=str(llama_cpp_dir)
            )
            logging.info("Model quantized")
            logging.info(f"Command output: {process.stdout.decode()}")
            logging.info(
                f"Command stderr: {process.stderr.decode() if process.stderr else 'None'}"
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
                "MLflow connection validation failed, but continuing with training"
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
                    Path(cfg.model.new_model) / f"checkpoint-{result['global_steps']}"
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
                    "enabled", True
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
        try:
            log_training_artifacts(cfg, cfg.paths.output_dir, steps)
            # Log evaluation metrics to MLflow if enabled
            log_evaluation_metrics({"eval_loss": eval_loss, "global_steps": steps})
        except Exception as e:
            raise ExperimentTrackingError(
                f"Failed to log training artifacts or metrics: {e}"
            ) from e

        logging.info("Training pipeline completed")
        return {"eval_loss": eval_loss}

    except (TrainingError, ConversionError, ExperimentTrackingError):
        raise
    except Exception as e:
        raise TrainingError(f"Unexpected error in training pipeline: {e}") from e
    finally:
        # Finish/cleanup configured logging backend (if any)
        try:
            finish_logging_backend()
        except Exception as e:
            logging.warning(f"Failed to cleanup logging backend: {e}")
        gc.collect()
        torch.cuda.empty_cache()


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

        try:
            test_file_path = Path(data_dir) / "test_ru.json"
            if not test_file_path.exists():
                raise DataProcessingError(f"Test dataset file not found: {test_file_path}")

            with test_file_path.open(encoding="utf-8") as file:
                test_dataset = json.load(file)

            dataset_to_json(
                test_dataset, cfg.testing.output_test_file, cfg.data_preparation.method
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
    try:
        url = "https://dataset.ser13volk.me/dataset_ru"
        dataset_path = Path("../data") / "dataset_ru.json"

        if not dataset_path.exists():
            raise DataProcessingError(f"Dataset file not found: {dataset_path}")

        with dataset_path.open("rb") as f:
            files = {"file": f}
            response = requests.post(
                url, files=files, auth=HTTPBasicAuth("admin", ""), timeout=30
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
