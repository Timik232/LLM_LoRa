"""File for training using grpo method"""

import json
import logging
from collections.abc import Callable
from logging import Logger
from pathlib import Path
from typing import Any

import torch
from bitsandbytes.nn import Int8Params, Params4bit
from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer

from .logging_utils import get_report_to_backend, log_dataset_samples, mlflow_phase_run
from .memory_utils import comprehensive_memory_cleanup, log_memory_usage
from .optimizer_factory import create_optimizer, get_optimizer_config_updates
from .utils import get_generation_config


def convert_model_dtype_for_training(
    model: PreTrainedModel, target_dtype: torch.dtype, logger: Logger = logging
) -> PreTrainedModel:
    """Convert non-quantized model parameters to target dtype.

    Only needed for bfloat16 training. FP16 doesn't need this due to auto-casting.
    Quantized layers (int4/int8) are skipped to preserve quantization.

    Args:
        model: The model to convert (can be PeftModel or base model)
        target_dtype: Target dtype (typically torch.bfloat16)
        logger: Logger instance for info messages

    Returns:
        model: Model with converted parameters
    """

    # Get the base model if it's a PeftModel
    base_model = model.base_model if hasattr(model, "base_model") else model

    converted_params = []
    for name, param in base_model.named_parameters():
        if isinstance(param, Params4bit | Int8Params):
            continue

        # Convert float32 parameters to target dtype (typically lm_head, embeddings)
        if param.dtype == torch.float32:
            param.data = param.data.to(target_dtype)
            converted_params.append(name)

    if converted_params:
        logger.info(
            f"Converted {len(converted_params)} non-quantized parameters to {target_dtype}"
        )
        logger.debug(f"Converted parameters: {converted_params}")
    else:
        logger.info("No float32 parameters found to convert (all already in correct dtype)")

    return model


def validate_grpo_config(cfg: DictConfig) -> bool:
    """Validate GRPO configuration parameters.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if configuration is valid
    """
    required_grpo_params = ["val_data", "train_data", "num_generations"]
    missing_params = [param for param in required_grpo_params if not hasattr(cfg.grpo, param)]

    if missing_params:
        logging.error(f"Missing required GRPO parameters: {missing_params}")
        return False

    # Validate data files exist
    data_dir = Path.cwd() / cfg.paths.data_dir
    train_file = data_dir / cfg.grpo.train_data
    val_file = data_dir / cfg.grpo.val_data

    if not train_file.exists():
        logging.error(f"GRPO training data file not found: {train_file}")
        return False

    if not val_file.exists():
        logging.error(f"GRPO validation data file not found: {val_file}")
        return False

    logging.info("GRPO configuration validation passed")
    return True


def debug_reward_function(test_completions: list[str], correct_answer: str) -> dict[str, Any]:
    """Test the reward function with sample completions for debugging.

    Args:
        test_completions: List of test completions to evaluate
        correct_answer: Expected correct answer

    Returns:
        Dictionary with test results and statistics
    """
    logging.info("Testing reward function...")

    results = reward_function(test_completions, correct_answer=correct_answer)

    stats = {
        "total_completions": len(test_completions),
        "positive_rewards": sum(1 for r in results if r > 0),
        "zero_rewards": sum(1 for r in results if r == 0),
        "negative_rewards": sum(1 for r in results if r < 0),
        "average_reward": sum(results) / len(results) if results else 0,
        "rewards": results,
    }

    logging.info(f"Reward function test results: {stats}")
    return stats


def reward_function(completions: list[str], **kwargs: Any) -> list[float]:
    """Compute rewards for GRPO training based on action matching.

    This function follows TRL's expected signature for reward functions.

    Args:
        completions (List[str]): List of model-generated completions
        **kwargs: Dataset row containing 'correct_answer'

    Returns:
        List[float]: List of reward values for each completion
    """
    correct_answer: str | None = kwargs.get("correct_answer")
    logging.debug("Generated completions: %s", completions)
    logging.debug("Correct answer: %s", correct_answer)

    rewards: list[float] = []
    if correct_answer is None:
        logging.warning(
            "No 'correct_answer' found in batch kwargs, applying penalty to all completions",
        )
        return [-1.0] * len(completions)

    for i, raw_completion in enumerate(completions):
        try:
            # Strip whitespace and handle potential formatting issues
            completion = raw_completion.strip()
            if not completion:
                logging.debug(f"Completion {i} is empty, applying penalty")
                rewards.append(-1.0)
                continue

            # Each `completion` is supposed to be a JSON string, e.g.
            #   '{"Content": {"Action": "Разговор"}}'
            parsed = json.loads(completion)

            # Validate expected structure
            if not isinstance(parsed, dict):
                logging.debug(f"Completion {i} is not a JSON object: {type(parsed)}")
                rewards.append(-1.0)
                continue

            content = parsed.get("Content")
            if not isinstance(content, dict):
                logging.debug(f"Completion {i} missing or invalid 'Content' field")
                rewards.append(-1.0)
                continue

            generated_action = content.get("Action")
            if generated_action is None:
                logging.debug(f"Completion {i} missing 'Action' field in Content")
                rewards.append(-1.0)
                continue

            # Compare actions (case-sensitive exact match)
            if generated_action == correct_answer:
                rewards.append(1.0)
                logging.debug(f"Completion {i} matches correct answer: {generated_action}")
            else:
                rewards.append(0.0)
                logging.debug(
                    f"Completion {i} mismatch - generated:"
                    f" '{generated_action}', expected: '{correct_answer}'"
                )
        except json.JSONDecodeError as e:
            logging.debug(
                f"Completion {i} JSON decode error: {e} - "
                f"Content: '{completion[:100]}...'",
            )
            rewards.append(-1.0)
        except (KeyError, TypeError, AttributeError) as e:
            logging.warning(
                f"Error parsing completion {i} structure: {e} - "
                f"Content: '{completion[:100]}...'"
            )
            rewards.append(-1.0)
        except Exception as e:
            logging.error(
                f"Unexpected error processing completion {i}: {e} - "
                f"Content: '{completion[:100]}...'"
            )
            rewards.append(-1.0)
            raise

    return rewards


def prepare_grpo_data(
    cfg: DictConfig,
) -> tuple[Dataset, Dataset]:
    """Prepare datasets for GRPO training with prompts and correct actions.

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        Tuple[Dataset, Dataset]: Tuple containing train and validation datasets
    """
    data_dir = Path.cwd() / cfg.paths.data_dir

    with (data_dir / cfg.grpo.val_data).open(encoding="utf-8") as file:
        test_dataset = json.load(file)

    with (data_dir / cfg.grpo.train_data).open(encoding="utf-8") as file:
        train_dataset = json.load(file)

    def process_dataset(dataset: dict) -> Dataset:
        """
        Convert a dataset structured as:
          {
            "system": "…",
            "examples": {
              "topic2": {
                "prompt": { … },
                "answer": { … }
              },
              "topic5": { … },
               …
            }
          }
        into a HuggingFace Dataset with fields "prompt"
        and "correct_answer" for GRPO training.
        """
        processed_data = []
        new_dataset = dict(dataset)

        # Extract system instruction if available
        system_instruction = new_dataset.get("system", "")
        new_dataset.pop("system", None)

        for topic_key, example in new_dataset["examples"].items():
            prompt_dict = example["prompt"]
            history = prompt_dict["History"][0] if prompt_dict.get("History") else ""
            available_actions = prompt_dict.get("AvailableActions", [])
            user_input = prompt_dict.get("UserInput", "")

            # Enhanced prompt engineering for GRPO with JSON output instruction
            prompt_parts = []

            # Add system instruction
            if system_instruction:
                prompt_parts.append(f"System: {system_instruction}")

            # Add context and history
            if history:
                prompt_parts.append(f"Context: {history}")

            # Add available actions with clear formatting
            if available_actions:
                actions_str = ", ".join(f'"{action}"' for action in available_actions)
                prompt_parts.append(f"Available Actions: [{actions_str}]")

            # Add user input
            if user_input:
                prompt_parts.append(f"User: {user_input}")

            # Add explicit JSON output instruction
            json_instruction = (
                "Assistant: You must respond with a valid JSON object in the "
                'following format: {"Content": {"Action": "<your_chosen_action>"}}'
            )
            action_instruction = (
                "Choose the most appropriate action from the available actions."
            )
            prompt_parts.append(f"{json_instruction}. {action_instruction}")

            # Join all parts with double newlines for clarity
            prompt_str = "\n\n".join(prompt_parts)

            # Extract correct answer
            answer_dict = example["answer"]
            if isinstance(answer_dict, dict) and "Content" in answer_dict:
                correct_action = answer_dict["Content"].get("Action")
            else:
                logging.warning(f"Invalid answer format in topic {topic_key}: {answer_dict}")
                continue

            if correct_action is None:
                logging.warning(f"Missing Action in answer for topic {topic_key}")
                continue

            processed_data.append(
                {
                    "prompt": prompt_str,
                    "correct_answer": correct_action,
                    "topic": topic_key,  # Add topic for debugging
                },
            )

            logging.debug(f"Processed topic {topic_key}: action={correct_action}")

        logging.info(f"Processed {len(processed_data)} examples from dataset")
        return Dataset.from_list(processed_data)

    train_data = process_dataset(train_dataset)
    val_data = process_dataset(test_dataset)
    return train_data, val_data


def grpo_train(
    model: AutoModel | PeftModel | PreTrainedModel,
    tokenizer: AutoTokenizer | PreTrainedTokenizer,
    cfg: DictConfig,
    data_preparing_func: Callable | None,
    reward_func: Callable = reward_function,
) -> int:
    """Execute GRPO training pipeline.

    Args:
        model (AutoModel): LLM model
        tokenizer (AutoTokenizer): LLM tokenizer
        cfg (DictConfig): Configuration object
        data_preparing_func (Callable): Function used
            to prepare the data. Should return Tuple[Dataset, Dataset]:
            Tuple containing train and validation datasets
        reward_func (Callable): Reward function for the grpo

    Returns:
        int: Number of global training steps completed
    """
    # Validate GRPO configuration
    if not validate_grpo_config(cfg):
        raise ValueError("GRPO configuration validation failed")

    # Test reward function with sample data
    sample_completions = [
        '{\\"Content\\": {\\"Action\\": \\"Разговор\\"}}',  # Correct format
        '{\\"Content\\": {\\"Action\\": \\"Игра\\"}}',  # Different action
        '{\\"Invalid\\": \\"JSON\\"}',  # Wrong structure
        "Not JSON at all",  # Invalid JSON
        "",  # Empty completion
    ]
    debug_reward_function(sample_completions, "Разговор")

    if data_preparing_func is None:
        train_data, val_data = prepare_grpo_data(cfg)
        # Log dataset samples for GRPO data preparation
        log_dataset_samples(train_data, cfg, "grpo_train", "processed")
        log_dataset_samples(val_data, cfg, "grpo_validation", "processed")
    else:
        train_data, val_data = data_preparing_func(cfg, tokenizer, should_add_prompt=True)

    logging.info("GRPO data prepared")
    logging.info(f"Training dataset size: {len(train_data)}")
    logging.info(f"Validation dataset size: {len(val_data)}")

    # Log sample training data for debugging
    if len(train_data) > 0:
        sample = train_data[0]
        logging.debug(f"Sample training data: {sample}")

    # Handle max_completion_length with proper defaults
    if cfg.grpo.max_completion_length is None or cfg.grpo.max_completion_length == "None":
        # Use global generation max_new_tokens as default (256), not model_max_length
        cfg.grpo.max_completion_length = getattr(cfg.generation, "max_new_tokens", 256)

    # Validate that completion length is reasonable
    max_allowed = min(tokenizer.model_max_length - 100, 1024)  # Leave buffer for prompt
    if cfg.grpo.max_completion_length > max_allowed:
        logging.warning(
            f"max_completion_length {cfg.grpo.max_completion_length} exceeds recommended "
            f"maximum {max_allowed}. This may cause CUDA errors during generation. "
            f"Capping to {max_allowed}."
        )
        cfg.grpo.max_completion_length = max_allowed

    logging.info(f"GRPO max_completion_length set to: {cfg.grpo.max_completion_length}")

    # Set up generation config using utility function with GRPO-specific parameters
    generation_config = get_generation_config(cfg, tokenizer, method="grpo")

    # Create custom optimizer if enabled
    custom_optimizer = None
    try:
        custom_optimizer = create_optimizer(model, cfg)
        if custom_optimizer is not None:
            logging.info("Using custom optimizer for GRPO training")
        else:
            logging.info("Using default optimizer for GRPO training")
    except Exception as e:
        raise ValueError(f"Failed to create custom optimizer: {e}") from e

    # Get the appropriate report_to backend based on configuration
    report_to_backend = get_report_to_backend(cfg)

    # Get optimizer configuration updates for custom optimizers
    optimizer_config_updates = get_optimizer_config_updates(cfg)

    # Apply optimizer configuration updates
    optim_name = optimizer_config_updates.get(
        "optim",
        getattr(cfg.training, "optim", "adamw_torch"),
    )

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = False

    if cfg.training.gradient_checkpointing:

        def propagate_gradient_checkpointing(module: nn.Module) -> None:
            """Recursively set gradient_checkpointing=True on all submodules."""
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
            for child in module.children():
                propagate_gradient_checkpointing(child)

        base = model.base_model if hasattr(model, "base_model") else model
        propagate_gradient_checkpointing(base)
        logging.info(
            "Propagated gradient_checkpointing attribute to all submodules (Llama fix)"
        )

    def patch_generate_no_cache(model_obj: nn.Module) -> None:
        if not hasattr(model_obj, "generate"):
            return
        original_generate = model_obj.generate

        def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
            kwargs["use_cache"] = False

            # Ensure consistent dtype behavior during generation by using eval mode
            # This prevents dtype mismatches between model weights (bfloat16) and activations
            was_training = model_obj.training
            model_obj.eval()
            try:
                return original_generate(*args, **kwargs)
            finally:
                if was_training:
                    model_obj.train()

        model_obj.generate = wrapped_generate

    patch_generate_no_cache(model)
    if hasattr(model, "base_model"):
        patch_generate_no_cache(model.base_model)

    logging.info("Patched generate() to enforce use_cache=False and consistent dtype behavior")

    grpo_config = GRPOConfig(
        output_dir=cfg.model.new_model,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        max_completion_length=cfg.grpo.max_completion_length,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        warmup_steps=cfg.training.warmup_steps,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        weight_decay=cfg.training.weight_decay,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to_backend,  # Use dynamic backend selection
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best,
        num_generations=cfg.grpo.num_generations,
        optim=optim_name,  # Use potentially updated optimizer name
        # GRPO-specific parameters
        epsilon=getattr(cfg.grpo, "epsilon", 0.2),
        beta=getattr(cfg.grpo, "beta", 0.01),
        loss_type=getattr(cfg.grpo, "loss_type", "grpo"),
        # Integrate generation parameters from generation_config into GRPOConfig
        temperature=getattr(generation_config, "temperature", 1.0),
        top_p=getattr(generation_config, "top_p", 1.0),
        top_k=getattr(generation_config, "top_k", 50),
    )

    logging.info(
        f"GRPO Config: epsilon={grpo_config.epsilon}, beta={grpo_config.beta}, "
        f"loss_type={grpo_config.loss_type}, num_generations={grpo_config.num_generations}, "
    )

    # Create GRPOTrainer with custom optimizer support
    trainer_kwargs = {
        "model": model,
        "args": grpo_config,
        "train_dataset": train_data,
        "eval_dataset": val_data,
        "processing_class": tokenizer,
        "reward_funcs": reward_func,
    }

    # Only add peft_config if model is not already a PeftModel
    if not isinstance(model, PeftModel):
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
        trainer_kwargs["peft_config"] = peft_config
        logging.info("Added peft_config to GRPOTrainer (model is not already a PeftModel)")
    else:
        logging.info("Model is already a PeftModel, skipping peft_config in GRPOTrainer")

    # Add custom optimizer if available
    if custom_optimizer is not None:
        trainer_kwargs["optimizers"] = (custom_optimizer, None)  # (optimizer, lr_scheduler)

    # Set default dtype based on training configuration to prevent dtype mismatches
    # BF16 is strict about dtype consistency, FP16 is more forgiving
    # This ensures all internal tensors (attention masks, etc.)
    # are created in the correct dtype
    import torch

    torch_dtype = (
        torch.bfloat16
        if cfg.training.bf16
        else (torch.float16 if cfg.training.fp16 else torch.float32)
    )
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    logging.info(f"Set default torch dtype to {torch_dtype} for GRPO training")

    # Convert non-quantized model parameters (lm_head, embeddings) for BF16 only
    # FP16 doesn't need this as PyTorch auto-casting handles float32/float16 mixing
    if cfg.training.bf16:
        model = convert_model_dtype_for_training(model, torch_dtype, logging)
        logging.info("Applied dtype conversion for BF16 training (lm_head, embeddings)")

    trainer = GRPOTrainer(**trainer_kwargs)

    # Memory optimization before training using enhanced utilities
    log_memory_usage("Before GRPO training: ", cfg=cfg)
    comprehensive_memory_cleanup(cfg=cfg)

    # Check if MLflow is enabled for nested run management
    mlflow_enabled = report_to_backend == "mlflow"

    logging.info("Starting GRPO training...")
    try:
        # Use nested MLflow run for GRPO phase to avoid parameter conflicts
        with mlflow_phase_run("grpo", enabled=mlflow_enabled):
            trainer.train()
        logging.info("GRPO training completed")
    finally:
        # Restore original default dtype after training
        torch.set_default_dtype(old_default_dtype)
        logging.info(f"Restored default torch dtype to {old_default_dtype}")

    global_steps = trainer.state.global_step

    # Log final training statistics
    if hasattr(trainer.state, "log_history") and trainer.state.log_history:
        final_log = trainer.state.log_history[-1]
        logging.info(f"Final training metrics: {final_log}")

    # Enhanced cleanup after GRPO training completion
    log_memory_usage("After GRPO training: ", cfg=cfg)
    comprehensive_memory_cleanup(aggressive=True, cfg=cfg)

    return global_steps
