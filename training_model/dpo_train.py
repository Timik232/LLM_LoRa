"""File for training using DPO (Direct Preference Optimization) method"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig, DPOTrainer

from .logging_utils import get_report_to_backend


def validate_dpo_config(cfg: DictConfig) -> bool:
    """Validate DPO configuration parameters.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if configuration is valid
    """
    required_dpo_params = ["val_data", "train_data"]
    missing_params = [param for param in required_dpo_params if not hasattr(cfg.dpo, param)]

    if missing_params:
        logging.error(f"Missing required DPO parameters: {missing_params}")
        return False

    # Validate data files exist
    data_dir = Path(get_original_cwd()) / cfg.paths.data_dir
    train_file = data_dir / cfg.dpo.train_data
    val_file = data_dir / cfg.dpo.val_data

    if not train_file.exists():
        logging.error(f"DPO training data file not found: {train_file}")
        return False

    if not val_file.exists():
        logging.error(f"DPO validation data file not found: {val_file}")
        return False

    logging.info("DPO configuration validation passed")
    return True


def prepare_dpo_data(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    """Prepare datasets for DPO training with preference pairs.

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        Tuple[Dataset, Dataset]: Tuple containing train and validation datasets
    """
    data_dir = Path(get_original_cwd()) / cfg.paths.data_dir

    with (data_dir / cfg.dpo.val_data).open(encoding="utf-8") as file:
        test_dataset = json.load(file)

    with (data_dir / cfg.dpo.train_data).open(encoding="utf-8") as file:
        train_dataset = json.load(file)

    def process_dpo_dataset(dataset: dict) -> Dataset:
        """
        Convert a DPO dataset structured as:
          {
            "system": "…",
            "examples": {
              "topic1": {
                "prompt": "...",
                "chosen": "...",
                "rejected": "..."
              },
              "topic2": { … },
               …
            }
          }
        into a HuggingFace Dataset with fields "prompt", "chosen",
        and "rejected" for DPO training.
        """
        processed_data = []
        new_dataset = dict(dataset)

        # Extract system instruction if available
        system_instruction = new_dataset.get("system", "")
        new_dataset.pop("system", None)

        for topic_key, example in new_dataset["examples"].items():
            prompt = example.get("prompt", "")
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")

            # Enhanced prompt with system instruction
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
            else:
                full_prompt = f"User: {prompt}"

            # Validate required fields
            if not prompt:
                logging.warning(f"Missing prompt for topic {topic_key}")
                continue

            if not chosen:
                logging.warning(f"Missing chosen response for topic {topic_key}")
                continue

            if not rejected:
                logging.warning(f"Missing rejected response for topic {topic_key}")
                continue

            processed_data.append(
                {
                    "prompt": full_prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "topic": topic_key,  # Add topic for debugging
                }
            )

            logging.debug(f"Processed DPO topic {topic_key}")

        logging.info(f"Processed {len(processed_data)} DPO examples from dataset")
        return Dataset.from_list(processed_data)

    train_data = process_dpo_dataset(train_dataset)
    val_data = process_dpo_dataset(test_dataset)
    return train_data, val_data


def dpo_train(
    model: AutoModel | PeftModel | PreTrainedModel,
    tokenizer: AutoTokenizer | PreTrainedTokenizer,
    cfg: DictConfig,
    data_preparing_func: Callable | None = None,
    ref_model: PreTrainedModel | None = None,
) -> int:
    """Execute DPO training pipeline.

    Args:
        model (AutoModel): LLM model
        tokenizer (AutoTokenizer): LLM tokenizer
        cfg (DictConfig): Configuration object
        data_preparing_func (Optional[Callable]): Function used
            to prepare the data. Should return Tuple[Dataset, Dataset]:
            Tuple containing train and validation datasets
        ref_model (Optional[PreTrainedModel]): Reference model for DPO.
            If None, uses the same model as the policy model.

    Returns:
        int: Number of global training steps completed
    """
    # Validate DPO configuration
    if not validate_dpo_config(cfg):
        raise ValueError("DPO configuration validation failed")

    if data_preparing_func is None:
        train_data, val_data = prepare_dpo_data(cfg)
    else:
        train_data, val_data = data_preparing_func(cfg)

    logging.info("DPO data prepared")
    logging.info(f"Training dataset size: {len(train_data)}")
    logging.info(f"Validation dataset size: {len(val_data)}")

    # Log sample training data for debugging
    if len(train_data) > 0:
        sample = train_data[0]
        logging.debug(f"Sample DPO training data: {sample}")

    # Get the appropriate report_to backend based on configuration
    report_to_backend = get_report_to_backend(cfg)

    # Set up DPO configuration
    dpo_config = DPOConfig(
        output_dir=cfg.model.new_model,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        max_length=getattr(cfg.dpo, "max_length", cfg.other.cutoff_len),
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
        # DPO-specific parameters
        beta=getattr(cfg.dpo, "beta", 0.1),
        loss_type=getattr(cfg.dpo, "loss_type", "sigmoid"),
        max_prompt_length=getattr(cfg.dpo, "max_prompt_length", 1024),
        max_target_length=getattr(cfg.dpo, "max_target_length", 1024),
    )

    logging.info(
        f"DPO Config: beta={dpo_config.beta}, "
        f"loss_type={dpo_config.loss_type}, "
        f"max_length={dpo_config.max_length}"
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # Reference model (can be None to use same model)
        args=dpo_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
    )

    # Memory optimization before training
    import gc

    import torch

    torch.cuda.empty_cache()
    gc.collect()

    logging.info("Starting DPO training...")
    trainer.train()
    logging.info("DPO training completed")

    global_steps = trainer.state.global_step

    # Log final training statistics
    if hasattr(trainer.state, "log_history") and trainer.state.log_history:
        final_log = trainer.state.log_history[-1]
        logging.info(f"Final DPO training metrics: {final_log}")

    return global_steps
