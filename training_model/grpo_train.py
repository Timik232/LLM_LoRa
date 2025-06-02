"""File for training using grpo method"""
import json
import logging
import os
from typing import Callable, List, Optional, Tuple

from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer


def reward_function(prompts: List[str], completions: List[str], **kwargs) -> list:
    """Compute rewards for GRPO training based on action matching.

    Args:
        prompts (list): List of prompts for trl consistency
        completions (list): List of model-generated completions
        kwargs (dict): Dataset row containing 'correct_actions'

    Returns:
        list: List of reward values for each completion
    """
    correct_answer: Optional[str] = kwargs.get("correct_answer")
    logging.debug("Generated completions: %s", completions)
    logging.debug("Correct answer: %s", correct_answer)

    rewards: List[float] = []
    if correct_answer is None:
        # If somehow there's no "correct_answer" in this batch, penalize everything.
        return [-1.0] * len(completions)

    for completion in completions:
        try:
            # Each `completion` is supposed to be a JSON‐string, e.g.
            #   '{"Content": {"Action": "Разговор"}}'
            parsed = json.loads(completion)
            generated_action = parsed.get("Content", {}).get("Action", None)
            if generated_action == correct_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except (json.JSONDecodeError, KeyError):
            # If the model output wasn’t valid JSON → penalize
            rewards.append(-1.0)

    return rewards


def prepare_grpo_data(
    cfg: DictConfig,
) -> Tuple[Dataset, Dataset]:
    """Prepare datasets for GRPO training with prompts and correct actions.

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        Tuple[Dataset, Dataset]: Tuple containing train and validation datasets
    """
    data_dir = os.path.join(get_original_cwd(), cfg.paths.data_dir)

    with open(os.path.join(data_dir, cfg.grpo.val_data), "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    with open(
        os.path.join(data_dir, cfg.grpo.train_data), "r", encoding="utf-8"
    ) as file:
        train_dataset = json.load(file)

    def process_dataset(dataset: dict):
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
        into a HuggingFace Dataset with fields “prompt” and “correct_actions” (or “correct_answer”).
        """
        processed_data = []
        new_dataset = dict(dataset)

        # Remove the top‐level "system" field; we only care about each example under "examples"
        new_dataset.pop("system", None)

        for _, example in new_dataset["examples"].items():
            prompt_dict = example["prompt"]
            history = prompt_dict["History"][0]  # First system message
            available_actions = prompt_dict["AvailableActions"]
            user_input = prompt_dict["UserInput"]
            prompt_str = (
                f"{history}\n"
                f"AvailableActions: {available_actions}\n"
                f"User: {user_input}"
            )

            answer_dict = example["answer"]
            correct_action = answer_dict["Content"]["Action"]

            processed_data.append(
                {"prompt": prompt_str, "correct_answer": correct_action}
            )

        return Dataset.from_list(processed_data)

    train_data = process_dataset(train_dataset)
    val_data = process_dataset(test_dataset)
    return train_data, val_data


def grpo_train(
    model: AutoModel | PeftModel | PreTrainedModel,
    tokenizer: AutoTokenizer | PreTrainedTokenizer,
    cfg: DictConfig,
    data_preparing_func: Optional[Callable],
    reward_func: Callable = reward_function,
) -> int:
    """Execute GRPO training pipeline.

    Args:
        model (AutoModel): LLM model
        tokenizer (AutoTokenizer): LLM tokenizer
        cfg (DictConfig): Configuration object
        data_preparing_func (Callable): Function used to prepare the data. Should return Tuple[Dataset, Dataset]: Tuple containing train and validation datasets
        reward_func (Callable): Reward function for the grpo

    Returns:
        int: Number of global training steps completed
    """
    if data_preparing_func is None:
        train_data, val_data = prepare_grpo_data(cfg)
    else:
        train_data, val_data = data_preparing_func(
            cfg, tokenizer, should_add_prompt=True
        )
    logging.info("GRPO data prepared")
    logging.debug(type(cfg.grpo.max_completion_length))
    if cfg.grpo.max_completion_length == "None":
        cfg.grpo.max_completion_length = tokenizer.model_max_length
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
        report_to="wandb",
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best,
        num_generations=cfg.grpo.num_generations,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        reward_funcs=reward_func,
    )

    trainer.train()
    logging.info("GRPO training completed")

    global_steps = trainer.state.global_step

    return global_steps
