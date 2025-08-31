"""Main file for model training."""

import functools
import gc
import json
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import requests
import torch
from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel
from requests.auth import HTTPBasicAuth
from sklearn.model_selection import train_test_split
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
)
from trl import SFTConfig, SFTTrainer

import wandb

from .grpo_train import grpo_train
from .dpo_train import dpo_train
from .logging_config import configure_logging
from .data_preparation import dataset_to_json

_LOGGING_BACKEND: Optional[str] = None


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
            init_kwargs: Dict[str, Any] = {}
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
            import mlflow  # type: ignore

            ml_cfg = getattr(log_dict, "mlflow", None) or {}
            experiment = getattr(ml_cfg, "experiment_name", "default")
            tracking_uri = getattr(ml_cfg, "tracking_uri", None)
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            mlflow.start_run()
            logging.info("Initialized mlflow logging backend")
        except Exception as e:
            logging.warning(
                f"Failed to initialize mlflow: {e}. Falling back to 'none'."
            )
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
            import mlflow  # type: ignore

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
    current_dir = os.getcwd()
    os.chdir(destination)
    try:
        yield
    finally:
        os.chdir(current_dir)


def generate_prompt(tokenizer: AutoTokenizer, data_point: Dict[str, str]) -> str:
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
    tokenizer: AutoTokenizer | Callable, cutoff_len: int, prompt: str
) -> Dict[str, torch.Tensor]:
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
    data_point: Dict[str, str],
    tokenizer: AutoTokenizer,
    cutoff: int,
    should_add_prompt: bool = False,
) -> Dict[str, str] | Dict[str, Tensor]:
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
    else:
        # Standard SFT prompt
        full_prompt = generate_prompt(tokenizer, data_point)
        tokenized_full_prompt = tokenize(
            tokenizer,
            cutoff,
            full_prompt,
        )
        return tokenized_full_prompt


def generate_grpo_prompt(tokenizer: AutoTokenizer, data_point: Dict[str, str]) -> str:
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
) -> Tuple[Dataset, Dataset]:
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
    """
    # Use current working directory or the configured data directory
    try:
        base = Path(get_original_cwd()) / cfg.paths.data_dir
    except ValueError:
        # Fallback if Hydra context is not available
        base = Path(os.getcwd()) / cfg.paths.data_dir
    if not cfg.testing.use_separate_files:
        single_path = base / cfg.paths.train_data
        raw = json.loads(single_path.read_text(encoding="utf-8"))
        all_items: List[dict]
        if isinstance(raw, dict) and "examples" in raw:
            all_items = raw["examples"]
        else:
            raise ValueError(f"Unrecognized JSON structure in {single_path}")
        list_train, list_test = train_test_split(
            all_items,
            test_size=cfg.testing.test_split_ratio,
            shuffle=True,
            random_state=cfg.training.seed,
        )
        train_dataset = {"system": raw["system"], "examples": list_train}
        test_dataset = {"system": raw["system"], "examples": list_test}

    else:
        train_path = base / cfg.paths.train_file
        test_path = base / cfg.paths.test_file

        train_dataset = json.loads(train_path.read_text(encoding="utf-8"))
        test_dataset = json.loads(test_path.read_text(encoding="utf-8"))

    # Use temporary directory for JSON files
    with TemporaryDirectory() as temp_dir:
        train_json = os.path.join(temp_dir, "train.json")
        test_json = os.path.join(temp_dir, "test.json")
        dataset_to_json(train_dataset, train_json, cfg.data_preparation.method)
        dataset_to_json(test_dataset, test_json, cfg.data_preparation.method)

        from datasets import load_dataset

        dataset = load_dataset(
            "json", data_files={"train": train_json, "test": test_json}
        )

        tokenize_partial = functools.partial(
            generate_and_tokenize_prompt,
            tokenizer=tokenizer,
            cutoff=cfg.other.cutoff_len,
            should_add_prompt=should_add_prompt,
        )
        train_data = dataset["train"].map(tokenize_partial)
        val_data = dataset["test"].map(tokenize_partial)

    return train_data, val_data


def model_merge_for_converting(cfg: DictConfig, steps: int, save_path: str) -> None:
    """Merge base model with adapter weights and save the result.

    Args:
        cfg: Configuration object.
        steps: Training step number for checkpoint selection.
        save_path: Path to save merged model.
    """
    model_path = cfg.model.model_name
    adapter_path = f"{cfg.model.new_model}/checkpoint-{steps}"
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Model merged")


def setup_model_and_tokenizer(cfg: DictConfig) -> Tuple[Any, AutoTokenizer]:
    """Set up model and tokenizer with quantization config.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (model, tokenizer).
    """
    torch_dtype = (
        getattr(torch, cfg.model.torch_dtype)
        if isinstance(cfg.model.torch_dtype, str)
        else cfg.model.torch_dtype
    )
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
        pass
        # model = Gemma3nForConditionalGeneration.from_pretrained(
        #     cfg.model.model_name,
        #     quantization_config=bnb_config,
        #     device_map="auto",
        #     attn_implementation=cfg.model.attn_implementation,
        #     use_cache=False,
        # )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=cfg.model.attn_implementation,
            use_cache=False,
        )
    logging.info("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.pad_token = "<|pad|>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.pad_token
            )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def run_sft_training(
    model: Any,
    tokenizer: AutoTokenizer,
    cfg: DictConfig,
    train_data: Dataset,
    val_data: Dataset,
) -> Tuple[int, float]:
    """Run SFT training phase.

    Args:
        model: The model to train.
        tokenizer: The tokenizer.
        cfg: Configuration object.
        train_data: Training dataset.
        val_data: Validation dataset.

    Returns:
        Tuple of (global_steps, eval_loss).
    """
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

    logging.info("Starting SFT training phase...")
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
        report_to="none",
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best,
    )
    if cfg.training.use_optuna_optimize:
        sft_config.run_name = f"{sft_config.run_name}_optuna"

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

    # Clean up SFT trainer to free memory before GRPO
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return global_steps, eval_loss


def train(cfg: DictConfig) -> dict[str, int | Any]:  # noqa: C901
    """Execute full training pipeline.

    Args:
        cfg: Configuration object.

    Returns:
        A dict with 'global_steps' and 'eval_loss'.
    """
    model, tokenizer = setup_model_and_tokenizer(cfg)
    train_data, val_data = data_preparation(cfg, tokenizer)
    logging.info("Data prepared")

    # Initialize training state tracking
    global_steps = 0
    eval_loss = 0.0
    training_completed = False

    # Phase 1: Supervised Fine-Tuning (SFT)
    if cfg.training.use_sft:
        global_steps, eval_loss = run_sft_training(
            model, tokenizer, cfg, train_data, val_data
        )
        training_completed = True

    # Phase 2: Group Relative Policy Optimization (GRPO)
    if cfg.training.use_grpo:
        logging.info("Starting GRPO training phase...")

        # If both SFT and GRPO are enabled, load the best checkpoint from SFT
        if cfg.training.use_sft and training_completed:
            logging.info("Loading SFT checkpoint for GRPO training...")
            checkpoint_path = os.path.join(
                cfg.model.new_model, f"checkpoint-{global_steps}"
            )
            if os.path.exists(checkpoint_path):
                # Load the adapter weights into the model
                model = PeftModel.from_pretrained(model, checkpoint_path)
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

    # Phase 3: Direct Preference Optimization (DPO)
    if cfg.training.use_dpo:
        logging.info("Starting DPO training phase...")

        # If previous training phases completed, load the best checkpoint
        if training_completed:
            logging.info("Loading previous checkpoint for DPO training...")
            checkpoint_path = os.path.join(
                cfg.model.new_model, f"checkpoint-{global_steps}"
            )
            if os.path.exists(checkpoint_path):
                # Load the adapter weights into the model
                model = PeftModel.from_pretrained(model, checkpoint_path)
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

    # Validate training completion
    if (
        not cfg.training.use_grpo
        and not cfg.training.use_sft
        and not cfg.training.use_dpo
    ):
        logging.warning("Model training not configured")
        return {"global_steps": 0, "eval_loss": 0.0}
    elif not training_completed:
        logging.error("Training was configured but did not complete successfully")
        return {"global_steps": 0, "eval_loss": 0.0}
    else:
        logging.info("Model training completed successfully")

    # Merge adapter weights from the final checkpoint
    final_checkpoint_path = os.path.join(
        cfg.model.new_model, f"checkpoint-{global_steps}"
    )
    if os.path.exists(final_checkpoint_path):
        merge_adapter_from_checkpoint(
            base_model_name=cfg.model.model_name,
            adapter_dir=final_checkpoint_path,
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
    logging.info("Model saved")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"global_steps": global_steps, "eval_loss": eval_loss}


def merge_adapter_from_checkpoint(
    base_model_name: str,
    adapter_dir: str,
    save_path: str,
    device: str = "cpu",
) -> None:
    """Merge a LoRA adapter checkpoint into the base model and save merged HF model.

    Args:
        base_model_name: HF model identifier or local path to base model.
        adapter_dir: Directory containing the PEFT adapter (e.g. checkpoints).
        save_path: Directory where merged model will be saved.
        device: Device to load model on; use 'cpu' to conserve GPU memory.
    """
    # load base on CPU to avoid GPU OOM, then attach adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": device} if device != "auto" else "auto",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(
        base_model, adapter_dir, device_map={"": device}
    )
    # merge LoRA weights into base weights and free adapter memory
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def convert_to_gguf(
    model_path: str | bytes,
    outfile: str | bytes,
    python_exe: str,
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
    """
    try:
        llama_cpp_dir = os.path.abspath(cfg.paths.llama_cpp_dir)
        conversion_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")

        if not os.path.isdir(llama_cpp_dir):
            raise FileNotFoundError(f"llama.cpp directory not found: {llama_cpp_dir}")
        if not os.path.exists(conversion_script):
            raise FileNotFoundError(f"Conversion script missing: {conversion_script}")

        model_path = os.path.normpath(os.path.abspath(model_path))
        outfile = os.path.normpath(os.path.abspath(outfile))
        python_exe = os.path.normpath(cfg.paths.venv_python_path)

        subprocess.run(
            [
                python_exe,
                conversion_script,
                model_path,
                "--outfile",
                outfile,
                "--outtype",
                outtype,
            ],
            check=True,
            cwd=llama_cpp_dir,
        )
    except subprocess.CalledProcessError:
        logging.error(
            f"GGUF conversion failed. Check:\n"
            f"- llama.cpp exists at {cfg.paths.llama_cpp_dir}\n"
            f"- Conversion script exists: "
            f"{os.path.join(cfg.paths.llama_cpp_dir, 'convert_hf_to_gguf.py')}\n"
            f"- Python executable: {python_exe}\n"
            f"- Model path: {model_path}"
        )


def convert_to_rkllm(
    model_path: str | bytes,
    output_dir: str | bytes,
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
            "rkllm-toolkit not found. Please install with: pip install rkllm-toolkit"
        ) from e

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename based on configuration
    model_name = os.path.basename(model_path.rstrip("/"))
    output_filename = f"{model_name}_{target_platform}_{quantization}.rkllm"
    output_path = os.path.join(output_dir, output_filename)

    logging.info("Converting model to RKLLM format...")
    logging.info(f"  Source: {model_path}")
    logging.info(f"  Target platform: {target_platform}")
    logging.info(f"  Quantization: {quantization}")
    logging.info(f"  Output: {output_path}")

    try:
        # Initialize RKLLM converter
        rkllm = RKLLM()

        # Load model configuration
        ret = rkllm.load_huggingface(
            model=model_path, target_platform=target_platform, num_npu_core=num_npu_core
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
        ret = rkllm.export_rkllm(output_path)

        if ret != 0:
            raise RuntimeError(f"Failed to export RKLLM model to: {output_path}")

        # Verify the output file exists
        if not os.path.exists(output_path):
            raise RuntimeError(f"RKLLM model file was not created: {output_path}")

        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        logging.info("RKLLM conversion completed successfully")
        logging.info(f"  Output file: {output_path}")
        logging.info(f"  File size: {file_size:.2f} MB")

        return output_path

    except Exception as e:
        logging.error(f"RKLLM conversion failed: {e}")
        # Clean up partial files
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"RKLLM conversion failed: {e}") from e

    finally:
        # Ensure RKLLM resources are cleaned up
        try:
            if "rkllm" in locals():
                del rkllm
        except Exception as cleanup_error:
            logging.warning(f"Failed to cleanup RKLLM resources: {cleanup_error}")


def rkllm_quantize(
    model_path: str,
    output_path: str,
    quantization: str = "w8a8",
    target_platform: str = "rk3588",
    **kwargs,
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
    """
    try:
        output_dir = os.path.dirname(output_path)
        result_path = convert_to_rkllm(
            model_path=model_path,
            output_dir=output_dir,
            target_platform=target_platform,
            quantization=quantization,
            **kwargs,
        )

        # If output filename is different, rename the file
        if result_path != output_path and os.path.exists(result_path):
            shutil.move(result_path, output_path)
            logging.info(f"RKLLM model moved to: {output_path}")

        return True

    except Exception as e:
        logging.error(f"RKLLM quantization failed: {e}")
        return False


def quantize_model(
    model_path: str,
    outfile: str,
    qtype: str = "q4_0",
    llama_cpp_path: str = ".",
    quantized_path: str = "llama-quantize.exe",
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
    """
    llama_cpp_dir = os.path.abspath(llama_cpp_path)
    llama_quantize_path = os.path.join(llama_cpp_dir, quantized_path)
    model_path = os.path.abspath(model_path)
    outfile = os.path.abspath(outfile)
    if not os.path.exists(llama_quantize_path):
        logging.error(f"Error: llama-quantize.exe not found at {llama_quantize_path}")
        return False

    logging.info("Trying to quantize model...")
    command = [llama_quantize_path, model_path, outfile, qtype]
    logging.info(f"Running command: {command}")
    try:
        process = subprocess.run(
            command, check=True, capture_output=True, cwd=llama_cpp_dir
        )
        logging.info("Model quantized")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        return False

    logging.info(f"Command output: {process.stdout.decode()}")
    logging.info(
        f"Command stderr: {process.stderr.decode() if process.stderr else 'No stderr output.'}"
    )
    return True


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
    """
    destination_path = os.path.join(destination, gguf_directory, file)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(os.path.join(os.getcwd(), file), destination_path)


def train_pipeline(cfg: DictConfig) -> Dict[str, Any]:
    """Execute complete training
    pipeline including conversion and quantization.

    Args:
        cfg: Configuration object.

    Raises:
        RuntimeError: If quantization step fails.

    Returns:
        A dict with 'eval_loss'.
    """
    # initialize optional experiment logging backend (wandb / mlflow / none)
    init_logging_backend(cfg)
    try:
        result = train(cfg)
        merge_adapter_from_checkpoint(
            base_model_name=cfg.model.model_name,
            adapter_dir=os.path.join(
                cfg.model.new_model, f"checkpoint-{result['global_steps']}"
            ),  # where trainer saved checkpoint-<steps>
            save_path=cfg.paths.output_dir,
            device="cpu",  # merge on CPU to avoid OOM
        )
        steps = result["global_steps"]
        eval_loss = result["eval_loss"]

        with TemporaryDirectory() as merged_model_dir:
            model_merge_for_converting(cfg, steps, merged_model_dir)

            outfile = cfg.model.outfile

            convert_to_gguf(
                model_path=merged_model_dir,
                outfile=os.path.join(merged_model_dir, outfile),
                python_exe=cfg.paths.venv_python_path,
                outtype="f16",
                cfg=cfg,
            )
            logging.info(f"Converted to GGUF: {outfile}")

            quantized_file = outfile
            if quantize_model(
                model_path=os.path.join(merged_model_dir, outfile),
                outfile=quantized_file,
                qtype=cfg.model.quant.qtype,
                llama_cpp_path=os.path.abspath(cfg.paths.llama_cpp_dir),
                quantized_path=cfg.paths.quantized_path,
            ):
                copy_data(
                    quantized_file,
                    cfg.model.quant.gguf_dir,
                    cfg.paths.final_weights_path,
                )
                if os.path.exists(quantized_file):
                    os.remove(quantized_file)
                    logging.info(f"Removed intermediate file: {quantized_file}")
            else:
                raise RuntimeError("Quantization failed")

            # RKLLM conversion (if enabled)
            if getattr(cfg.model, "rkllm", {}).get("enabled", False):
                logging.info("Starting RKLLM conversion...")
                try:
                    rkllm_config = cfg.model.rkllm
                    rkllm_output_dir = os.path.join(
                        cfg.paths.output_dir, rkllm_config.output_dir
                    )

                    rkllm_model_path = convert_to_rkllm(
                        model_path=cfg.paths.output_dir,  # Use the merged model directory
                        output_dir=rkllm_output_dir,
                        target_platform=rkllm_config.target_platform,
                        quantization=rkllm_config.quantization,
                        do_parallelize=rkllm_config.get("do_parallelize", False),
                        hybrid_quantization=rkllm_config.get(
                            "hybrid_quantization", False
                        ),
                        num_npu_core=rkllm_config.get("num_npu_core", 1),
                    )

                    logging.info(f"RKLLM conversion completed: {rkllm_model_path}")

                except Exception as e:
                    logging.error(f"RKLLM conversion failed: {e}")
                    # Don't raise error - RKLLM conversion is optional
                    logging.info("Continuing without RKLLM conversion...")
            else:
                logging.info("RKLLM conversion disabled in configuration")

        logging.info("Training pipeline completed")
        return {"eval_loss": eval_loss}

    finally:
        # finish/cleanup configured logging backend (if any)
        finish_logging_backend()
        gc.collect()
        torch.cuda.empty_cache()


def main_train(data_dir: str, cfg: DictConfig) -> Dict[str, Any]:
    """Main training entry point with dataset processing.

    Args:
        data_dir: Directory containing training data.
        cfg: Configuration object.

    Returns:
        Result dictionary from training pipeline.
    """
    result = train_pipeline(cfg)
    with open(os.path.join(data_dir, "test_ru.json"), "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    dataset_to_json(
        test_dataset, cfg.testing.output_test_file, cfg.data_preparation.method
    )
    return result


def post_new_dataset() -> None:
    """Upload new dataset version to remote server."""
    url = "https://dataset.ser13volk.me/dataset_ru"
    with open(os.path.join("../data", "dataset_ru.json"), "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files, auth=HTTPBasicAuth("admin", ""))
    logging.info(response.json())


if __name__ == "__main__":
    configure_logging(logging.DEBUG)
    post_new_dataset()
