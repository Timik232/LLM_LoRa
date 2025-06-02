"""Main file for model training"""

import functools
import gc
import json
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Generator, Tuple

# from typing import Any, Dict, List, Union
# import numpy as np
import requests
import torch
from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, get_peft_model
from requests.auth import HTTPBasicAuth
from torch import Tensor

# from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

import wandb

from .grpo_train import grpo_train
from .logging_config import configure_logging
from .utils import dataset_to_json, tokens_init


@contextmanager
def change_dir(destination: str) -> Generator[None, None, None]:
    """Context manager for temporarily changing the working directory.

    Args:
        destination (str): Path to the target directory

    Yields:
        None: Enters the target directory during context execution
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
        tokenizer (AutoTokenizer): Hugging Face tokenizer
        data_point (Dict[str, str]): Dictionary containing system, user and bot messages

    Returns:
        str: Formatted chat prompt
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
        tokenizer (AutoTokenizer): Hugging Face tokenizer
        cutoff_len (int): Maximum sequence length
        prompt (str): Text to tokenize

    Returns:
        Dict[str, torch.Tensor]: Tokenized output dictionary
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
        data_point (Dict[str, str]): Dictionary containing conversation data
        tokenizer (AutoTokenizer): Hugging Face tokenizer
        cutoff (int): Maximum sequence length
        should_add_prompt (bool): used for grpo, when
            needed dict with keyword "prompt" returned

    Returns:
        Dict[str, torch.Tensor]: Tokenized prompt dictionary
    """
    full_prompt = generate_prompt(tokenizer, data_point)
    tokenized_full_prompt = tokenize(
        tokenizer,
        cutoff,
        full_prompt,
    )
    if should_add_prompt:
        return {"prompt": full_prompt, "correct_answer": data_point["bot"]}
    else:
        return tokenized_full_prompt


def data_preparation(
    cfg: DictConfig, tokenizer: AutoTokenizer, should_add_prompt: bool = False
) -> Tuple[Dataset, Dataset]:
    """Prepare and preprocess training and validation datasets.

    Args:
        cfg (DictConfig): Configuration object
        tokenizer (AutoTokenizer): Hugging Face tokenizer
        should_add_prompt (bool): If True, returns dict with "prompt" key for grpo training

    Returns:
        Tuple[Dataset, Dataset]: Tuple containing train and validation datasets
    """
    data_dir = os.path.join(get_original_cwd(), cfg.paths.data_dir)
    with open(os.path.join(data_dir, "test_ru.json"), "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    with open(
        os.path.join(get_original_cwd(), cfg.model.dataset_name), "r", encoding="utf-8"
    ) as file:
        train_dataset = json.load(file)

    # Use temporary directory for JSON files
    with TemporaryDirectory() as temp_dir:
        train_json = os.path.join(temp_dir, "train.json")
        test_json = os.path.join(temp_dir, "test.json")
        dataset_to_json(train_dataset, train_json)
        dataset_to_json(test_dataset, test_json)

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
        cfg (DictConfig): Configuration object
        steps (int): Training step number for checkpoint selection
        save_path (str): Path to save merged model
    """
    model_path = cfg.model.model_name
    adapter_path = f"{cfg.model.new_model}/checkpoint-{steps}"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Model merged")


def train(cfg: DictConfig) -> int:
    """Execute full training pipeline.

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        int: Number of global training steps completed
    """
    tokens_init(cfg)
    torch_dtype = (
        getattr(torch, cfg.model.torch_dtype)
        if isinstance(cfg.model.torch_dtype, str)
        else cfg.model.torch_dtype
    )
    if not cfg.model.use_8bit:
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
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        quantization_config=bnb_config,
        device_map="auto",
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
    peft_config = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
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
    model = get_peft_model(model, peft_config)
    train_data, val_data = data_preparation(cfg, tokenizer)
    logging.info("Data prepared")
    global_steps = 0
    if cfg.training.use_sft:
        sft_config = SFTConfig(
            output_dir=cfg.model.new_model,
            max_seq_length=cfg.training.max_seq_length,
            dataset_kwargs={"skip_prepare_dataset": True},
            packing=False,
            run_name=cfg.model.new_model,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            # max_steps=cfg.model.train_steps,
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
            report_to="wandb",
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=cfg.training.load_best,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=sft_config,
        )
        trainer.train()
        global_steps: int = trainer.state.global_step
    if cfg.training.use_grpo:
        grpo_train(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            data_preparing_func=None,
        )
    if not cfg.training.use_grpo and not cfg.training.use_sft:
        logging.warning("Model training not configured")
    else:
        logging.info("Model trained")

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(cfg.paths.output_dir)
    tokenizer.save_pretrained(cfg.paths.output_dir)
    logging.info("Model saved")
    del model, merged_model
    gc.collect()
    torch.cuda.empty_cache()
    return global_steps


def convert_to_gguf(
    model_path: str,
    outfile: str,
    python_exe: str,
    outtype: str,
    cfg: DictConfig,
) -> None:
    """Convert Hugging Face model to GGUF format.

    Args:
        model_path (str): Path to input model directory
        outfile (str): Output file path
        python_exe (str): Python executable path
        outtype (str): Output type specification
        cfg (DictConfig): Configuration object

    Raises:
        FileNotFoundError: If required paths are missing
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
            f"- Conversion script exists: {os.path.join(cfg.paths.llama_cpp_dir, 'convert_hf_to_gguf.py')}\n"
            f"- Python executable: {python_exe}\n"
            f"- Model path: {model_path}"
        )


def quantize_model(
    model_path: str,
    outfile: str,
    qtype: str = "q4_0",
    llama_cpp_path: str = ".",
    quantized_path: str = "llama-quantize.exe",
) -> bool:
    """Quantize GGUF model using llama.cpp quantizer.

    Args:
        model_path (str): Path to input GGUF model
        outfile (str): Path for quantized output
        qtype (str): Quantization type (default: q4_0)
        llama_cpp_path (str): Path to llama.cpp directory
        quantized_path (str): Name of quantizer executable

    Returns:
        bool: True if quantization succeeded, False otherwise
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
        file (str): Source file name
        gguf_directory (str): Version subdirectory
        destination (str): Root destination directory
    """
    destination_path = os.path.join(destination, gguf_directory, file)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(os.path.join(os.getcwd(), file), destination_path)


def train_pipeline(cfg: DictConfig) -> None:
    """Execute complete training pipeline including conversion and quantization.

    Args:
        cfg (DictConfig): Configuration object

    Raises:
        RuntimeError: If quantization step fails
    """
    try:
        steps = train(cfg)

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
                qtype=cfg.model.qtype,
                llama_cpp_path=os.path.abspath(cfg.paths.llama_cpp_dir),
                quantized_path=cfg.paths.quantized_path,
            ):
                copy_data(
                    quantized_file,
                    cfg.model.gguf_directory,
                    cfg.paths.final_weights_path,
                )
                if os.path.exists(quantized_file):
                    os.remove(quantized_file)
                    logging.info(f"Removed intermediate file: {quantized_file}")
            else:
                raise RuntimeError("Quantization failed")

        logging.info("Training pipeline completed")

    finally:
        wandb.finish()
        logging.info("Wandb finished")
        gc.collect()
        torch.cuda.empty_cache()


def main_train(data_dir: str, cfg: DictConfig) -> None:
    """Main training entry point with dataset processing.

    Args:
        data_dir (str): Directory containing training data
        cfg (DictConfig): Configuration object
    """
    train_pipeline(cfg)
    with open(os.path.join(data_dir, "test_ru.json"), "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    dataset_to_json(test_dataset, cfg.testing.output_test_file)


def post_new_dataset() -> None:
    """Upload new dataset version to remote server."""
    url = "https://dataset.ser13volk.me/dataset_ru"
    with open(os.path.join("../data", "dataset_ru.json"), "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files, auth=HTTPBasicAuth("admin", ""))
    logging.info(response.json())


if __name__ == "__main__":
    configure_logging(logging.DEBUG)
    main_train()
    # post_new_dataset()
