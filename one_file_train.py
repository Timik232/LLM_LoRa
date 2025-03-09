import functools
import gc
import json
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager

# from typing import Any, Dict, List, Union
import hydra

# import numpy as np
import requests
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, get_peft_model
from requests.auth import HTTPBasicAuth

# from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

import wandb
from logging_config import configure_logging
from testing_model.test import test_via_lmstudio
from utils import dataset_to_json, tokens_init


@contextmanager
def change_dir(destination: str):
    """Контекстный менеджер для временной смены рабочей директории."""
    current_dir = os.getcwd()
    os.chdir(destination)
    try:
        yield
    finally:
        os.chdir(current_dir)


def generate_prompt(tokenizer, data_point: dict) -> str:
    """generate template for the model"""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": data_point["system"]},
            {"role": "user", "content": data_point["user"]},
            {"role": "assistant", "content": data_point["bot"]},
        ],
        tokenize=False,
    )


def tokenize(tokenizer, cutoff_len: int, prompt: str):
    return tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors=None,
        add_special_tokens=True,
    )


def generate_and_tokenize_prompt(
    data_point,
    tokenizer,
    cutoff: int,
):
    full_prompt = generate_prompt(tokenizer, data_point)
    tokenized_full_prompt = tokenize(
        tokenizer,
        cutoff,
        full_prompt,
    )
    return tokenized_full_prompt


def data_preparation(cfg: DictConfig, tokenizer: AutoTokenizer):
    data_dir = os.path.join(get_original_cwd(), cfg.paths.data_dir)
    with open(os.path.join(data_dir, "test_ru.json"), "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    with open(
        os.path.join(get_original_cwd(), cfg.model.dataset_name), "r", encoding="utf-8"
    ) as file:
        train_dataset = json.load(file)
    dataset_to_json(train_dataset, "train.json")
    dataset_to_json(test_dataset, "test.json")

    from datasets import load_dataset

    dataset = load_dataset(
        "json", data_files={"train": "train.json", "test": "test.json"}
    )
    tokenize_partial = functools.partial(
        generate_and_tokenize_prompt,
        tokenizer=tokenizer,
        cutoff=cfg.other.cutoff_len,
    )
    train_data = dataset["train"].map(tokenize_partial)
    val_data = dataset["test"].map(tokenize_partial)
    return train_data, val_data


def model_merge_for_converting(cfg: DictConfig, steps: int = 60):
    model_path = cfg.model.model_name
    adapter_path = f"{cfg.model.new_model}/checkpoint-{steps}"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(cfg.paths.merged_model_path)
    tokenizer.save_pretrained(cfg.paths.merged_model_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Model merged")


def train(cfg: DictConfig):
    tokens_init(cfg)
    torch_dtype = (
        getattr(torch, cfg.model.torch_dtype)
        if isinstance(cfg.model.torch_dtype, str)
        else cfg.model.torch_dtype
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,  # Основное изменение
    #     llm_int8_threshold=6.0,  # Опционально: порог для квантизации
    #     torch_dtype=torch_dtype,  # Сохраняем указание типа данных
    # )
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
    logging.info("Model trained")
    global_steps = trainer.state.global_step
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(cfg.paths.output_dir)
    tokenizer.save_pretrained(cfg.paths.output_dir)
    logging.info("Model saved")
    del model, merged_model
    gc.collect()
    torch.cuda.empty_cache()
    return global_steps


def convert_to_gguf(model_path: str, outfile: str, python_exe="python", outtype="f16"):
    subprocess.run(
        [
            python_exe,
            "convert_hf_to_gguf.py",
            f"../{model_path}",
            "--outfile",
            outfile,
            "--outtype",
            outtype,
        ],
        check=True,
    )


def quantize_model(model_path: str, outfile: str, qtype="q4_0", llama_cpp_path="."):
    """
    Квантование модели с помощью инструмента llama-quantize.exe.
    """
    llama_quantize_path = os.path.join(llama_cpp_path, "llama-quantize.exe")
    if not os.path.exists(llama_quantize_path):
        logging.error(f"Error: llama-quantize.exe not found at {llama_quantize_path}")
        return False

    command = [llama_quantize_path, model_path, outfile, qtype]
    try:
        process = subprocess.run(
            command, check=True, capture_output=True, cwd=llama_cpp_path
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        return False

    logging.info(f"Command output: {process.stdout.decode()}")
    logging.info(
        f"Command stderr: {process.stderr.decode() if process.stderr else 'No stderr output.'}"
    )
    return True


def copy_data(file: str, version="v1", destination=r"T:\lm-studio\models\game-model"):
    destination_path = os.path.join(destination, version, file)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(os.path.join(os.getcwd(), file), destination_path)


def train_pipeline(cfg: DictConfig):
    steps = train(cfg)
    merged_model_path = "merged_model_fp16"
    model_merge_for_converting(cfg, steps)
    with change_dir("llama.cpp"):
        venv_python_path = r"T:\projects\LLM_LoRa\venv\Scripts\python.exe"
        outfile = cfg.model.outfile
        convert_to_gguf(
            merged_model_path, outfile, python_exe=venv_python_path, outtype="f16"
        )
        logging.info("Converted")
        new_file_name = outfile[:-5] + f"{cfg.model.quant_postfix}.gguf"
        quantize_model(outfile, new_file_name, qtype=cfg.model.qtype)
        copy_data(new_file_name, version="v5")
    logging.info("Train Done")
    wandb.finish()


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    train_pipeline(cfg)
    data_dir = os.path.join(get_original_cwd(), cfg.paths.data_dir)
    with open(os.path.join(data_dir, "test_ru.json"), "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    dataset_to_json(test_dataset, "test.json")
    next = input("Загрузите модель и нажмите Enter. Для пропуска введите 0\n")
    if next == "0":
        return
    test_via_lmstudio(
        cfg,
        test_dataset=os.path.join(data_dir, "dataset_ru.json"),
        test_file="train.json",
    )
    test_via_lmstudio(
        cfg,
        test_dataset=os.path.join(data_dir, "dataset_ru.json"),
        test_file="train.json",
    )


def post_new_dataset():
    url = "https://dataset.ser13volk.me/dataset_ru"
    with open(os.path.join("data", "dataset_ru.json"), "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files, auth=HTTPBasicAuth("admin", ""))
    logging.info(response.json())


if __name__ == "__main__":
    configure_logging()
    main()
    # post_new_dataset()
