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
from .logging_config import configure_logging
from .vika_utils import dataset_to_json

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
    backend = getattr(cfg, "logging_backend", None)
    if backend is None:
        backend = "wandb" if getattr(cfg, "wandb", None) else "none"
    backend = str(backend).lower()
    _LOGGING_BACKEND = backend

    if backend == "wandb":
        try:
            wb_cfg = getattr(cfg, "wandb", None) or {}
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

            ml_cfg = getattr(cfg, "mlflow", None) or {}
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
        should_add_prompt: used for grpo, when needed dict with keyword "prompt" returned.

    Returns:
        Tokenized prompt dictionary or a dict containing "prompt" when should_add_prompt=True.
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
    base = Path(get_original_cwd()) / cfg.paths.data_dir
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


def train(cfg: DictConfig) -> dict[str, int | Any]:
    """Execute full training pipeline.

    Args:
        cfg: Configuration object.

    Returns:
        A dict with 'global_steps' and 'eval_loss'.
    """
    # tokens_init(cfg)
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
    # model = get_peft_model(model, peft_config)
    train_data, val_data = data_preparation(cfg, tokenizer)
    logging.info("Data prepared")
    global_steps = 0
    eval_loss = 0.0
    if cfg.training.use_sft:
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
        global_steps: int = trainer.state.global_step
        eval_results = trainer.evaluate()
        eval_loss = eval_results["eval_loss"]
        logging.info(f"Evaluation loss: {eval_loss}")
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

    merge_adapter_from_checkpoint(
        base_model_name=cfg.model.model_name,
        adapter_dir=os.path.join(cfg.model.new_model, f"checkpoint-{global_steps}"),
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
    model_path: str,
    outfile: str,
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
        # result = train(cfg)
        result = {"global_steps": 732, "eval_loss": 0.79}
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
    dataset_to_json(test_dataset, cfg.testing.output_test_file)
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
    main_train()
    # post_new_dataset()
