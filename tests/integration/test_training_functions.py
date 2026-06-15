"""Integration tests for training_model.one_file_train functions.

Uses built-in monkeypatch fixture (no pytest-mock dependency).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

import training_model.one_file_train as oft
from training_model.one_file_train import (
    change_dir,
    convert_to_gguf,
    copy_data,
    data_preparation,
    generate_and_tokenize_prompt,
    generate_prompt,
    model_merge_for_converting,
    quantize_model,
    tokenize,
)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests and clean up afterwards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def tokenizer() -> AutoTokenizer:
    """Return a GPT-2 tokenizer patched to behave like training code expects.

    Adds a pad token when missing and a minimal `apply_chat_template` function.
    """
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        pad = tok.eos_token or "<|pad|>"
        tok.add_special_tokens({"pad_token": pad})
        tok.pad_token = pad
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.convert_tokens_to_ids(pad)

    def apply_chat_template(messages: list, tokenize: bool = False) -> str:
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        return "\n".join(parts)

    tok.apply_chat_template = apply_chat_template  # type: ignore[attr-defined]
    return tok


@pytest.fixture
def test_config() -> DictConfig:
    """Return a config object matching fields used by one_file_train functions."""
    return OmegaConf.create(
        {
            "paths": {
                "data_dir": ".",
                "train_data": "dataset.json",
                "llama_cpp_dir": ".",
                "venv_python_path": "python",
                "final_weights_path": ".",
            },
            "testing": {
                "use_separate_files": False,
                "test_split_ratio": 0.2,
                "output_test_file": "test.json",
            },
            "other": {"cutoff_len": 128},
            "data_preparation": {"method": "default"},
            "model": {
                "model_name": "gpt2",
                "new_model": "new_model",
                "outfile": "model.gguf",
                "quant": {"qtype": "q4_0", "gguf_dir": "quantized"},
                "model_type": "hf",
            },
            "training": {"seed": 42, "use_grpo": False, "use_sft": False},
        },
    )


@pytest.fixture
def sample_dataset(temp_dir: str) -> str:
    """Create a small dataset.json (dict with '
    system' and 'examples') and return its path.

    Note: data_preparation expects a
    dict with "examples" key (not a bare list).
    """
    examples = [
        {"system": "System message", "user": "User query", "bot": "Assistant response"},
        {"system": "Another system", "user": "Second query", "bot": "Another response"},
    ]
    dataset_struct = {"system": "System message", "examples": examples}
    dataset_path = Path(temp_dir) / "dataset.json"
    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_struct, f)
    return str(dataset_path)


def test_change_dir(temp_dir: str) -> None:
    """Test change_dir context manager changes cwd inside context."""
    original_dir = os.getcwd()
    with change_dir(temp_dir):
        assert os.getcwd() == temp_dir
    assert os.getcwd() == original_dir


def test_generate_prompt(tokenizer: AutoTokenizer) -> None:
    """generate_prompt should return the
    chat-formatted string containing all parts."""
    data_point = {
        "system": "System message",
        "user": "User query",
        "bot": "Assistant response",
    }
    prompt = generate_prompt(tokenizer, data_point)
    assert isinstance(prompt, str)
    assert "System message" in prompt
    assert "User query" in prompt
    assert "Assistant response" in prompt


def test_tokenize(tokenizer: AutoTokenizer) -> None:
    """tokenize should return a dict with input_ids padded to cutoff_len."""
    text = "This is a test string"
    result = tokenize(tokenizer, 128, text)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert len(result["input_ids"]) == 128


def test_generate_and_tokenize_prompt(tokenizer: AutoTokenizer) -> None:
    """generate_and_tokenize_prompt should
    return tokenized prompt dict shaped to cutoff."""
    data_point = {"system": "System", "user": "User", "bot": "Bot"}
    result = generate_and_tokenize_prompt(data_point, tokenizer, 128)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert len(result["input_ids"]) == 128


def test_data_preparation(
    test_config: DictConfig,
    tokenizer: AutoTokenizer,
    sample_dataset: str,
    monkeypatch,
) -> None:
    """data_preparation should load dataset dict
    and return train/val Dataset objects."""
    test_config.paths.data_dir = "."
    test_config.paths.train_data = Path(sample_dataset).name

    # monkeypatch get_original_cwd used inside one_file_train
    monkeypatch.setattr(oft, "get_original_cwd", lambda: str(Path(sample_dataset).parent))

    train_data, val_data = data_preparation(test_config, tokenizer)

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert "input_ids" in train_data.features
    assert "attention_mask" in val_data.features


def test_copy_data(temp_dir: str) -> None:
    """copy_data expects source file in os.getcwd(); create file there inside change_dir."""
    with change_dir(temp_dir):
        test_file = Path(temp_dir) / "test.txt"
        with test_file.open("w", encoding="utf-8") as f:
            f.write("test content")

        dest_dir = Path(temp_dir) / "destination"
        dest_dir.mkdir(parents=True, exist_ok=True)

        copy_data("test.txt", "subdir", str(dest_dir))

    dest_path = dest_dir / "subdir" / "test.txt"
    assert dest_path.exists()
    with dest_path.open(encoding="utf-8") as f:
        assert f.read() == "test content"


def test_model_merge_for_converting(
    test_config: DictConfig,
    temp_dir: str,
    monkeypatch,
) -> None:
    """Mock HF/PEFT loading calls inside
    model_merge_for_converting and assert they were used."""
    calls = {
        "auto_from_pretrained": False,
        "tokenizer_from_pretrained": False,
        "peft_from_pretrained": False,
    }

    class FakeModel:
        def resize_token_embeddings(self, _: Any) -> None:
            pass

        def save_pretrained(self, _: Any) -> None:
            pass

    class FakePeft:
        @staticmethod
        def from_pretrained(_model, _adapter_path) -> SimpleNamespace:
            calls["peft_from_pretrained"] = True
            return SimpleNamespace(
                merge_and_unload=lambda: SimpleNamespace(save_pretrained=lambda _p: None),
            )

    class FakeAutoModelClass:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> FakeModel:
            calls["auto_from_pretrained"] = True
            return FakeModel()

    class FakeTokenizer:
        def __len__(self) -> int:
            return 100

        def save_pretrained(self, _) -> None:
            pass

    class FakeAutoTokenizerClass:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> FakeTokenizer:
            calls["tokenizer_from_pretrained"] = True
            return FakeTokenizer()

    monkeypatch.setattr(oft, "AutoModelForCausalLM", FakeAutoModelClass)
    monkeypatch.setattr(oft, "AutoTokenizer", FakeAutoTokenizerClass)
    monkeypatch.setattr(oft, "PeftModel", FakePeft)

    save_path = str(Path(temp_dir) / "merged_model")
    model_merge_for_converting(test_config, 100, save_path)

    assert calls["auto_from_pretrained"]
    assert calls["tokenizer_from_pretrained"]
    assert calls["peft_from_pretrained"]


def test_convert_to_gguf(test_config: DictConfig, temp_dir: str, monkeypatch) -> None:
    """convert_to_gguf should call subprocess.run when conversion script exists."""
    test_config.paths.llama_cpp_dir = temp_dir
    conv_script = Path(temp_dir) / "convert_hf_to_gguf.py"
    with conv_script.open("w", encoding="utf-8") as f:
        f.write("# dummy converter")

    model_path = Path(temp_dir) / "model"
    model_path.mkdir(parents=True, exist_ok=True)

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    # Patch only subprocess.run (do not replace the module object)
    monkeypatch.setattr(oft.subprocess, "run", fake_run)

    convert_to_gguf(
        model_path=str(model_path),
        outfile=str(Path(temp_dir) / "model.gguf"),
        python_exe="python",
        outtype="f16",
        cfg=test_config,
    )


def test_quantize_model_success(temp_dir: str, monkeypatch) -> None:
    """quantize_model returns True when quantizer exists and subprocess.run succeeds."""
    model_path = Path(temp_dir) / "model.gguf"
    with model_path.open("w", encoding="utf-8") as f:
        f.write("GGUF mock data")

    quantizer_path = Path(temp_dir) / "llama-quantize"
    with quantizer_path.open("w", encoding="utf-8") as f:
        f.write("")

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(stdout=b"ok", stderr=b"")

    monkeypatch.setattr(oft.subprocess, "run", fake_run)

    result = quantize_model(
        model_path=str(model_path),
        outfile=str(Path(temp_dir) / "quantized.gguf"),
        qtype="q4_0",
        llama_cpp_path=temp_dir,
        quantized_path="llama-quantize",
    )

    assert result is True


def test_quantize_model_failure(temp_dir: str, monkeypatch) -> None:
    """quantize_model should return False when subprocess.run raises CalledProcessError."""
    quantizer_path = Path(temp_dir) / "llama-quantize"
    with quantizer_path.open("w", encoding="utf-8") as f:
        f.write("")

    def fake_run_raises(*_args: Any, **_kwargs: Any) -> None:
        # raise the real CalledProcessError from the real subprocess module
        raise subprocess.CalledProcessError(returncode=1, cmd="llama-quantize")

    # Patch only subprocess.run so quantize_model's except subprocess.CalledProcessError
    # still references the real exception class on the real subprocess module.
    monkeypatch.setattr(oft.subprocess, "run", fake_run_raises)

    result = quantize_model(
        model_path=str(Path(temp_dir) / "nonexistent.gguf"),
        outfile=str(Path(temp_dir) / "quantized.gguf"),
        qtype="q4_0",
        llama_cpp_path=temp_dir,
        quantized_path="llama-quantize",
    )

    assert result is False
