import json
import os
import tempfile

import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer

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


# Фикстура для временной директории
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Фикстура для токенизатора
@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


# Фикстура для конфигурации
@pytest.fixture
def test_config():
    return OmegaConf.create(
        {
            "paths": {
                "data_dir": "data",
                "dataset": "dataset.json",
                "output_dir": "output",
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
            "model": {
                "model_name": "gpt2",
                "new_model": "new_model",
                "outfile": "model.gguf",
                "quant": {"qtype": "q4_0", "gguf_dir": "quantized"},
            },
            "training": {"seed": 42, "use_grpo": False, "use_sft": False},
        }
    )


# Фикстура для тестового датасета
@pytest.fixture
def sample_dataset(temp_dir):
    data = [
        {"system": "System message", "user": "User query", "bot": "Assistant response"},
        {"system": "Another system", "user": "Second query", "bot": "Another response"},
    ]
    dataset_path = os.path.join(temp_dir, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(data, f)
    return dataset_path


# Тест для change_dir
def test_change_dir(temp_dir):
    original_dir = os.getcwd()

    with change_dir(temp_dir):
        assert os.getcwd() == temp_dir

    assert os.getcwd() == original_dir


# Тест для generate_prompt
def test_generate_prompt(tokenizer):
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


# Тест для tokenize
def test_tokenize(tokenizer):
    text = "This is a test string"
    result = tokenize(tokenizer, 128, text)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert len(result["input_ids"]) == 128


# Тест для generate_and_tokenize_prompt
def test_generate_and_tokenize_prompt(tokenizer):
    data_point = {"system": "System", "user": "User", "bot": "Bot"}
    result = generate_and_tokenize_prompt(data_point, tokenizer, 128)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert len(result["input_ids"]) == 128


# Тест для data_preparation
def test_data_preparation(test_config, tokenizer, sample_dataset, monkeypatch):
    # Подменяем пути в конфиге
    test_config.paths.dataset = sample_dataset
    test_config.paths.data_dir = os.path.dirname(sample_dataset)

    # Монкипатчим get_original_cwd
    monkeypatch.setattr(
        "main.get_original_cwd", lambda: os.path.dirname(sample_dataset)
    )

    train_data, val_data = data_preparation(test_config, tokenizer)

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert "input_ids" in train_data.features
    assert "attention_mask" in val_data.features


# Тест для copy_data
def test_copy_data(temp_dir):
    # Создаем тестовый файл
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test content")

    dest_dir = os.path.join(temp_dir, "destination")
    os.makedirs(dest_dir, exist_ok=True)

    copy_data("test.txt", "subdir", dest_dir)

    dest_path = os.path.join(dest_dir, "subdir", "test.txt")
    assert os.path.exists(dest_path)
    with open(dest_path, "r") as f:
        assert f.read() == "test content"


# Тест для model_merge_for_converting (мок-тест)
def test_model_merge_for_converting(test_config, temp_dir, mocker):
    # Мокаем зависимости
    mock_from_pretrained = mocker.patch("main.AutoModelForCausalLM.from_pretrained")
    mock_tokenizer = mocker.patch("main.AutoTokenizer.from_pretrained")
    mock_peft = mocker.patch("main.PeftModel.from_pretrained")

    # Вызываем функцию
    save_path = os.path.join(temp_dir, "merged_model")
    model_merge_for_converting(test_config, 100, save_path)

    # Проверяем вызовы
    mock_from_pretrained.assert_called_once()
    mock_tokenizer.assert_called_once()
    mock_peft.assert_called_once()


# Тест для convert_to_gguf (мок-тест)
def test_convert_to_gguf(test_config, temp_dir, mocker):
    # Мокаем subprocess
    mock_subprocess = mocker.patch("main.subprocess.run")

    # Создаем фиктивную модель
    model_path = os.path.join(temp_dir, "model")
    os.makedirs(model_path, exist_ok=True)

    # Вызываем функцию
    convert_to_gguf(
        model_path=model_path,
        outfile="model.gguf",
        python_exe="python",
        outtype="f16",
        cfg=test_config,
    )

    # Проверяем вызов
    mock_subprocess.assert_called_once()


# Тест для quantize_model
def test_quantize_model_success(temp_dir, mocker):
    # Создаем фиктивный файл модели
    model_path = os.path.join(temp_dir, "model.gguf")
    with open(model_path, "w") as f:
        f.write("GGUF mock data")

    # Мокаем subprocess
    mock_subprocess = mocker.patch("main.subprocess.run")
    mock_subprocess.return_value.returncode = 0

    # Вызываем функцию
    result = quantize_model(
        model_path=model_path,
        outfile=os.path.join(temp_dir, "quantized.gguf"),
        llama_cpp_path=temp_dir,
        quantized_path="llama-quantize",
    )

    assert result is True


# Тест для quantize_model (ошибка)
def test_quantize_model_failure(temp_dir, mocker):
    # Мокаем subprocess для вызова исключения
    mock_subprocess = mocker.patch("main.subprocess.run")
    mock_subprocess.side_effect = Exception("Quantization error")

    result = quantize_model(
        model_path="nonexistent.gguf",
        outfile="quantized.gguf",
        llama_cpp_path=temp_dir,
        quantized_path="llama-quantize",
    )

    assert result is False
