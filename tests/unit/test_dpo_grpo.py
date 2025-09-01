"""
Comprehensive pytest test suite for DPO and GRPO functionality.

This module tests:
- DPO configuration validation
- DPO data preparation and processing
- GRPO configuration validation
- GRPO data preparation and reward functions
- Integration with training pipeline
- Data format validation
- Error handling and edge cases
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

# Import modules to test
from training_model.dpo_train import (
    dpo_train,
    prepare_dpo_data,
    validate_dpo_config,
)
from training_model.grpo_train import (
    debug_reward_function,
    grpo_train,
    prepare_grpo_data,
    reward_function,
    validate_grpo_config,
)


class TestDPOConfiguration:
    """Test DPO configuration validation and setup."""

    def test_validate_dpo_config_success(self):
        """Test successful DPO configuration validation."""
        # Create temporary data files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test config
            cfg = OmegaConf.create(
                {
                    "dpo": {
                        "val_data": "dpo_test.json",
                        "train_data": "dpo_dataset.json",
                    },
                    "paths": {"data_dir": "data"},
                }
            )

            # Create test data files
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir(exist_ok=True)

            test_data = {
                "system": "Test system",
                "examples": {
                    "test1": {
                        "prompt": "Test prompt",
                        "chosen": "Good response",
                        "rejected": "Bad response",
                    }
                },
            }

            with (data_dir / "dpo_test.json").open("w") as f:
                json.dump(test_data, f)
            with (data_dir / "dpo_dataset.json").open("w") as f:
                json.dump(test_data, f)

            # Mock get_original_cwd to return temp directory
            with patch("training_model.dpo_train.get_original_cwd", return_value=temp_dir):
                result = validate_dpo_config(cfg)
                assert result is True

    def test_validate_dpo_config_missing_params(self):
        """Test DPO configuration validation with missing parameters."""
        cfg = OmegaConf.create(
            {
                "dpo": {
                    "val_data": "dpo_test.json"
                    # Missing train_data
                }
            }
        )

        result = validate_dpo_config(cfg)
        assert result is False

    def test_validate_dpo_config_missing_files(self):
        """Test DPO configuration validation with missing data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create(
                {
                    "dpo": {
                        "val_data": "nonexistent.json",
                        "train_data": "also_nonexistent.json",
                    },
                    "paths": {"data_dir": "data"},
                }
            )

            with patch("training_model.dpo_train.get_original_cwd", return_value=temp_dir):
                result = validate_dpo_config(cfg)
                assert result is False


class TestDPODataPreparation:
    """Test DPO data preparation and processing."""

    def test_prepare_dpo_data_success(self):
        """Test successful DPO data preparation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create(
                {
                    "dpo": {
                        "val_data": "dpo_test.json",
                        "train_data": "dpo_dataset.json",
                    },
                    "paths": {"data_dir": "data"},
                }
            )

            # Create test data
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir(exist_ok=True)

            test_data = {
                "system": "You are a helpful assistant",
                "examples": {
                    "topic1": {
                        "prompt": "What is AI?",
                        "chosen": "AI is artificial intelligence.",
                        "rejected": "AI is just a buzzword.",
                    },
                    "topic2": {
                        "prompt": "How do computers work?",
                        "chosen": "Computers process data using binary operations.",
                        "rejected": "Computers are magic boxes.",
                    },
                },
            }

            with (data_dir / "dpo_test.json").open("w") as f:
                json.dump(test_data, f)
            with (data_dir / "dpo_dataset.json").open("w") as f:
                json.dump(test_data, f)

            with patch("training_model.dpo_train.get_original_cwd", return_value=temp_dir):
                train_data, val_data = prepare_dpo_data(cfg)

                assert isinstance(train_data, Dataset)
                assert isinstance(val_data, Dataset)
                assert len(train_data) == 2
                assert len(val_data) == 2

                # Check data structure
                sample = train_data[0]
                assert "prompt" in sample
                assert "chosen" in sample
                assert "rejected" in sample
                assert "topic" in sample
                assert "System: You are a helpful assistant" in sample["prompt"]

    def test_prepare_dpo_data_invalid_structure(self):
        """Test DPO data preparation with invalid data structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create(
                {
                    "dpo": {
                        "val_data": "dpo_test.json",
                        "train_data": "dpo_dataset.json",
                    },
                    "paths": {"data_dir": "data"},
                }
            )

            # Create test data with missing fields
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir(exist_ok=True)

            invalid_data = {
                "system": "Test system",
                "examples": {
                    "topic1": {
                        "prompt": "Test prompt",
                        "chosen": "Good response",
                        # Missing rejected field
                    },
                    "topic2": {
                        "chosen": "Another good response",
                        "rejected": "Bad response",
                        # Missing prompt field
                    },
                },
            }

            with (Path(data_dir) / "dpo_test.json").open("w") as f:
                json.dump(invalid_data, f)
            with (Path(data_dir) / "dpo_dataset.json").open("w") as f:
                json.dump(invalid_data, f)

            with patch("training_model.dpo_train.get_original_cwd", return_value=temp_dir):
                train_data, val_data = prepare_dpo_data(cfg)

                # Should filter out invalid examples
                assert len(train_data) == 0  # Both examples are invalid
                assert len(val_data) == 0


class TestGRPOConfiguration:
    """Test GRPO configuration validation and setup."""

    def test_validate_grpo_config_success(self):
        """Test successful GRPO configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create(
                {
                    "grpo": {
                        "val_data": "test_ru.json",
                        "train_data": "dataset_ru.json",
                        "num_generations": 2,
                    },
                    "paths": {"data_dir": "data"},
                }
            )

            # Create test data files
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir(exist_ok=True)

            test_data = {
                "system": "Test system",
                "examples": {
                    "topic1": {
                        "prompt": {"History": ["Test"], "UserInput": "Test input"},
                        "answer": {"Content": {"Action": "Test action"}},
                    }
                },
            }

            with (data_dir / "test_ru.json").open("w") as f:
                json.dump(test_data, f)
            with (data_dir / "dataset_ru.json").open("w") as f:
                json.dump(test_data, f)

            with patch("training_model.grpo_train.get_original_cwd", return_value=temp_dir):
                result = validate_grpo_config(cfg)
                assert result is True

    def test_validate_grpo_config_missing_params(self):
        """Test GRPO configuration validation with missing parameters."""
        cfg = OmegaConf.create(
            {
                "grpo": {
                    "val_data": "test_ru.json",
                    "train_data": "dataset_ru.json",
                    # Missing num_generations
                }
            }
        )

        result = validate_grpo_config(cfg)
        assert result is False


class TestGRPORewardFunction:
    """Test GRPO reward function and evaluation."""

    def test_reward_function_correct_json(self):
        """Test reward function with correct JSON format."""
        completions = [
            '{"Content": {"Action": "Разговор"}}',
            '{"Content": {"Action": "Игра"}}',
            '{"Content": {"Action": "Разговор"}}',
        ]
        correct_answer = "Разговор"

        rewards = reward_function(completions, correct_answer=correct_answer)

        assert len(rewards) == 3
        assert rewards[0] == 1.0  # Correct match
        assert rewards[1] == 0.0  # Wrong action
        assert rewards[2] == 1.0  # Correct match

    def test_reward_function_invalid_json(self):
        """Test reward function with invalid JSON."""
        completions = [
            "not json at all",
            '{"Invalid": "structure"}',
            '{"Content": "not a dict"}',
            "",
        ]
        correct_answer = "Разговор"

        rewards = reward_function(completions, correct_answer=correct_answer)

        assert len(rewards) == 4
        assert all(reward == -1.0 for reward in rewards)

    def test_reward_function_missing_correct_answer(self):
        """Test reward function without correct answer."""
        completions = ['{"Content": {"Action": "Test"}}']

        rewards = reward_function(completions)  # No correct_answer

        assert len(rewards) == 1
        assert rewards[0] == -1.0

    def test_test_reward_function(self):
        """Test the reward function testing utility."""
        completions = [
            '{"Content": {"Action": "Разговор"}}',
            '{"Content": {"Action": "Игра"}}',
            "invalid json",
        ]
        correct_answer = "Разговор"

        results = debug_reward_function(completions, correct_answer)

        assert results["total_completions"] == 3
        assert results["positive_rewards"] == 1
        assert results["zero_rewards"] == 1
        assert results["negative_rewards"] == 1
        assert results["average_reward"] == 0.0


class TestGRPODataPreparation:
    """Test GRPO data preparation and processing."""

    def test_prepare_grpo_data_success(self):
        """Test successful GRPO data preparation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create(
                {
                    "grpo": {
                        "val_data": "test_ru.json",
                        "train_data": "dataset_ru.json",
                    },
                    "paths": {"data_dir": "data"},
                }
            )

            # Create test data
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir(exist_ok=True)

            test_data = {
                "system": "You are a helpful assistant",
                "examples": {
                    "topic1": {
                        "prompt": {
                            "History": ["Previous conversation"],
                            "AvailableActions": ["Разговор", "Игра", "Обучение"],
                            "UserInput": "Привет, как дела?",
                        },
                        "answer": {"Content": {"Action": "Разговор"}},
                    },
                    "topic2": {
                        "prompt": {
                            "History": ["Game started"],
                            "AvailableActions": ["Игра", "Обучение"],
                            "UserInput": "Давай играть",
                        },
                        "answer": {"Content": {"Action": "Игра"}},
                    },
                },
            }

            with (data_dir / "test_ru.json").open("w") as f:
                json.dump(test_data, f)
            with (data_dir / "dataset_ru.json").open("w") as f:
                json.dump(test_data, f)

            with patch("training_model.grpo_train.get_original_cwd", return_value=temp_dir):
                train_data, val_data = prepare_grpo_data(cfg)

                assert isinstance(train_data, Dataset)
                assert isinstance(val_data, Dataset)
                assert len(train_data) == 2
                assert len(val_data) == 2

                # Check data structure
                sample = train_data[0]
                assert "prompt" in sample
                assert "correct_answer" in sample
                assert "topic" in sample
                assert "System: You are a helpful assistant" in sample["prompt"]
                assert "Available Actions:" in sample["prompt"]
                assert "JSON object" in sample["prompt"]

    def test_prepare_grpo_data_invalid_structure(self):
        """Test GRPO data preparation with invalid data structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create(
                {
                    "grpo": {
                        "val_data": "test_ru.json",
                        "train_data": "dataset_ru.json",
                    },
                    "paths": {"data_dir": "data"},
                }
            )

            # Create test data with invalid structure
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir(exist_ok=True)

            invalid_data = {
                "system": "Test system",
                "examples": {
                    "topic1": {
                        "prompt": {"UserInput": "Test"},
                        "answer": "invalid_answer_format",  # Should be dict
                    },
                    "topic2": {
                        "prompt": {"UserInput": "Test"},
                        "answer": {"Content": {}},  # Missing Action
                    },
                },
            }

            with (Path(data_dir) / "test_ru.json").open("w") as f:
                json.dump(invalid_data, f)
            with (Path(data_dir) / "dataset_ru.json").open("w") as f:
                json.dump(invalid_data, f)

            with patch("training_model.grpo_train.get_original_cwd", return_value=temp_dir):
                train_data, val_data = prepare_grpo_data(cfg)

                # Should filter out invalid examples
                assert len(train_data) == 0
                assert len(val_data) == 0


class TestTrainingIntegration:
    """Test integration with training pipeline."""

    @patch("training_model.dpo_train.DPOTrainer")
    @patch("training_model.dpo_train.DPOConfig")
    def test_dpo_train_integration(self, mock_config, mock_trainer):
        """Test DPO training integration."""
        # Mock configuration
        cfg = OmegaConf.create(
            {
                "dpo": {
                    "val_data": "dpo_test.json",
                    "train_data": "dpo_dataset.json",
                    "beta": 0.1,
                    "loss_type": "sigmoid",
                    "max_length": 1024,
                    "max_prompt_length": 512,
                    "max_target_length": 512,
                },
                "model": {"new_model": "test_model"},
                "training": {
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "num_train_epochs": 1,
                    "logging_steps": 10,
                    "eval_steps": 100,
                    "warmup_steps": 0,
                    "fp16": False,
                    "bf16": True,
                    "weight_decay": 0.01,
                    "gradient_checkpointing": True,
                    "save_total_limit": 1,
                    "load_best": False,
                },
                "other": {"cutoff_len": 2048},
            }
        )

        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.state.global_step = 100
        mock_trainer_instance.state.log_history = [{"eval_loss": 0.5}]
        mock_trainer.return_value = mock_trainer_instance

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock data preparation
        mock_train_data = Dataset.from_list(
            [{"prompt": "test", "chosen": "good", "rejected": "bad"}]
        )
        mock_val_data = Dataset.from_list(
            [{"prompt": "test2", "chosen": "good2", "rejected": "bad2"}]
        )

        with (
            patch("training_model.dpo_train.validate_dpo_config", return_value=True),
            patch(
                "training_model.dpo_train.prepare_dpo_data",
                return_value=(mock_train_data, mock_val_data),
            ),
        ):
            result = dpo_train(mock_model, mock_tokenizer, cfg)

            assert result == 100
            mock_trainer.assert_called_once()
            mock_trainer_instance.train.assert_called_once()

    @patch("training_model.grpo_train.GRPOTrainer")
    @patch("training_model.grpo_train.GRPOConfig")
    def test_grpo_train_integration(self, mock_config, mock_trainer):
        """Test GRPO training integration."""
        # Mock configuration
        cfg = OmegaConf.create(
            {
                "grpo": {
                    "val_data": "test_ru.json",
                    "train_data": "dataset_ru.json",
                    "num_generations": 2,
                    "max_completion_length": 1024,
                    "epsilon": 0.2,
                    "beta": 0.01,
                    "loss_type": "sigmoid",
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "response_length": 256,
                },
                "model": {"new_model": "test_model"},
                "training": {
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "num_train_epochs": 1,
                    "logging_steps": 10,
                    "eval_steps": 100,
                    "warmup_steps": 0,
                    "fp16": False,
                    "bf16": True,
                    "weight_decay": 0.01,
                    "gradient_checkpointing": True,
                    "save_total_limit": 1,
                    "load_best": False,
                },
            }
        )

        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.state.global_step = 200
        mock_trainer_instance.state.log_history = [{"train_loss": 0.3}]
        mock_trainer.return_value = mock_trainer_instance

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.model_max_length = 2048

        # Mock data preparation
        mock_train_data = Dataset.from_list(
            [{"prompt": "test prompt", "correct_answer": "Разговор"}]
        )
        mock_val_data = Dataset.from_list(
            [{"prompt": "test prompt2", "correct_answer": "Игра"}]
        )

        with (
            patch("training_model.grpo_train.validate_grpo_config", return_value=True),
            patch(
                "training_model.grpo_train.prepare_grpo_data",
                return_value=(mock_train_data, mock_val_data),
            ),
        ):
            result = grpo_train(mock_model, mock_tokenizer, cfg, None)

            assert result == 200
            mock_trainer.assert_called_once()
            mock_trainer_instance.train.assert_called_once()


class TestDataFormatValidation:
    """Test data format validation and structure."""

    def test_dpo_data_format_validation(self):
        """Test DPO data format validation against schema."""
        # Load actual DPO data files if they exist
        project_root = Path(__file__).parent.parent
        dpo_dataset_path = project_root / "data" / "dpo_dataset.json"

        if dpo_dataset_path.exists():
            with dpo_dataset_path.open(encoding="utf-8") as f:
                data = json.load(f)

            # Validate schema
            assert "system" in data
            assert "examples" in data
            assert isinstance(data["examples"], dict)

            for topic, example in data["examples"].items():
                assert isinstance(topic, str)
                assert "prompt" in example
                assert "chosen" in example
                assert "rejected" in example
                assert isinstance(example["prompt"], str)
                assert isinstance(example["chosen"], str)
                assert isinstance(example["rejected"], str)

    def test_grpo_data_format_requirements(self):
        """Test GRPO data format requirements."""
        # Test expected GRPO data structure
        test_data = {
            "system": "Test system",
            "examples": {
                "topic1": {
                    "prompt": {
                        "History": ["conversation history"],
                        "AvailableActions": ["Action1", "Action2"],
                        "UserInput": "user input",
                    },
                    "answer": {"Content": {"Action": "Action1"}},
                }
            },
        }

        # Validate structure
        assert "system" in test_data
        assert "examples" in test_data

        for example in test_data["examples"].values():
            assert "prompt" in example
            assert "answer" in example

            prompt = example["prompt"]
            assert "UserInput" in prompt

            answer = example["answer"]
            assert "Content" in answer
            assert "Action" in answer["Content"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_dpo_train_validation_failure(self):
        """Test DPO training with validation failure."""
        cfg = OmegaConf.create({"dpo": {}})  # Missing required fields

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with pytest.raises(ValueError, match="DPO configuration validation failed"):
            dpo_train(mock_model, mock_tokenizer, cfg)

    def test_grpo_train_validation_failure(self):
        """Test GRPO training with validation failure."""
        cfg = OmegaConf.create({"grpo": {}})  # Missing required fields

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with pytest.raises(ValueError, match="GRPO configuration validation failed"):
            grpo_train(mock_model, mock_tokenizer, cfg, None)

    def test_reward_function_exception_handling(self):
        """Test reward function with exceptions."""
        completions = [
            '{"Content": {"Action": "Test"}}',  # Valid
            None,  # Will cause exception
            42,  # Will cause exception
        ]

        # Convert to strings as the function expects
        str_completions = [str(c) if c is not None else "" for c in completions]

        rewards = reward_function(str_completions, correct_answer="Test")

        assert len(rewards) == 3
        assert rewards[0] == 1.0  # Valid and correct
        assert rewards[1] == -1.0  # Empty string
        assert rewards[2] == -1.0  # Invalid JSON


# Pytest fixtures for common test data
@pytest.fixture
def sample_dpo_config():
    """Fixture providing sample DPO configuration."""
    return OmegaConf.create(
        {
            "dpo": {
                "val_data": "dpo_test.json",
                "train_data": "dpo_dataset.json",
                "beta": 0.1,
                "loss_type": "sigmoid",
                "max_length": 1024,
            },
            "paths": {"data_dir": "data"},
            "model": {"new_model": "test_model"},
            "training": {
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "num_train_epochs": 1,
                "logging_steps": 10,
                "eval_steps": 100,
                "warmup_steps": 0,
                "fp16": False,
                "bf16": True,
                "weight_decay": 0.01,
                "gradient_checkpointing": True,
                "save_total_limit": 1,
                "load_best": False,
            },
            "other": {"cutoff_len": 2048},
        }
    )


@pytest.fixture
def sample_grpo_config():
    """Fixture providing sample GRPO configuration."""
    return OmegaConf.create(
        {
            "grpo": {
                "val_data": "test_ru.json",
                "train_data": "dataset_ru.json",
                "num_generations": 2,
                "max_completion_length": 1024,
                "epsilon": 0.2,
                "beta": 0.01,
                "loss_type": "sigmoid",
                "do_sample": True,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "response_length": 256,
            },
            "paths": {"data_dir": "data"},
            "model": {"new_model": "test_model"},
            "training": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "num_train_epochs": 1,
                "logging_steps": 10,
                "eval_steps": 100,
                "warmup_steps": 0,
                "fp16": False,
                "bf16": True,
                "weight_decay": 0.01,
                "gradient_checkpointing": True,
                "save_total_limit": 1,
                "load_best": False,
            },
        }
    )


@pytest.fixture
def sample_dpo_data():
    """Fixture providing sample DPO data."""
    return {
        "system": "You are a helpful assistant",
        "examples": {
            "topic1": {
                "prompt": "What is the capital of France?",
                "chosen": "The capital of France is Paris.",
                "rejected": "I don't know.",
            },
            "topic2": {
                "prompt": "How do you make coffee?",
                "chosen": (
                    "To make coffee, you need coffee beans, hot water, "
                    "and a brewing method like a coffee maker or French press."
                ),
                "rejected": "Just add water to coffee.",
            },
        },
    }


@pytest.fixture
def sample_grpo_data():
    """Fixture providing sample GRPO data."""
    return {
        "system": "You are Vika, a helpful AI assistant",
        "examples": {
            "topic1": {
                "prompt": {
                    "History": ["User started conversation"],
                    "AvailableActions": ["Разговор", "Игра", "Обучение"],
                    "UserInput": "Привет!",
                },
                "answer": {"Content": {"Action": "Разговор"}},
            },
            "topic2": {
                "prompt": {
                    "History": ["Playing a game"],
                    "AvailableActions": ["Игра", "Обучение"],
                    "UserInput": "Продолжим играть?",
                },
                "answer": {"Content": {"Action": "Игра"}},
            },
        },
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
