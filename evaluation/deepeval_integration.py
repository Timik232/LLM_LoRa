"""DeepEval integration for model evaluation framework"""

import contextlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from hydra import compose, initialize

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from testing_model.models import CustomLocalModel, CustomMistralModel


try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from testing_model.models import CustomLocalModel, CustomMistralModel
except Exception:
    CustomLocalModel = None
    CustomMistralModel = None

logger = logging.getLogger(__name__)


def _load_environment_if_needed() -> None:
    """Load environment variables from .env if configured to do so."""
    try:
        with initialize(version_base=None, config_path="../conf"):
            cfg: DictConfig = compose(config_name="config")
            if cfg.get("environment", {}).get("use_dotenv", False) and load_dotenv:
                # Ignore dotenv failures during initialization
                with contextlib.suppress(Exception):
                    load_dotenv()
    except (ImportError, OSError):
        # Fallback: try to load dotenv if available (for backward compatibility)
        if load_dotenv:
            with contextlib.suppress(Exception):
                load_dotenv()


# Load environment only when needed, not at module level
def _get_mistral_model() -> "CustomMistralModel":
    """Get Mistral model with lazy initialization."""
    _load_environment_if_needed()
    # The testing_model classes are imported at module level when available.
    if CustomMistralModel is None:
        # testing_model package not available in this environment
        raise RuntimeError() from None

    mistral_api = os.getenv("MISTRAL_API", "")
    return CustomMistralModel(
        api_key=mistral_api,
        model="mistral-small-latest",
        temperature=0.7,
    )


def _get_local_model() -> "CustomLocalModel":
    """Get local model with lazy initialization."""
    # Use the top-level imported symbol; avoid local imports
    if CustomLocalModel is None:
        raise RuntimeError() from None

    return CustomLocalModel()


def set_local_model_via_cli(
    model_name: str = "vikhr-yandexgpt-5-lite-8b-it_gguf",
    base_url: str = "http://localhost:1234/v1",
) -> None:
    """
    Set the local model via CLI using deepeval command.

    Args:
        model_name (str, optional): Name of the model.
            Defaults to "vikhr-yandexgpt-5-lite-8b-it_gguf".
        base_url (str, optional): Base URL for the model.
            Defaults to "http://localhost:1234/v1".

    Prints:
        - Success message with command output
        - Error message if command fails
    """
    command = [
        "python",
        "-m",
        "deepeval",
        "set-local-model",
        f"--model-name={model_name}",
        f"--base-url={base_url}",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Command executed successfully:")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.exception("Command execution error:")
        logger.exception(e.stderr)
    except FileNotFoundError:
        logger.exception(
            "Command not found. Make sure "
            "Python and deepeval are installed and available in PATH.",
        )


def test_mention_number_of_values(user_input: str, output: str) -> bool:
    """
    Check if the model mentions the number of values inappropriately.

    Args:
        user_input (str): The original user input.
        output (str): The model's generated output.

    Returns:
        bool: Result of the DeepEval test.

    Raises:
        AssertionError: If the test fails based on the defined criteria.
    """
    metric = GEval(
        name="Answer question by itself",
        criteria="Check that the model doesn't by "
        "itself write answer to the question from the VIKA.",
        # "Also check that the model does not write to the user the correct answer for the"
        # "question from the VIKA",
        # criteria="Проверьте, что модель не пишет сама
        # пользователю конкретное количество ценностей в ответе"
        #          "Также проверьте, что модель не пишет
        #          пользователю правильный ответ на свой вопрос, который"
        #          "от него ожидает услышать.",
        # evaluation_steps=[
        #     "Check that the model does not write the number of values by itself",
        #     # "Check that the Actual Output does not provide the correct
        #     answer to the VIKA question as specified in the Input.",
        #     # "Confirm that the Actual Output does not directly answer
        #     the question from the VIKA, even if user want it."
        # ],
        model=_get_mistral_model(),
        verbose_mode=True,
        threshold=0.7,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    return assert_test(
        test_case=LLMTestCase(input=user_input, actual_output=output),
        metrics=[metric],
    )


def test_from_dataset(
    test_dataset: str | Path = "data/test_ru.json",
    test_file: str | Path = "test.json",
) -> None:
    """
    Test the model using a dataset of prompts.

    Args:
        test_dataset (str, optional): Path to the test dataset JSON file.
            Defaults to "data/test_ru.json".
        test_file (str, optional): Path to the processed test file.
            Defaults to "test.json".

    Logs:
        - Errors for failed tests
        - Final test metrics
    """
    llm_url = "http://localhost:1234/v1/chat/completions"
    with Path(test_dataset).open(encoding="utf-8") as file:
        test_dataset = json.load(file)
    # dataset_to_json_for_test(test_dataset, test_file)
    with Path(test_file).open(encoding="utf-8") as f:
        prompts = json.load(f)
    prompts_to_check = [prompt["user"] for prompt in prompts]
    # answers = [
    #     test_dataset["examples"][bot]["answer"]["Content"]["Action"]
    #     for bot in test_dataset["examples"]
    # ]
    total_tests = len(prompts_to_check)
    passed_tests = 0
    for user_input in prompts_to_check:
        data = {
            "messages": [{"role": "user", "content": user_input}],
            "model": "game-model/v4/model-game_v4.1_q4.gguf",
        }
        response = requests.post(llm_url, json=data, timeout=30)
        model_answer = json.loads(response.json()["choices"][0]["message"]["content"])[
            "MessageText"
        ]
        try:
            test_mention_number_of_values(user_input, model_answer)
            passed_tests += 1
        except AssertionError:
            logger.exception(
                "Test failed for request: %s. \nModel response %s.",
                user_input,
                model_answer,
            )

    final_metric = passed_tests / total_tests if total_tests > 0 else 0
    logger.info(
        "Final metric: %.2f (%s/%s tests passed)",
        final_metric,
        passed_tests,
        total_tests,
    )
