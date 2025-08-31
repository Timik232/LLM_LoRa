"""DeepEval integration for model evaluation framework"""
import json
import logging
import os
import subprocess
from typing import TYPE_CHECKING

import requests
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from hydra import compose, initialize
from omegaconf import DictConfig

if TYPE_CHECKING:
    pass


def _load_environment_if_needed() -> None:
    """Load environment variables from .env if configured to do so."""
    try:
        with initialize(version_base=None, config_path="../conf"):
            cfg: DictConfig = compose(config_name="config")
            if cfg.get("environment", {}).get("use_dotenv", False):
                from dotenv import load_dotenv

                load_dotenv()
    except Exception:
        # Fallback: try to load dotenv if available (for backward compatibility)
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass


# Load environment only when needed, not at module level
def _get_mistral_model():
    """Get Mistral model with lazy initialization."""
    _load_environment_if_needed()
    from testing_model.models import CustomMistralModel

    mistral_api = os.getenv("MISTRAL_API")
    return CustomMistralModel(
        api_key=mistral_api, model="mistral-small-latest", temperature=0.7
    )


def _get_local_model():
    """Get local model with lazy initialization."""
    from testing_model.models import CustomLocalModel

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
        print("Команда выполнена успешно:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Ошибка выполнения команды:")
        print(e.stderr)
    except FileNotFoundError as e:
        print(
            "Команда не найдена. Убедитесь, "
            "что Python и deepeval установлены и доступны в PATH."
        )
        print(e)


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
        test_case=LLMTestCase(input=user_input, actual_output=output), metrics=[metric]
    )


def test_from_dataset(
    test_dataset: str = "data/test_ru.json", test_file: str = "test.json"
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
    with open(test_dataset, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    # dataset_to_json_for_test(test_dataset, test_file)
    with open(test_file, "r", encoding="utf-8") as f:
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
        response = requests.post(llm_url, json=data)
        model_answer = json.loads(response.json()["choices"][0]["message"]["content"])[
            "MessageText"
        ]
        try:
            test_mention_number_of_values(user_input, model_answer)
            passed_tests += 1
        except AssertionError:
            logging.error(
                f"Тест не пройден для запроса: {user_input}. \nОтвет модели {model_answer}."
            )

    final_metric = passed_tests / total_tests if total_tests > 0 else 0
    logging.info(
        f"Итоговая метрика: {final_metric:.2f} ({passed_tests}/{total_tests} тестов пройдено)"
    )
