"""DeepEval integration for model evaluation framework"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from hydra import compose, initialize
from omegaconf import DictConfig

if TYPE_CHECKING:
    from testing_model.models import CustomLocalModel, CustomMistralModel, CustomOpenAIModel

# Defer model imports until needed to avoid circular imports
CustomLocalModel: CustomLocalModel | None = None  # type: ignore[assignment]
CustomMistralModel: CustomMistralModel | None = None  # type: ignore[assignment]
CustomOpenAIModel: CustomOpenAIModel | None = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _ensure_models_imported() -> None:
    """Ensure model classes are imported."""
    global CustomLocalModel, CustomMistralModel, CustomOpenAIModel
    if CustomLocalModel is None:
        try:
            from testing_model.models import (
                CustomLocalModel,
                CustomMistralModel,
                CustomOpenAIModel,
            )
        except Exception as e:
            logger.debug(f"Failed to import model classes: {e}")


logger = logging.getLogger(__name__)


def _load_environment_if_needed() -> None:
    """Load environment variables from .env if configured to do so."""
    try:
        with initialize(version_base=None, config_path="../conf"):
            cfg: DictConfig = compose(config_name="config")
            if cfg.get("environment", {}).get("use_dotenv", False) and load_dotenv:
                with contextlib.suppress(Exception):
                    load_dotenv()
    except (ImportError, OSError):
        if load_dotenv:
            with contextlib.suppress(Exception):
                load_dotenv()


def _get_mistral_model(cfg: DictConfig | None = None) -> CustomMistralModel:
    """Get Mistral model with lazy initialization and config support."""
    _ensure_models_imported()
    _load_environment_if_needed()

    if CustomMistralModel is None:
        raise RuntimeError("CustomMistralModel not available") from None

    if cfg is None:
        cfg = DictConfig(
            {
                "api_key": os.getenv("MISTRAL_API", ""),
                "model_name": "mistral-small-latest",
                "temperature": 0.7,
            }
        )
    else:
        cfg = cfg.get("mistral", {})

    mistral_api = cfg.get("api_key", os.getenv("MISTRAL_API", ""))
    model_name = cfg.get("model_name", "mistral-small-latest")
    temperature = cfg.get("temperature", 0.7)

    return CustomMistralModel(
        api_key=mistral_api,
        model=model_name,
        temperature=temperature,
    )


def _get_local_model(cfg: DictConfig | None = None) -> CustomLocalModel:
    """Get local model with lazy initialization and config support."""
    _ensure_models_imported()
    if CustomLocalModel is None:
        raise RuntimeError("CustomLocalModel not available") from None

    return CustomLocalModel()


def _get_custom_api_model(cfg: DictConfig) -> CustomOpenAIModel | None:
    """Get custom API model client wrapped for DeepEval.

    Args:
        cfg (DictConfig): Hydra configuration with deepeval settings

    Returns:
        CustomOpenAIModel: DeepEval-compatible wrapper for OpenAI-compatible API

    Raises:
        RuntimeError: If CustomOpenAIModel is not available
    """
    _ensure_models_imported()

    custom_cfg = cfg.get("custom_api", {})
    endpoint = custom_cfg.get("endpoint", "http://your-api.com/v1")
    api_key = custom_cfg.get("api_key", os.getenv("CUSTOM_API_KEY", ""))
    model_name = custom_cfg.get("model_name", "your-model-name")
    temperature = custom_cfg.get("temperature", 0.7)

    logger.info(f"Using custom API: {endpoint} with model: {model_name}")

    return CustomOpenAIModel(
        api_key=api_key,
        base_url=endpoint,
        model_name=model_name,
        temperature=temperature,
    )


def _get_evaluation_model(
    cfg: DictConfig,
) -> CustomMistralModel | CustomLocalModel | CustomOpenAIModel:
    """Get evaluation model based on configuration.

    Args:
        cfg (DictConfig): Hydra configuration with deepeval settings

    Returns:
        Model instance for evaluation (Mistral, Local, or Custom API)

    Raises:
        ValueError: If model type is unknown
        RuntimeError: If model initialization fails
    """
    _ensure_models_imported()
    deepeval_cfg = cfg.get("deepeval", {})
    eval_model_cfg = deepeval_cfg.get("evaluation_model", {})
    model_type = eval_model_cfg.get("type", "mistral").lower()

    logger.info(f"Initializing evaluation model: {model_type}")

    if model_type == "mistral":
        return _get_mistral_model(deepeval_cfg)
    if model_type == "local":
        return _get_local_model(deepeval_cfg)
    if model_type == "custom":
        return _get_custom_api_model(deepeval_cfg)
    raise ValueError(f"Unknown evaluation model type: {model_type}")


def set_local_model_via_cli(
    model_name: str = "vikhr-yandexgpt-5-lite-8b-it_gguf",
    base_url: str = "http://localhost:1234/v1",
) -> None:
    """Set the local model via CLI using deepeval command.

    Args:
        model_name (str, optional): Name of the model.
        base_url (str, optional): Base URL for the model.
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
        logger.exception("Command not found. Make sure Python and deepeval are installed.")


def test_mention_number_of_values(cfg: DictConfig, user_input: str, output: str) -> bool:
    """Check if the model mentions the number of values inappropriately.

    Args:
        cfg (DictConfig): Hydra configuration
        user_input (str): The original user input.
        output (str): The model's generated output.

    Returns:
        bool: Result of the DeepEval test.

    Raises:
        AssertionError: If the test fails based on the defined criteria.
    """
    deepeval_cfg = cfg.get("deepeval", {})
    threshold = deepeval_cfg.get("evaluation_model", {}).get("threshold", 0.7)
    verbose = deepeval_cfg.get("evaluation_model", {}).get("verbose_mode", True)

    metric = GEval(
        name="Answer question by itself",
        criteria="IF there is question in the prompt for the user, check that "
        "the model doesn't by itself write answer to the question. "
        "The model should not reveal the correct answer to the user, for example,"
        "if there is question: 'Назовите количество ценностей в компании',"
        "model should not write the answer to the question: 'вы неправильно сказали, "
        "в компании 6 ценностей, а не 8'. This response will be error. In other case,"
        "if there is not any question, model should pass this criteria.",
        model=_get_evaluation_model(cfg),
        verbose_mode=verbose,
        threshold=threshold,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    return assert_test(
        test_case=LLMTestCase(input=user_input, actual_output=output),
        metrics=[metric],
    )


def test_unintended_answer_mention(
    cfg: DictConfig, user_input: str, output: str, expected_answer: str
) -> bool:
    """Detect when model mentions the correct answer when it shouldn't.

    Args:
        cfg (DictConfig): Hydra configuration
        user_input (str): The original user input
        output (str): The model's generated output
        expected_answer (str): The expected correct answer

    Returns:
        bool: Result of the DeepEval test

    Raises:
        AssertionError: If the test fails
    """
    deepeval_cfg = cfg.get("deepeval", {})
    threshold = deepeval_cfg.get("evaluation_model", {}).get("threshold", 0.7)
    verbose = deepeval_cfg.get("evaluation_model", {}).get("verbose_mode", True)

    metric = GEval(
        name="Unintended Answer Mention Detection",
        criteria="IF there is question in the prompt for the user, check that "
        "the model doesn't by itself write answer to the question. "
        "The model should not reveal the correct answer to the user, for example,"
        "if there is question: 'Назовите количество ценностей в компании',"
        "model should not write the answer to the question: 'вы неправильно сказали, "
        "в компании 6 ценностей, а не 8'. This response will be error. In other case,"
        "if there is not any question, model should pass this criteria.",
        model=_get_evaluation_model(cfg),
        verbose_mode=verbose,
        threshold=threshold,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )
    return assert_test(
        test_case=LLMTestCase(input=user_input, actual_output=output),
        metrics=[metric],
    )


def test_russian_language_quality(cfg: DictConfig, output: str) -> bool:
    deepeval_cfg = cfg.get("deepeval", {})
    threshold = deepeval_cfg.get("evaluation_model", {}).get("threshold", 0.7)
    verbose = deepeval_cfg.get("evaluation_model", {}).get("verbose_mode", True)

    metric = GEval(
        name="Russian Language Quality",
        criteria="Evaluate the Russian language quality of the output. Check for: "
        "1) Correct grammar and syntax, 2) Proper spelling, 3) Natural and fluent phrasing, "
        "4) Appropriate word choice, 5) Correct punctuation. "
        "The response should be grammatically correct and linguistically natural.",
        model=_get_evaluation_model(cfg),
        verbose_mode=verbose,
        threshold=threshold,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )

    return assert_test(
        test_case=LLMTestCase(input="", actual_output=output),
        metrics=[metric],
    )


def test_game_context_appropriateness(
    cfg: DictConfig,
    output: str,
    available_actions: list[str],
    user_input: str | None = None,
) -> bool:
    """Verify response is contextually appropriate for game usage.

    Args:
        cfg (DictConfig): Hydra configuration
        output (str): The model's generated output
        available_actions (list[str]): List of valid actions in the game
        user_input (str, optional): The user input for context

    Returns:
        bool: Result of the DeepEval test

    Raises:
        AssertionError: If the test fails
    """
    deepeval_cfg = cfg.get("deepeval", {})
    threshold = deepeval_cfg.get("evaluation_model", {}).get("threshold", 0.7)
    verbose = deepeval_cfg.get("evaluation_model", {}).get("verbose_mode", True)

    actions_str = ", ".join(f"'{action}'" for action in available_actions)
    criteria = (
        f"Evaluate if the response is contextually appropriate for game usage. Check: "
        f"1) The chosen action is valid and from available actions: {actions_str}, "
        f"2) The response maintains narrative flow and game immersion, "
        f"3) No meta-commentary or breaking of game context, "
        f"4) Response follows expected JSON format with Content.Action structure. "
        f"The response should be appropriate for a game context. If answer is neutral"
        f"and can be appropriate both for the game usage and for classic dialog,"
        f"it is still good."
    )

    metric = GEval(
        name="Game Context Appropriateness",
        criteria=criteria,
        model=_get_evaluation_model(cfg),
        verbose_mode=verbose,
        threshold=threshold,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ]
        + ([LLMTestCaseParams.INPUT] if user_input else []),
    )

    return assert_test(
        test_case=LLMTestCase(input=user_input or "", actual_output=output),
        metrics=[metric],
    )


def run_all_deepeval_metrics(
    cfg: DictConfig,
    user_input: str,
    model_output: str,
    expected_answer: str,
    available_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Run all DeepEval metrics and aggregate results.

    Args:
        cfg (DictConfig): Hydra configuration
        user_input (str): The user input
        model_output (str): The model's output
        expected_answer (str): Expected correct answer
        available_actions (list[str], optional): Available game actions

    Returns:
        dict with results for each metric and aggregate score
    """
    results = {
        "all_passed": True,
        "individual_results": {},
        "timestamp": None,
    }

    metrics_to_run = [
        (
            "unintended_answer_mention",
            lambda: test_unintended_answer_mention(
                cfg, user_input, model_output, expected_answer
            ),
        ),
        ("russian_language_quality", lambda: test_russian_language_quality(cfg, model_output)),
    ]

    if available_actions:
        metrics_to_run.append(
            (
                "game_context_appropriateness",
                lambda: test_game_context_appropriateness(
                    cfg, model_output, available_actions, user_input
                ),
            )
        )

    for metric_name, metric_func in metrics_to_run:
        try:
            metric_func()
            results["individual_results"][metric_name] = True
            logger.info(f"✓ {metric_name} passed")
        except AssertionError as e:
            results["individual_results"][metric_name] = False
            results["all_passed"] = False
            logger.warning(f"✗ {metric_name} failed: {e}")
        except Exception as e:
            results["individual_results"][metric_name] = False
            results["all_passed"] = False
            logger.error(f"✗ {metric_name} error: {e}")

    passed_count = sum(1 for v in results["individual_results"].values() if v)
    total_count = len(results["individual_results"])
    results["score"] = passed_count / total_count if total_count > 0 else 0.0

    logger.info(
        f"DeepEval metrics summary: {passed_count}/{total_count} passed "
        f"(score: {results['score']:.2f})"
    )

    return results


def test_from_dataset(
    cfg: DictConfig,
    test_dataset: str | Path = "data/test_ru.json",
    test_file: str | Path = "test.json",
    run_all_metrics: bool = False,
) -> None:
    """Test the model using a dataset of prompts with updated DeepEval metrics.

    Args:
        cfg (DictConfig): Hydra configuration
        test_dataset (str, optional): Path to the test dataset JSON file.
        test_file (str, optional): Path to the processed test file.
        run_all_metrics (bool): If True, run all metrics;
        if False, run only answer mention test
    """
    llm_url = "http://localhost:1234/v1/chat/completions"
    # with Path(test_dataset).open(encoding="utf-8") as file:
    #     test_dataset_data = json.load(file)

    with Path(test_file).open(encoding="utf-8") as f:
        prompts = json.load(f)

    prompts_to_check = [prompt["user"] for prompt in prompts]
    total_tests = len(prompts_to_check)
    passed_tests = 0

    for user_input in prompts_to_check:
        data = {
            "messages": [{"role": "user", "content": user_input}],
            "model": "game-model/v4/model-game_v4.1_q4.gguf",
        }
        try:
            response = requests.post(llm_url, json=data, timeout=30)
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            model_json = json.loads(content)
            model_answer = model_json.get("MessageText", "")
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError):
            logger.exception("LLM request/parse failed for input: %s", user_input)
            continue

        try:
            if run_all_metrics:
                test_unintended_answer_mention(cfg, user_input, model_answer, "")
            else:
                test_mention_number_of_values(cfg, user_input, model_answer)
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
