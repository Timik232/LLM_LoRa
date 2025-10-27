"""Model evaluation tools for LLM LoRa training framework"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import ollama
from llama_cpp import Llama
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from evaluation.game_evaluation import test_actions

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig

    CallableAny = Callable[..., Any]


# Moved get_user_prompt here to avoid circular import
def get_user_prompt(data: dict[str, Any]) -> str:
    """
    Construct a user prompt from conversation data.

    Args:
        data (Dict[str, Any]): Dictionary containing conversation history and metadata:
            - History: List of previous messages
            - AvailableActions: List of available actions
            - UserInput: Current user input

    Returns:
        str: Formatted prompt string with conversation context.
    """
    prompt = (
        "Системное сообщение, которому ты должен следовать, отмечено словом 'system'. "
        "Предыдущие сообщения пользователя отмечены словом 'user'. "
        "Твои предыдущие сообщения отмечены словом 'VIKA'."
        "\n\nИстория сообщений:"
    )
    for message in data.get("History", []):
        prompt += f"\n{message}"
    prompt += (
        "\n\nТы можешь совершать только действия из представленного списка.\n"
        f"Доступные действия: Разговор, {', '.join(data.get('AvailableActions', []))}"
    )
    prompt += (
        "\n\nОтветь на сообщение пользователя, беря во внимания всю предыдущую информацию.\n"
        f"Сообщение пользователя: {data.get('UserInput', '')}"
    )
    return prompt


# from evaluation.deepeval_integration import test_mention_number_of_values

ollama.base_url = "http://localhost:11434"


class Content(BaseModel):
    """The inner element of the pydantic schema for testing model"""

    model_config = ConfigDict(extra="forbid")

    Action: str = Field(..., description="The action associated with the message.")


class MainModel(BaseModel):
    """Pydantic schema for testing model"""

    model_config = ConfigDict(extra="forbid")

    MessageText: str = Field(..., description="The text of the message.")
    Content: Content


def dataset_to_json_for_test(dataset: dict[str, Any], filename: str | Path) -> None:
    """
    Convert a dataset to a JSON file for testing purposes.

    Args:
        dataset (Dict[str, Any]): The dataset containing system and example information.
        filename (str | Path): The path to the output JSON file.

    Returns:
        None
    """
    json_objects: list[dict[str, str]] = []
    system = dataset["system"]
    dataset = dataset["examples"]

    with Path(filename).open("w", encoding="utf-8") as file:
        file.write("")

    for row in dataset:
        system_message = system
        user_message = get_user_prompt(dataset[row]["prompt"])
        # user_message = str(dataset[row]['prompt'])
        bot_message = str(dataset[row]["answer"])

        json_object = {
            "system": system_message,
            "user": user_message,
            "bot": bot_message,
        }

        json_objects.append(json_object)

    with Path(filename).open("a", encoding="utf-8") as file:
        file.write(json.dumps(json_objects, indent=4, ensure_ascii=False))


def ollama_generate(
    client: ollama.Client,
    model_name: str | bytes,
    prompt: str,
    schema: dict,
) -> dict:
    """
    Wrapper function to generate a response using Ollama's structured outputs.

    Args:
        client (ollama.Client): The Ollama client instance.
        model_name (str): The name of the model in Ollama.
        prompt (str): The formatted prompt to send to the model.
        schema (Dict): The JSON schema for the expected response.

    Returns:
        Dict: The parsed JSON response conforming to the schema.
    """
    response = client.generate(
        model=model_name,
        prompt=prompt,
        format=schema,
    )
    logger = logging.getLogger(__name__)
    logger.debug(response)
    return response["response"]


def call_llm(prompt: str, model: str, client: OpenAI) -> str:
    """
    Sends a prompt to the LLM and returns its response as a dictionary.

    Args:
        prompt (str): The user prompt.
        model (str): The model identifier.
        client (OpenAI): openai client for the llm.

    Returns:
        dict: Parsed LLM response.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.choices:
        # Explicitly avoid hiding the source; do not chain to a different exception here
        raise RuntimeError() from None
    return response.choices[0].message.content


def _execute_single_test(
    test_case_index: int,
    prompt: str,
    correct_answer: str,
    cfg: DictConfig,
    client: OpenAI | ollama.Client,
    test_func: callable | None,
    use_ollama: bool,
    results_lock: Lock,
    results: dict,
) -> None:
    """Execute a single test case and store results in a thread-safe manner.

    Args:
        test_case_index: Index of the test case
        prompt: The user prompt
        correct_answer: Expected answer
        cfg: Hydra configuration
        client: OpenAI or Ollama client
        test_func: Test function(s) to execute
        use_ollama: Whether to use Ollama
        results_lock: Thread lock for synchronizing result updates
        results: Dictionary to store results
    """
    logger = logging.getLogger(__name__)
    try:
        if not use_ollama:
            model_answer = call_llm(
                prompt,
                client=client,
                model=cfg.model.outfile,
            ).strip()
        else:
            schema = MainModel.model_json_schema()
            model_answer = ollama_generate(
                client=client,
                model_name=cfg.model.outfile.replace(".gguf", ""),
                prompt=prompt,
                schema=schema,
            )

        if test_func is not None:
            if isinstance(test_func, list):
                for func in test_func:
                    func_name = func.__name__
                    try:
                        func(model_answer, correct_answer, user_input=prompt)
                        with results_lock:
                            if func_name not in results["function_metrics"]:
                                results["function_metrics"][func_name] = {
                                    "passed": 0,
                                    "total": 0,
                                }
                            results["function_metrics"][func_name]["passed"] += 1
                            results["function_metrics"][func_name]["total"] += 1
                    except AssertionError as e:
                        logger.debug(
                            f"Test {func_name} failed for case {test_case_index}: {e}"
                        )
                        with results_lock:
                            if func_name not in results["function_metrics"]:
                                results["function_metrics"][func_name] = {
                                    "passed": 0,
                                    "total": 0,
                                }
                            results["function_metrics"][func_name]["total"] += 1
                            results["failed_cases"].append(
                                {
                                    "index": test_case_index,
                                    "test": func_name,
                                    "error": str(e),
                                }
                            )
                        return
            else:
                func_name = test_func.__name__
                try:
                    test_func(model_answer, correct_answer, user_input=prompt)
                    with results_lock:
                        if func_name not in results["function_metrics"]:
                            results["function_metrics"][func_name] = {"passed": 0, "total": 0}
                        results["function_metrics"][func_name]["passed"] += 1
                        results["function_metrics"][func_name]["total"] += 1
                        results["passed_tests"] += 1
                except AssertionError as e:
                    logger.debug(f"Test failed for case {test_case_index}: {e}")
                    with results_lock:
                        if func_name not in results["function_metrics"]:
                            results["function_metrics"][func_name] = {"passed": 0, "total": 0}
                        results["function_metrics"][func_name]["total"] += 1
                        results["failed_cases"].append(
                            {
                                "index": test_case_index,
                                "test": func_name,
                                "error": str(e),
                            }
                        )
                    return

            with results_lock:
                if isinstance(test_func, list):
                    results["passed_tests"] += 1
    except Exception as e:
        logger.debug(f"Error processing test case {test_case_index}: {e}")
        with results_lock:
            results["failed_cases"].append(
                {"index": test_case_index, "test": "execution", "error": str(e)}
            )


@dataclass
class TestSummary:
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    execution_mode: str = "sequential"
    num_workers: int = 1
    test_functions: list[str] = field(default_factory=list)
    metrics_per_function: dict[str, dict] = field(default_factory=dict)
    failed_cases: list[dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def duration_str(self) -> str:
        duration = self.duration
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def to_dict(self) -> dict:
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": f"{self.success_rate * 100:.1f}%",
            "execution_mode": self.execution_mode,
            "num_workers": self.num_workers,
            "test_functions": self.test_functions,
            "metrics_per_function": self.metrics_per_function,
            "duration_seconds": round(self.duration, 2),
            "duration_str": self.duration_str(),
        }


def _log_test_summary(summary: TestSummary, logger: logging.Logger) -> None:
    success_rate_pct = summary.success_rate * 100
    separator = "=" * 60
    logger.info(separator)
    logger.info("TEST EXECUTION SUMMARY")
    logger.info(separator)
    logger.info(f"Execution Mode: {summary.execution_mode}")
    if summary.execution_mode == "parallel":
        logger.info(f"Number of Workers: {summary.num_workers}")
    logger.info(f"Total Tests: {summary.total_tests}")
    logger.info(f"  ✓ Passed: {summary.passed_tests} ({success_rate_pct:.1f}%)")
    logger.info(f"  ✗ Failed: {summary.failed_tests} ({100 - success_rate_pct:.1f}%)")

    if summary.test_functions:
        logger.info("\nTest Functions:")
        for func_name in summary.test_functions:
            logger.info(f"  - {func_name}")

    if summary.metrics_per_function:
        logger.info("\nDetailed Metrics by Function:")
        for func_name, metrics in summary.metrics_per_function.items():
            passed = metrics.get("passed", 0)
            total = metrics.get("total", 0)
            success_pct = (passed / total * 100) if total > 0 else 0
            logger.info(f"  {func_name}: {passed}/{total} passed ({success_pct:.1f}%)")

    logger.info(f"\nExecution Time: {summary.duration_str()}")
    logger.info(separator)


def run_tests(
    cfg: DictConfig,
    client: OpenAI | ollama.Client,
    test_dataset_path: str | Path = "data/test_ru.json",
    test_file: str | Path = "test.json",
    test_func: callable | None = None,
    use_ollama: bool = False,
    suppress_log: bool = False,
) -> TestSummary:
    """
    Runs tests by comparing the LLM responses with expected answers from a dataset.

    Supports both sequential and parallel execution based on configuration.

    Args:
        cfg (DictConfig): Configuration with model settings.
        client (OpenAI | ollama.client): OpenAI or ollama client for LLM interaction.
        test_dataset_path (str, optional): Path to the test dataset JSON file.
            Defaults to "data/test_ru.json".
        test_file (str, optional): Path to save the processed test file.
            Defaults to "test.json".
        test_func (callable, optional): Additional test function to execute on each result.
            This function should accept the user prompt,
            LLM's message text and the correct answer.
        use_ollama (bool) : Flag to indicate if Ollama should be used for testing.
        suppress_log (bool): If True, suppress logging of test summary. Used when running
            multiple test functions to avoid duplicate summary logs.

    Returns:
        TestSummary: Summary object containing test execution details and metrics.
    """
    logger = logging.getLogger(__name__)
    summary = TestSummary()
    summary.start_time = time.time()

    with Path(test_dataset_path).open(encoding="utf-8") as file:
        test_dataset = json.load(file)

    dataset_to_json_for_test(test_dataset, test_file)

    with Path(test_file).open(encoding="utf-8") as f:
        prompts = json.load(f)

    prompts_to_check = [prompt["user"] for prompt in prompts]
    expected_answers = [answer["bot"] for answer in prompts]

    summary.total_tests = len(prompts_to_check)

    parallel_cfg = cfg.get("testing", {}).get("parallel_execution", {})
    use_parallel = parallel_cfg.get("enabled", False)
    num_workers = parallel_cfg.get("num_workers", 4)

    if test_func is not None:
        if isinstance(test_func, list):
            summary.test_functions = [f.__name__ for f in test_func]
        else:
            summary.test_functions = [test_func.__name__]

    if not use_parallel or num_workers <= 1:
        logger.info("Running tests sequentially")
        summary.execution_mode = "sequential"
        summary = _run_tests_sequential(
            cfg,
            client,
            prompts_to_check,
            expected_answers,
            test_func,
            use_ollama,
            logger,
            summary,
        )
    else:
        logger.info(f"Running tests in parallel with {num_workers} workers")
        summary.execution_mode = "parallel"
        summary.num_workers = num_workers
        summary = _run_tests_parallel(
            cfg,
            client,
            prompts_to_check,
            expected_answers,
            test_func,
            use_ollama,
            num_workers,
            logger,
            summary,
        )

    summary.end_time = time.time()
    summary.failed_tests = summary.total_tests - summary.passed_tests

    if not suppress_log:
        _log_test_summary(summary, logger)

    return summary


def _run_tests_sequential(
    cfg: DictConfig,
    client: OpenAI | ollama.Client,
    prompts_to_check: list[str],
    expected_answers: list[str],
    test_func: callable | None,
    use_ollama: bool,
    logger: logging.Logger,
    summary: TestSummary,
) -> TestSummary:
    """Execute tests sequentially and update summary."""
    passed_test = 0
    total_tests = len(prompts_to_check)
    function_metrics = {}

    for number in range(total_tests):
        prompt = prompts_to_check[number]
        correct_answer = expected_answers[number].strip()

        try:
            if not use_ollama:
                model_answer = call_llm(
                    prompt,
                    client=client,
                    model=cfg.model.outfile,
                ).strip()
            else:
                schema = MainModel.model_json_schema()
                model_answer = ollama_generate(
                    client=client,
                    model_name=cfg.model.outfile.replace(".gguf", ""),
                    prompt=prompt,
                    schema=schema,
                )

            if test_func is not None:
                if isinstance(test_func, list):
                    all_passed = True
                    for func in test_func:
                        func_name = func.__name__
                        if func_name not in function_metrics:
                            function_metrics[func_name] = {"passed": 0, "total": 0}
                        function_metrics[func_name]["total"] += 1

                        try:
                            func(model_answer, correct_answer, user_input=prompt)
                            function_metrics[func_name]["passed"] += 1
                        except AssertionError:
                            logger.exception(
                                "Test %s failed for prompt: %s.\nModel answer: %s\n"
                                "Expected answer: %s\n",
                                func.__name__,
                                prompt,
                                model_answer,
                                correct_answer,
                            )
                            all_passed = False
                            break
                    if all_passed:
                        passed_test += 1
                else:
                    func_name = test_func.__name__
                    if func_name not in function_metrics:
                        function_metrics[func_name] = {"passed": 0, "total": 0}
                    function_metrics[func_name]["total"] += 1

                    try:
                        test_func(model_answer, correct_answer, user_input=prompt)
                        function_metrics[func_name]["passed"] += 1
                        passed_test += 1
                    except AssertionError:
                        logger.exception(
                            "Test failed for prompt: %s.\nModel answer: %s\n"
                            "Expected answer: %s\n",
                            prompt,
                            model_answer,
                            correct_answer,
                        )
        except Exception as e:
            logger.exception("Error processing test case %d: %s", number, e)

    summary.passed_tests = passed_test
    summary.metrics_per_function = function_metrics

    final_metric = passed_test / total_tests if total_tests > 0 else 0
    logger.info("Metrics: %.2f (%s/%s tests passed)", final_metric, passed_test, total_tests)

    return summary


def _run_tests_parallel(
    cfg: DictConfig,
    client: OpenAI | ollama.Client,
    prompts_to_check: list[str],
    expected_answers: list[str],
    test_func: callable | None,
    use_ollama: bool,
    num_workers: int,
    logger: logging.Logger,
    summary: TestSummary,
) -> TestSummary:
    """Execute tests in parallel using ThreadPoolExecutor."""
    total_tests = len(prompts_to_check)
    results_lock = Lock()
    results = {
        "passed_tests": 0,
        "failed_cases": [],
        "function_metrics": {},
    }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for test_index in range(total_tests):
            prompt = prompts_to_check[test_index]
            correct_answer = expected_answers[test_index].strip()

            future = executor.submit(
                _execute_single_test,
                test_index,
                prompt,
                correct_answer,
                cfg,
                client,
                test_func,
                use_ollama,
                results_lock,
                results,
            )
            futures.append(future)

        completed = 0
        for future in as_completed(futures):
            completed += 1  # noqa: SIM113
            if completed % 10 == 0 or completed == total_tests:
                logger.info(f"Progress: {completed}/{total_tests} tests completed")
            try:
                future.result()
            except Exception as e:
                logger.exception(f"Unexpected error in parallel test execution: {e}")

    passed_test = results["passed_tests"]
    failed_count = len(results["failed_cases"])
    final_metric = passed_test / total_tests if total_tests > 0 else 0

    summary.passed_tests = passed_test
    summary.metrics_per_function = results["function_metrics"]
    summary.failed_cases = results["failed_cases"]

    logger.info(
        "Metrics: %.2f (%s/%s tests passed, %s failed)",
        final_metric,
        passed_test,
        total_tests,
        failed_count,
    )

    if results["failed_cases"] and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Failed test cases:")
        for failed in results["failed_cases"][:10]:
            logger.debug(f"  Case {failed['index']}: {failed['test']} - {failed['error']}")

    return summary


def _log_test_llm_summary(summaries: list[TestSummary], logger: logging.Logger) -> None:
    """Log comprehensive summary for multiple test functions.

    Args:
        summaries: List of TestSummary objects from each test function
        logger: Logger instance for output
    """
    separator = "=" * 70
    logger.info(separator)
    logger.info("COMPREHENSIVE TEST EXECUTION SUMMARY (All Test Functions)")
    logger.info(separator)

    total_tests_all = summaries[0].total_tests if summaries else 0
    total_passed_all = sum(s.passed_tests for s in summaries)
    total_failed_all = sum(s.failed_tests for s in summaries)

    logger.info(f"\nTotal Test Cases: {total_tests_all}")
    logger.info(f"Number of Test Functions: {len(summaries)}\n")

    logger.info("Results by Test Function:")
    logger.info("-" * 70)

    for idx, summary in enumerate(summaries):
        func_name = summary.test_functions[0] if summary.test_functions else f"Test {idx + 1}"
        success_pct = summary.success_rate * 100
        logger.info(f"\n[{idx + 1}] {func_name}")
        logger.info(
            f"    ✓ Passed: {summary.passed_tests}/{summary.total_tests} ({success_pct:.1f}%)"
        )
        logger.info(
            f"    ✗ Failed: {summary.failed_tests}/{summary.total_tests} "
            f"({100 - success_pct:.1f}%)"
        )
        logger.info(f"    ⏱ Time: {summary.duration_str()}")

        if summary.metrics_per_function:
            for func_name_detail, metrics in summary.metrics_per_function.items():
                passed = metrics.get("passed", 0)
                total = metrics.get("total", 0)
                success_pct_detail = (passed / total * 100) if total > 0 else 0
                logger.info(
                    f"      └─ {func_name_detail}: {passed}/{total} "
                    f"({success_pct_detail:.1f}%)"
                )

    logger.info("\n" + "-" * 70)
    logger.info("Overall Statistics:")
    logger.info(f"  • Total Passed Across All Functions: {total_passed_all}")
    logger.info(f"  • Total Failed Across All Functions: {total_failed_all}")
    logger.info(f"  • Total Execution Time: {sum(s.duration for s in summaries):.1f}s")
    logger.info(separator + "\n")


def test_llm(
    cfg: DictConfig,
    path_test_dataset: str | Path = "data/test_ru.json",
    test_file: str | Path = "test.json",
    test_func: list[Callable] | None = None,
    llm_url: str | None = "http://localhost:1234/v1/",
    use_ollama: bool = False,
    ollama_client: ollama.Client | None = None,
) -> list[TestSummary]:
    """
    Test the LLM via LM Studio by comparing model responses with expected answers.

    Args:
        cfg (DictConfig): Configuration dictionary containing model settings.
        path_test_dataset (str, optional): Path to the test dataset JSON file.
            Defaults to "data/test_ru.json".
        test_file (str, optional): Path to save the processed test file.
            Defaults to "test.json".
        test_func (Optional[List[Callable]]): List of
            additional test functions to execute on each result.
        llm_url (str, optional): URL of the LLM service. Defaults to "http://localhost:1234/v1/".
        use_ollama (bool) : Flag to indicate if Ollama should be used for testing.
        ollama_client (Optional[ollama.Client]): Ollama client for connection

    Returns:
        list[TestSummary]: List of summary objects for each test function executed.

    Raises:
        Logs errors for failed tests and prints accuracy metrics.
    """
    if test_func is None:
        test_func = [test_actions]
    if ollama_client is None:
        client = OpenAI(api_key="dummy", base_url=llm_url)
    else:
        client = ollama_client

    logger = logging.getLogger(__name__)
    summaries = []
    has_multiple_tests = len(test_func) > 1
    suppress_intermediate_logs = has_multiple_tests

    for test_idx, test in enumerate(test_func):
        try:
            logger.info(
                f"Running test function {test_idx + 1}/{len(test_func)}: {test.__name__}"
            )
            summary = run_tests(
                cfg=cfg,
                client=client,
                test_dataset_path=path_test_dataset,
                test_file=test_file,
                test_func=test,
                use_ollama=use_ollama,
                suppress_log=suppress_intermediate_logs,
            )
            summaries.append(summary)
            logger.info(f"✓ Test function {test.__name__} completed successfully")
        except Exception as e:
            logger.error(f"✗ Test function {test.__name__} failed with error:")
            logger.exception(e)

    if has_multiple_tests and summaries:
        _log_test_llm_summary(summaries, logger)

    return summaries


def llamacpp_execute_test(
    llm: CallableAny,
    system_prompt: str,
    prompt: str,
    expected_answer: str,
    max_tokens: int,
    temperature: float,
) -> tuple[dict, bool]:
    """
    Выполняет тест для одного запроса.

    Args:
        llm: Модель для генерации ответов.
        system_prompt (str): Системный промпт с инструкциями.
        prompt (str): Пользовательский запрос.
        expected_answer (str): Ожидаемый результат.
        max_tokens (int): Максимальное число генерируемых токенов.
        temperature (float): Параметр температуры для генерации.

    Returns:
        tuple: Кортеж, содержащий словарь с
        результатами теста и булевое значение (True, если тест пройден).
    """
    formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"

    response = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>"],
    )
    response_text = response["choices"][0]["text"]

    json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if json_match:
        try:
            json_response = json.loads(json_match.group(1))
            predicted_action = json_response.get("Content", {}).get("Action")
            passed = predicted_action == expected_answer

            result = {
                "prompt": prompt,
                "expected": expected_answer,
                "predicted": predicted_action,
                "full_response": response_text,
                "passed": passed,
            }

            logger = logging.getLogger(__name__)
            if passed:
                logger.info("Test passed")
            else:
                logger.error("Test failed")
                logger.error("Expected: %s, Got: %s", expected_answer, predicted_action)
        except json.JSONDecodeError:
            logger = logging.getLogger(__name__)
            logger.exception("Test failed: Invalid JSON response")
            logger.exception("Response: %s", response_text)
            result = {
                "prompt": prompt,
                "expected": expected_answer,
                "predicted": "ERROR: Invalid JSON",
                "full_response": response_text,
                "passed": False,
            }
            passed = False
    else:
        logger = logging.getLogger(__name__)
        logger.error("Test failed: No JSON found in response")
        logger.error("Response: %s", response_text)
        result = {
            "prompt": prompt,
            "expected": expected_answer,
            "predicted": "ERROR: No JSON found",
            "full_response": response_text,
            "passed": False,
        }
        passed = False

    return result, passed


def test_via_llamacpp(
    model_path: str | bytes | Path,
    test_dataset: str | Path = "data/test_ru.json",
    test_file: str | Path = "test.json",
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    test_func: Callable = llamacpp_execute_test,
    system_prompt: str | None = None,
) -> float:
    """
    Тестирование GGUF модели через llama.cpp с
    использованием передаваемой функции тестирования.

    Args:
        model_path (str | bytes): Путь к файлу модели GGUF.
        test_dataset (str, optional): Путь к JSON файлу с тестовыми данными.
            По умолчанию "data/test_ru.json".
        test_file (str, optional): Путь для сохранения обработанного тестового файла.
            По умолчанию "test.json".
        n_gpu_layers (int, optional): Количество слоёв для вычислений на GPU.
            По умолчанию -1 (все слои).
        n_ctx (int, optional): Размер окна контекста.
            По умолчанию 2048.
        temperature (float, optional): Температура сэмплинга.
            По умолчанию 0.7.
        max_tokens (int, optional): Максимальное количество генерируемых токенов.
            По умолчанию 2048.
        test_func (Callable): Функция, реализующая принцип тестирования.
        system_prompt (Optional[str], optional): Системный промпт для модели.

    Returns:
        float: Значение точности (accuracy).
    """
    llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=True)
    json_schema = MainModel.model_json_schema()

    with Path(test_dataset).open(encoding="utf-8") as file:
        test_dataset_data = json.load(file)

    dataset_to_json_for_test(test_dataset_data, test_file)

    with Path(test_file).open(encoding="utf-8") as f:
        prompts = json.load(f)

    prompts_to_check = [prompt["user"] for prompt in prompts]
    answers = [
        test_dataset_data["examples"][bot]["answer"]["Content"]["Action"]
        for bot in test_dataset_data["examples"]
    ]

    logger = logging.getLogger(__name__)
    logger.debug(f"Expected answers: {answers}")
    logger.debug(f"Number of prompts: {len(prompts_to_check)}")

    count = 0
    results = []
    if system_prompt is None:
        system_prompt = (
            "Ты – помощник по имени ВИКА на заброшенной космической станции. "
            "У тебя есть доступ к системам станции. "
            "Отвечай только в формате JSON с ключами 'MessageText' и 'Content', "
            "где Content содержит ключ 'Action' с одним из доступных тебе действий. "
            f"Используй следующую JSON схему: {json.dumps(json_schema, ensure_ascii=False)} "
            "Заканчивай ответ символом }."
        )

    for number, prompt in enumerate(prompts_to_check):
        result, passed = test_func(
            llm=llm,
            system_prompt=system_prompt,
            prompt=prompt,
            expected_answer=answers[number],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        results.append(result)
        if passed:
            count += 1
            logger.info(f"Test {number} passed")
        else:
            logger.error(f"Test {number} failed")

    accuracy = count / len(prompts_to_check)
    logger.info(f"Accuracy: {accuracy:.4f} ({count}/{len(prompts_to_check)})")

    with Path("test_results.json").open("w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=2)

    return accuracy
