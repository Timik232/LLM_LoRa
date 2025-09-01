"""Model evaluation tools for LLM LoRa training framework"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
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


def run_tests(
    cfg: DictConfig,
    client: OpenAI | ollama.Client,
    test_dataset_path: str | Path = "data/test_ru.json",
    test_file: str | Path = "test.json",
    test_func: callable | None = None,
    use_ollama: bool = False,
) -> None:
    """
    Runs tests by comparing the LLM responses with expected answers from a dataset.

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

    Returns:
        None
    """
    with Path(test_dataset_path).open(encoding="utf-8") as file:
        test_dataset = json.load(file)

    dataset_to_json_for_test(test_dataset, test_file)

    with Path(test_file).open(encoding="utf-8") as f:
        prompts = json.load(f)

    prompts_to_check = [prompt["user"] for prompt in prompts]
    expected_answers = [answer["bot"] for answer in prompts]

    passed_test = 0

    for number in range(len(prompts_to_check)):
        prompt = prompts_to_check[number]
        correct_answer = expected_answers[number].strip()
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
            try:
                test_func(prompt, model_answer, correct_answer)
                passed_test += 1
            except AssertionError:
                logger = logging.getLogger(__name__)
                logger.exception(
                    "Test failed for prompt: %s.\n Error: \n"
                    "Model answer: %s\n"
                    "Expected answer: %s\n",
                    prompt,
                    model_answer,
                    correct_answer,
                )

    total_tests = len(prompts_to_check)
    final_metric = passed_test / total_tests if total_tests > 0 else 0
    logger = logging.getLogger(__name__)
    logger.info("Metrics: %.2f (%s/%s tests passed)", final_metric, passed_test, total_tests)


def test_llm(
    cfg: DictConfig,
    path_test_dataset: str | Path = "data/test_ru.json",
    test_file: str | Path = "test.json",
    test_func: list[Callable] | None = None,
    llm_url: str | None = "http://localhost:1234/v1/",
    use_ollama: bool = False,
    ollama_client: ollama.Client | None = None,
) -> None:
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
        None

    Raises:
        Logs errors for failed tests and prints accuracy metrics.
    """
    if test_func is None:
        test_func = [test_actions]
    if ollama_client is None:
        client = OpenAI(api_key="dummy", base_url=llm_url)
    else:
        client = ollama_client
    for test in test_func:
        try:
            run_tests(
                cfg=cfg,
                client=client,
                test_dataset_path=path_test_dataset,
                test_file=test_file,
                test_func=test,
                use_ollama=use_ollama,
            )
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception(f"Test function {test.__name__} failed with error")


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
