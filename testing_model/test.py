"""File for testing llm model"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from llama_cpp import Llama
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from training_model.utils import get_user_prompt

# from .deepeval_func import test_mention_number_of_values
from .functions_to_test_game import test_actions


class Content(BaseModel):
    """The inner element of the pydantic schema for testing model"""

    model_config = ConfigDict(extra="forbid")

    Action: str = Field(..., description="The action associated with the message.")


class MainModel(BaseModel):
    """Pydantic schema for testing model"""

    model_config = ConfigDict(extra="forbid")

    MessageText: str = Field(..., description="The text of the message.")
    Content: Content


def dataset_to_json_for_test(dataset: Dict[str, Any], filename: str) -> None:
    """
    Convert a dataset to a JSON file for testing purposes.

    Args:
        dataset (Dict[str, Any]): The dataset containing system and example information.
        filename (str): The path to the output JSON file.

    Returns:
        None
    """
    json_objects: List[Dict[str, str]] = []
    system = dataset["system"]
    dataset = dataset["examples"]

    with open(filename, "w", encoding="utf-8") as file:
        file.write("")

    for row in dataset.keys():
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

    with open(filename, "a", encoding="utf-8") as file:
        file.write(json.dumps(json_objects, indent=4, ensure_ascii=False))


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
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    model_answer = content
    return model_answer


def run_tests(
    cfg: DictConfig,
    client: OpenAI,
    test_dataset_path: str = "data/test_ru.json",
    test_file: str = "test.json",
    test_func: callable = None,
) -> None:
    """
    Runs tests by comparing the LLM responses with expected answers from a dataset.

    Args:
        cfg (DictConfig): Configuration with model settings.
        client (OpenAI): OpenAI client for LLM interaction.
        test_dataset_path (str, optional): Path to the test dataset JSON file.
            Defaults to "data/test_ru.json".
        test_file (str, optional): Path to save the processed test file.
            Defaults to "test.json".
        test_func (callable, optional): Additional test function to execute on each result.
            This function should accept the user prompt, LLM's message text and the correct answer.

    Returns:
        None
    """
    with open(test_dataset_path, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    dataset_to_json_for_test(test_dataset, test_file)

    with open(test_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    prompts_to_check = [prompt["user"] for prompt in prompts]
    expected_answers = [answer["bot"] for answer in prompts]

    count = 0
    passed_test = 0

    for number in range(len(prompts_to_check)):
        prompt = prompts_to_check[number]
        correct_answer = expected_answers[number].strip()
        model_answer = call_llm(
            prompt,
            client=client,
            model=cfg.model.outfile,
        ).strip()

        if test_func is not None:
            try:
                test_func(prompt, model_answer, correct_answer)
                passed_test += 1
            except AssertionError as e:
                logging.error(
                    f"Test failed for prompt: {prompt}.\n Error: {e}\nModel answer: {model_answer}\nExpected answer: {correct_answer}\n"
                )

    total_tests = len(prompts_to_check)
    logging.info("Accuracy: " + str(count / total_tests))
    final_metric = passed_test / total_tests if total_tests > 0 else 0
    logging.info(
        f"Metrics: {final_metric:.2f} ({passed_test}/{total_tests} tests passed)"
    )


def test_via_lmstudio(
    cfg: DictConfig,
    path_test_dataset: str = "data/test_ru.json",
    test_file: str = "test.json",
    test_func: Optional[List[Callable]] = None,
    llm_url: str = "http://localhost:1234/v1/",
) -> None:
    """
    Test the LLM via LM Studio by comparing model responses with expected answers.

    Args:
        cfg (DictConfig): Configuration dictionary containing model settings.
        path_test_dataset (str, optional): Path to the test dataset JSON file.
            Defaults to "data/test_ru.json".
        test_file (str, optional): Path to save the processed test file.
            Defaults to "test.json".
        test_func (Optional[List[Callable]]): List of additional test functions to execute on each result.
        llm_url (str, optional): URL of the LLM service. Defaults to "http://localhost:1234/v1/".

    Returns:
        None

    Raises:
        Logs errors for failed tests and prints accuracy metrics.
    """
    if test_func is None:
        test_func = [test_actions]
    client = OpenAI(api_key="dummy", base_url=llm_url)
    for test in test_func:
        run_tests(
            cfg=cfg,
            client=client,
            test_dataset_path=path_test_dataset,
            test_file=test_file,
            test_func=test,
        )


def llamacpp_execute_test(
    llm,
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
        tuple: Кортеж, содержащий словарь с результатами теста и булевое значение (True, если тест пройден).
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

            if passed:
                logging.info("Test passed")
            else:
                logging.error("Test failed")
                logging.error(f"Expected: {expected_answer}, Got: {predicted_action}")
        except json.JSONDecodeError:
            logging.error("Test failed: Invalid JSON response")
            logging.error(f"Response: {response_text}")
            result = {
                "prompt": prompt,
                "expected": expected_answer,
                "predicted": "ERROR: Invalid JSON",
                "full_response": response_text,
                "passed": False,
            }
            passed = False
    else:
        logging.error("Test failed: No JSON found in response")
        logging.error(f"Response: {response_text}")
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
    model_path: str | bytes,
    test_dataset: str = "data/test_ru.json",
    test_file: str = "test.json",
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    test_func: Callable = llamacpp_execute_test,
    system_prompt: Optional[str] = None,
) -> float:
    """
    Тестирование GGUF модели через llama.cpp с использованием передаваемой функции тестирования.

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
    llm = Llama(
        model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=True
    )
    json_schema = MainModel.model_json_schema()

    with open(test_dataset, "r", encoding="utf-8") as file:
        test_dataset_data = json.load(file)

    dataset_to_json_for_test(test_dataset_data, test_file)

    with open(test_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    prompts_to_check = [prompt["user"] for prompt in prompts]
    answers = [
        test_dataset_data["examples"][bot]["answer"]["Content"]["Action"]
        for bot in test_dataset_data["examples"]
    ]

    logging.debug(f"Expected answers: {answers}")
    logging.debug(f"Number of prompts: {len(prompts_to_check)}")

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
            logging.info(f"Test {number} passed")
        else:
            logging.error(f"Test {number} failed")

    accuracy = count / len(prompts_to_check)
    logging.info(f"Accuracy: {accuracy:.4f} ({count}/{len(prompts_to_check)})")

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=2
        )

    return accuracy
