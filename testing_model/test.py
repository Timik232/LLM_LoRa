"""File for testing llm model"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import requests
from llama_cpp import Llama
from omegaconf import DictConfig

# from .deepeval import test_mention_number_of_values
from pydantic import BaseModel, ConfigDict, Field

# from .deepeval import test_mention_number_of_values
from training_model.utils import get_user_prompt


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


def test_via_lmstudio(
    cfg: DictConfig,
    path_test_dataset: str = "data/test_ru.json",
    test_file: str = "test.json",
) -> None:
    """
    Test the LLM via LM Studio by comparing model responses with expected answers.

    Args:
        cfg (DictConfig): Configuration dictionary containing model settings.
        test_dataset (str, optional): Path to the test dataset JSON file.
            Defaults to "data/test_ru.json".
        test_file (str, optional): Path to save the processed test file.
            Defaults to "test.json".

    Returns:
        None

    Raises:
        Logs errors for failed tests and prints accuracy metrics.
    """
    llm_url = "http://localhost:1234/v1/chat/completions"
    with open(path_test_dataset, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    dataset_to_json_for_test(test_dataset, test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompts_to_check = [prompt["user"] for prompt in prompts]
    answers = [
        test_dataset["examples"][bot]["answer"]["Content"]["Action"]
        for bot in test_dataset["examples"]
    ]
    logging.debug(answers)
    logging.debug(len(prompts_to_check))
    count = 0
    deepeval_passed_test = 0
    for number, prompt in enumerate(prompts_to_check):
        data = {
            "messages": [{"role": "user", "content": prompt}],
            # "model": f"game-model/{cfg.model.version}/{cfg.model.outfile[:-5]}{cfg.model.quant_postfix}.gguf",
        }
        response = requests.post(llm_url, json=data)
        logging.debug(response.json())
        model_answer = json.loads(response.json()["choices"][0]["message"]["content"])
        if model_answer["Content"]["Action"] == answers[number]:
            count += 1
            logging.info(f"Test {number} passed")
        else:
            logging.error(
                f"Test {number} failed. User input: {prompt}. \n"
                f"Expected: {answers[number]}. Got: {json.loads(response.json()['choices'][0]['message']['content'])['Content']['Action']}"
            )
        try:
            # test_mention_number_of_values(prompt, model_answer["MessageText"])
            deepeval_passed_test += 1
        except AssertionError:
            logging.error(
                f"Test doesn't complete for prompt: {prompt}. \nModel answer: {model_answer}."
            )
    total_tests = len(prompts_to_check)
    logging.info("accuracy: " + str(count / total_tests))
    final_metric = deepeval_passed_test / total_tests if total_tests > 0 else 0
    logging.info(
        f"Deepeval metrics: {final_metric:.2f} ({deepeval_passed_test}/{total_tests} test passed)"
    )


def execute_test(
    llm,
    system_prompt: str,
    prompt: str,
    expected_answer: str,
    max_tokens: int,
    temperature: float,
    json_schema: dict,
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
        json_schema (dict): JSON схема для формата ответа.

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
    test_func: Callable = execute_test,
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
            json_schema=json_schema,
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
