"""File for testing llm model"""

import json
import logging
from typing import Any, Dict, List

import requests
from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# from .deepeval import test_mention_number_of_values
from training_model.utils import get_user_prompt

# from .deepeval import test_mention_number_of_values


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


def test_via_vllm(
    llm: LLM, test_dataset: str = "data/test_ru.json", test_file: str = "test.json"
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
    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "MessageText": {
                "type": "string",
                "description": "The text of the message.",
            },
            "Content": {
                "type": "object",
                "properties": {
                    "Action": {
                        "type": "string",
                        "description": "The action associated with the message.",
                    }
                },
                "required": ["Action"],
                "additionalProperties": False,
            },
        },
        "required": ["MessageText", "Content"],
        "additionalProperties": False,
    }
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
    with open(test_dataset, "r", encoding="utf-8") as file:
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
    for number, prompt in enumerate(prompts_to_check):
        data = [
            {
                "role": "system",
                "content": "Ты – помощник по имени ВИКА на заброшенной космической станции. У тебя есть доступ к системам станции. Отвечай только в формате JSON с ключами 'MessageText' и 'Actions', содержащими как минимум одно (или несколько) доступных вам действий. Если в Actions есть имя действия, оно будет исполнено. Заканчивайте ответ символом }. Ниже – история сообщений из предыдущего диалога с пользователем, а также список доступных тебе действий.",
            },
            {"role": "user", "content": prompt},
        ]
        response = llm.chat(messages=data, sampling_params=sampling_params)
        logging.debug(response[0].outputs[0].text)
        if (
            json.loads(response[0].outputs[0].text)["Content"]["Action"]
            == answers[number]
        ):
            count += 1
            logging.info(f"Test {number} passed")
        else:
            logging.error(f"Test {number} failed")

    logging.info("accuracy: " + str(count / len(prompts_to_check)))
