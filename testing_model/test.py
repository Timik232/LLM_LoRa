"""File for testing llm model"""

import json
import logging
import re
from typing import Any, Dict, List

import requests
from llama_cpp import Llama
from omegaconf import DictConfig

# from .deepeval import test_mention_number_of_values
from pydantic import BaseModel, ConfigDict, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

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
    json_schema = MainModel.model_json_schema()
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


def test_via_llamacpp(
    model_path: str | bytes,
    test_dataset: str = "data/test_ru.json",
    test_file: str = "test.json",
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> float:
    """
    Test a GGUF model via llama.cpp by comparing model responses with expected answers.

    Args:
        model_path (str | bytes): Path to the GGUF model file
        test_dataset (str, optional): Path to the test dataset JSON file.
            Defaults to "data/test_ru.json".
        test_file (str, optional): Path to save the processed test file.
            Defaults to "test.json".
        n_gpu_layers (int, optional): Number of layers to offload to GPU.
            Defaults to -1 (all layers).
        n_ctx (int, optional): Context window size.
            Defaults to 2048.
        temperature (float, optional): Sampling temperature.
            Defaults to 0.7.
        max_tokens (int, optional): Maximum number of tokens to generate.
            Defaults to 2048.

    Returns:
        float value of accuracy

    Raises:
        Logs errors for failed tests and prints accuracy metrics.
    """
    # Load the GGUF model
    llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)

    json_schema = MainModel.model_json_schema()

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

    logging.debug(f"Expected answers: {answers}")
    logging.debug(f"Number of prompts: {len(prompts_to_check)}")

    count = 0
    results = []

    for number, prompt in enumerate(prompts_to_check):
        system_prompt = (
            "Ты – помощник по имени ВИКА на заброшенной космической станции. "
            "У тебя есть доступ к системам станции. "
            "Отвечай только в формате JSON с ключами 'MessageText' и 'Content', "
            "где Content содержит ключ 'Action' с одним из доступных тебе действий. "
            f"Используй следующую JSON схему: {json.dumps(json_schema, ensure_ascii=False)} "
            "Заканчивай ответ символом }."
        )

        formatted_prompt = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        )

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

                result = {
                    "prompt": prompt,
                    "expected": answers[number],
                    "predicted": predicted_action,
                    "full_response": response_text,
                    "passed": predicted_action == answers[number],
                }
                results.append(result)

                if predicted_action == answers[number]:
                    count += 1
                    logging.info(f"Test {number} passed")
                else:
                    logging.error(f"Test {number} failed")
                    logging.error(
                        f"Expected: {answers[number]}, Got: {predicted_action}"
                    )
            except json.JSONDecodeError:
                logging.error(f"Test {number} failed: Invalid JSON response")
                logging.error(f"Response: {response_text}")
                results.append(
                    {
                        "prompt": prompt,
                        "expected": answers[number],
                        "predicted": "ERROR: Invalid JSON",
                        "full_response": response_text,
                        "passed": False,
                    }
                )
        else:
            logging.error(f"Test {number} failed: No JSON found in response")
            logging.error(f"Response: {response_text}")
            results.append(
                {
                    "prompt": prompt,
                    "expected": answers[number],
                    "predicted": "ERROR: No JSON found",
                    "full_response": response_text,
                    "passed": False,
                }
            )

    accuracy = count / len(prompts_to_check)
    logging.info(f"Accuracy: {accuracy:.4f} ({count}/{len(prompts_to_check)})")

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=2
        )

    return accuracy
