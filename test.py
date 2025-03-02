import json
import logging
import os
import requests
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from utils import get_user_prompt


def dataset_to_json_for_test(dataset, filename):
    json_objects = []
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

def sanity_check(llm_url: str, prompts_to_check: list, answers: list):
    pass

def test_via_lmstudio():
    llm_url = "http://localhost:1234/v1/chat/completions"
    with open(os.path.join("data", "test_ru.json"), "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    dataset_to_json_for_test(test_dataset, "test.json")
    with open("test.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompts_to_check = [prompt["user"] for prompt in prompts]
    answers = [test_dataset["examples"][bot]["answer"]["Content"]["Action"] for bot in test_dataset["examples"]]
    logging.debug(answers)
    logging.debug(len(prompts_to_check))
    count = 0
    for number, prompt in enumerate(prompts_to_check):
        data = {'messages': [{'role': 'user', 'content': prompt}]}
        response = requests.post(llm_url, json=data)
        logging.debug(response.json())
        if json.loads(response.json()["choices"][0]["message"]["content"])["Content"]["Action"] == answers[number]:
            count += 1
            logging.info(f"Test {number} passed")
        else:
            logging.error(f"Test {number} failed")

    logging.info("accuracy: " + str(count/len(prompts_to_check)))


def test_via_vllm(llm: LLM):
    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "MessageText": {
                "type": "string",
                "description": "The text of the message."
            },
            "Content": {
                "type": "object",
                "properties": {
                    "Action": {
                        "type": "string",
                        "description": "The action associated with the message."
                    }
                },
                "required": ["Action"],
                "additionalProperties": False
            }
        },
        "required": ["MessageText", "Content"],
        "additionalProperties": False
    }
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
    with open(os.path.join("data", "test_ru.json"), "r", encoding="utf-8") as file:
        test_dataset = json.load(file)
    dataset_to_json_for_test(test_dataset, "test.json")
    with open("test.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompts_to_check = [prompt["user"] for prompt in prompts]
    answers = [test_dataset["examples"][bot]["answer"]["Content"]["Action"] for bot in test_dataset["examples"]]
    logging.debug(answers)
    logging.debug(len(prompts_to_check))
    count = 0
    for number, prompt in enumerate(prompts_to_check):
        data =  [
            {
                "role": "system",
                "content": "Ты – помощник по имени ВИКА на заброшенной космической станции. У тебя есть доступ к системам станции. Отвечай только в формате JSON с ключами 'MessageText' и 'Actions', содержащими как минимум одно (или несколько) доступных вам действий. Если в Actions есть имя действия, оно будет исполнено. Заканчивайте ответ символом }. Ниже – история сообщений из предыдущего диалога с пользователем, а также список доступных тебе действий.",
            },
            {'role': 'user', 'content': prompt}
        ]
        response = llm.chat(messages=data, sampling_params=sampling_params)
        logging.debug(response[0].outputs[0].text)
        if json.loads(response[0].outputs[0].text)["Content"]["Action"] == answers[number]:
            count += 1
            logging.info(f"Test {number} passed")
        else:
            logging.error(f"Test {number} failed")

    logging.info("accuracy: " + str(count/len(prompts_to_check)))


