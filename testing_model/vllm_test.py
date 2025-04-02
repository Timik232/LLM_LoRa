import json
import logging

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .test import MainModel, dataset_to_json_for_test


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
