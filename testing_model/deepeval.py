import asyncio
import json
import logging
import subprocess
import time
from typing import Optional

import requests
from deepeval import assert_test
from deepeval.metrics import GEval

# from .test import dataset_to_json_for_test
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI
from mistralai import Mistral

from private_api import MISTRAL_API


class CustomLocalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model="vikhr-yandexgpt-5-lite-8b-it_gguf",
        url="http://localhost:1234/v1/",
        *args,
        **kwargs,
    ):
        self.model = ChatOpenAI(
            base_url=url,
            api_key="dummy",
            model=model,
        )
        self.model_name = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


class CustomMistralModel(DeepEvalBaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        temperature: float = 0.1,
        *args,
        **kwargs,
    ):
        self.client = Mistral(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.last_request_time: Optional[float] = None
        self.rate_limit_delay = 1.2  # 1.2 seconds to stay safely under limit

    def _enforce_rate_limit(self):
        """Ensure we wait at least 1 second between requests"""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                time.sleep(sleep_time)
        self.last_request_time = time.time()

    async def _aenforce_rate_limit(self):
        """Async version of rate limiting"""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                await asyncio.sleep(sleep_time)
        self.last_request_time = time.time()

    def generate(self, prompt: str) -> str:
        self._enforce_rate_limit()
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        await self._aenforce_rate_limit()
        response = await self.client.chat.complete_async(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def get_model_name(self):
        return self.model_name

    def load_model(self):
        return self.client


mistral_model = CustomMistralModel(
    api_key=MISTRAL_API, model="mistral-small-latest", temperature=0.7
)

local_model = CustomLocalModel()


def set_local_model_via_cli(
    model_name="vikhr-yandexgpt-5-lite-8b-it_gguf", base_url="http://localhost:1234/v1"
):
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
            "Команда не найдена. Убедитесь, что Python и deepeval установлены и доступны в PATH."
        )
        print(e)


def test_mention_number_of_values(user_input: str, output: str):
    """check if model mention the number of values"""
    metric = GEval(
        name="Mention number of values",
        criteria="Check that the model does not itself write to the user 'there is six values'.",
        # "Also check that the model does not write to the user the correct answer for the"
        # "question from the VIKA",
        # criteria="Проверьте, что модель не пишет сама пользователю конкретное количество ценностей в ответе"
        #          "Также проверьте, что модель не пишет пользователю правильный ответ на свой вопрос, который"
        #          "от него ожидает услышать.",
        evaluation_steps=[
            "Check that the model does not write the number of values by itself",
            # "Check that the Actual Output does not provide the correct answer to the VIKA question as specified in the Input.",
            # "Confirm that the Actual Output does not directly answer the question from the VIKA, even if user want it."
        ],
        model=mistral_model,
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


def test_from_dataset(test_dataset="data/test_ru.json", test_file="test.json"):
    """Test for the dataset"""
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
