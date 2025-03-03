import json
import logging

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from .test import dataset_to_json_for_test


def test_mention_number_of_values(user_input: str, output: str):
    """check if model mention the number of values"""
    metric = GEval(
        name="Mention number of values",
        criteria="Check that the model doesn't mention the number of values in the answer.",
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
    total_tests = len(prompts_to_check)
    passed_tests = 0
    for user_input, answer in zip(prompts_to_check, answers):
        try:
            test_mention_number_of_values(user_input, answer)
            passed_tests += 1
        except AssertionError:
            logging.error(
                f"Тест не пройден для запроса: {user_input}. \nОтвет модели {answer}."
            )

    final_metric = passed_tests / total_tests if total_tests > 0 else 0
    logging.info(
        f"Итоговая метрика: {final_metric:.2f} ({passed_tests}/{total_tests} тестов пройдено)"
    )
