import json
import logging
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import ollama

# from pydantic import BaseModel
from .test import MainModel, dataset_to_json_for_test


def ollama_generate(model_name: str, prompt: str, schema: Dict, options: Dict) -> Dict:
    """
    Wrapper function to generate a response using Ollama's structured outputs.

    Args:
        model_name (str): The name of the model in Ollama.
        prompt (str): The formatted prompt to send to the model.
        schema (Dict): The JSON schema for the expected response.
        options (Dict): Additional options for the generate call.

    Returns:
        Dict: The parsed JSON response conforming to the schema.
    """
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        format="json",
        options=options,
        schema=schema,
    )
    return json.loads(response["response"])


def execute_test(
    llm: Callable,
    system_prompt: str,
    prompt: str,
    expected_answer: str,
    max_tokens: int,
    temperature: float,
    json_schema: Dict,
) -> Tuple[Dict, bool]:
    """
    Executes a test for a single query.

    Args:
        llm (Callable): Function to generate responses, compatible with the specified format.
        system_prompt (str): System prompt with instructions.
        prompt (str): User query.
        expected_answer (str): Expected result.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        json_schema (Dict): JSON schema for the response format.

    Returns:
        Tuple: A tuple containing a dictionary with test results and a boolean (True if the test passed).
    """
    formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"

    options = {"num_predict": max_tokens, "temperature": temperature, "stop": ["</s>"]}

    try:
        json_response = llm(
            prompt=formatted_prompt, schema=json_schema, options=options
        )
        predicted_action = json_response["Content"]["Action"]
        passed = predicted_action == expected_answer

        result = {
            "prompt": prompt,
            "expected": expected_answer,
            "predicted": predicted_action,
            "full_response": json_response,
            "passed": passed,
        }

        if passed:
            logging.info("Test passed")
        else:
            logging.error("Test failed")
            logging.error(f"Expected: {expected_answer}, Got: {predicted_action}")
    except (KeyError, json.JSONDecodeError) as e:
        logging.error(f"Test failed: Invalid response format - {str(e)}")
        logging.error(f"Response: {json_response}")
        result = {
            "prompt": prompt,
            "expected": expected_answer,
            "predicted": "ERROR: Invalid response",
            "full_response": json_response,
            "passed": False,
        }
        passed = False

    return result, passed


def test_via_ollama(
    model_name: str,
    test_dataset: str = "data/test_ru.json",
    test_file: str = "test.json",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    test_func: Callable = execute_test,
    system_prompt: Optional[str] = None,
) -> float:
    """
    Tests a model via Ollama using the provided test function and JSON schema.

    Args:
        model_name (str): Name of the model in Ollama (e.g., 'llama2').
        test_dataset (str, optional): Path to the JSON file with test data. Defaults to "data/test_ru.json".
        test_file (str, optional): Path to save the processed test file. Defaults to "test.json".
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 2048.
        test_func (Callable): Function implementing the testing logic.
        system_prompt (Optional[str], optional): System prompt for the model.

    Returns:
        float: Accuracy score.

    Note:
        Assumes the Ollama server is running and the model is loaded via `ollama pull <model_name>`.
    """
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
            "You are an assistant named VIKA on an abandoned space station. "
            "You have access to the station's systems. "
            "Respond in JSON format conforming to the specified schema."
        )

    llm = partial(ollama_generate, model_name)

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
