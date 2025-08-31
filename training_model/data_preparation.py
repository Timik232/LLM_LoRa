"""
Module with utility functions for training the model.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def get_user_prompt(data: str) -> str:
    """
    Construct a user prompt from conversation data.

    Args:
        data (str): conversation data

    Returns:
        str: Formatted prompt string with conversation context.
    """
    prompt = f"Ответь на вопрос пользователя коротко: {data}"
    return prompt


def dataset_to_json(
    dataset: Dict[str, Any], filename: str, method: str = "classic"
) -> List[Dict[str, str]]:
    """
    Convert dataset to JSON lines format and save to a file.

    Args:
        dataset (Dict[str, Any]): Source dataset containing:
            - 'system': System prompt template
            - 'examples': Dictionary of conversation examples
        filename (str): Output file path where JSON lines are written.
        method (str): Data preparation method - "classic" or "game". Defaults to "classic".

    Returns:
        List[Dict[str, str]]: List of JSON objects representing each example.
    """
    json_objects: List[Dict[str, str]] = []
    system_template = dataset.get("system", "")
    examples = dataset.get("examples", {})

    # Import the game get_user_prompt if needed
    if method == "game":
        from evaluation.model_evaluation import get_user_prompt as game_get_user_prompt
    else:
        game_get_user_prompt = None

    # Initialize (or clear) the output file
    with open(filename, "w", encoding="utf-8"):
        pass

    for example in examples:
        system_message = system_template

        # Use different data preparation methods based on configuration
        if method == "game":
            # Game method: expects "prompt" field with complex structure
            user_message = game_get_user_prompt(example.get("prompt", {}))
            bot_message = str(example.get("answer", ""))
        else:
            # Classic method: simple instruction/output format
            user_message = get_user_prompt(example.get("instruction", {}))
            bot_message = str(example.get("output", ""))

        json_object = {
            "system": system_message,
            "user": user_message,
            "bot": bot_message,
        }
        json_objects.append(json_object)

        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_object, ensure_ascii=False) + "\n")

    return json_objects


def transform_topics(topics: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Transforms a dictionary of topics with examples and responses into
    a list of dictionaries with 'instruction' and 'output' keys.

    Args:
        topics (Dict[str, Any]): A dictionary where each key is a topic ID
            and each value is a dict containing 'examples' (list of str)
            and 'responses' (list of str).

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing:
            - 'instruction': one of the example strings
            - 'output': one of the response strings
    """
    transformed: List[Dict[str, str]] = []
    link_pattern = re.compile(r"\b(?:https?|ftp)://[^\s\"'<>(){}|\\^`[\]]+")

    for topic_data in topics.values():
        examples = topic_data.get("examples", [])
        responses = topic_data.get("responses", [])
        for example in examples:
            for response in responses:
                if not link_pattern.search(response):
                    transformed.append({"instruction": example, "output": response})

    return transformed


def do_transform():
    """One run function to convert the dataset."""
    base_dir = Path(__file__).resolve().parent.parent

    input_file_path = base_dir / "data" / "intents_dataset.json"
    with input_file_path.open("r", encoding="utf-8") as f:
        input_json = json.load(f)

    result = transform_topics(input_json)

    output_file_path = base_dir / "data" / "intent_responses.json"
    with output_file_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("Converted to the new format")


if __name__ == "__main__":
    do_transform()
