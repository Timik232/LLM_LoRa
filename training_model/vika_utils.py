"""
Module with utility functions for training the model.
"""
import json
from typing import Any, Dict, List


def get_user_prompt(data: Dict[str, Any]) -> str:
    """
    Construct a user prompt from conversation data.

    Args:
        data (Dict[str, Any]): Dictionary containing conversation history and metadata:
            - History: List of previous messages
            - AvailableActions: List of available actions
            - UserInput: Current user input

    Returns:
        str: Formatted prompt string with conversation context.
    """
    prompt = (
        "Системное сообщение, которому ты должен следовать, отмечено словом 'system'. "
        "Предыдущие сообщения пользователя отмечены словом 'user'. "
        "Твои предыдущие сообщения отмечены словом 'VIKA'."
        "\n\nИстория сообщений:"
    )
    for message in data.get("History", []):
        prompt += f"\n{message}"
    prompt += (
        "\n\nТы можешь совершать только действия из представленного списка.\n"
        f"Доступные действия: Разговор, {', '.join(data.get('AvailableActions', []))}"
    )
    prompt += (
        "\n\nОтветь на сообщение пользователя, беря во внимания всю предыдущую информацию.\n"
        f"Сообщение пользователя: {data.get('UserInput', '')}"
    )
    return prompt


def dataset_to_json(dataset: Dict[str, Any], filename: str) -> List[Dict[str, str]]:
    """
    Convert dataset to JSON lines format and save to a file.

    Args:
        dataset (Dict[str, Any]): Source dataset containing:
            - 'system': System prompt template
            - 'examples': Dictionary of conversation examples
        filename (str): Output file path where JSON lines are written.

    Returns:
        List[Dict[str, str]]: List of JSON objects representing each example.
    """
    json_objects: List[Dict[str, str]] = []
    system_template = dataset.get("system", "")
    examples = dataset.get("examples", {})

    # Initialize (or clear) the output file
    with open(filename, "w", encoding="utf-8"):
        pass

    for _, example in examples.items():
        system_message = system_template
        user_message = get_user_prompt(example.get("prompt", {}))
        bot_message = str(example.get("answer", ""))

        json_object = {
            "system": system_message,
            "user": user_message,
            "bot": bot_message,
        }
        json_objects.append(json_object)

        # Append JSON object per line
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_object, ensure_ascii=False) + "\n")

    return json_objects
