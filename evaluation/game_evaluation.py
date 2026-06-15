"""Game evaluation functions for model testing"""

import json
from json import JSONDecodeError


def test_actions(model_answer: str, correct_answer: str, **kwargs: object) -> None:
    """
    Test if the model's answer matches the expected action.
    Args:
        model_answer (str): The model's answer in JSON format.
        correct_answer (str): The expected answer from model. Should be in json format
            with "content" and inside "action" keys
        **kwargs: Additional arguments (unused but allows for flexible integration)

    Returns:
        None

    Raises:
        AssertionError
    """

    try:
        correct_answer = correct_answer.replace("'", '"')
        model_answer_dict = json.loads(model_answer)
        correct_answer_dict = json.loads(correct_answer)
        predicted_action = model_answer_dict["Content"]["Action"]
        correct_action = correct_answer_dict["Content"]["Action"]
        if predicted_action != correct_action:
            _raise_action_mismatch()
    except JSONDecodeError as err:
        error_msg = "Model answer or correct answer is not in JSON format"
        raise AssertionError(error_msg) from err
    except KeyError as err:
        error_msg = "Model answer or correct answer doesn't have correct keys"
        raise AssertionError(error_msg) from err
    except Exception as e:
        error_msg = f"Error in test_actions: {e}"
        raise AssertionError(error_msg) from e


def _raise_action_mismatch() -> None:
    """Helper to raise a standardized AssertionError for action mismatch."""
    raise AssertionError("Action doesn't match")
