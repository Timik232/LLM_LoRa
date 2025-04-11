import json
from json import JSONDecodeError


def test_actions(_: str, model_answer: str, correct_answer: str):
    """
    Test if the model's answer matches the expected action.
    Args:
        _ (str): doesn't use, but added for the correct integration with common tests
        model_answer (str): The model's answer in JSON format.
        correct_answer (str): The expected answer from model. Should be in json format
            with "content" and inside "action" keys

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
        assert predicted_action == correct_action, "Action doesn't match"
    except JSONDecodeError as err:
        error_msg = "Model answer or correct answer is not in JSON format"
        raise AssertionError(error_msg) from err
    except KeyError as err:
        error_msg = "Model answer or correct answer doesn't have correct keys"
        raise AssertionError(error_msg) from err
    except Exception as e:
        error_msg = f"Error in test_actions: {e}"
        raise AssertionError(error_msg) from e
