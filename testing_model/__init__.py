from .deepeval_func import test_from_dataset, test_mention_number_of_values
from .ollama_test import test_via_ollama
from .test import test_via_llamacpp, test_via_lmstudio

__all__ = [
    "test_via_lmstudio",
    "test_mention_number_of_values",
    "test_from_dataset",
    "test_via_llamacpp",
    "test_via_ollama",
]
