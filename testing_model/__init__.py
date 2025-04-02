from .deepeval_func import test_from_dataset, test_mention_number_of_values
from .ollama_test import test_via_ollama
from .test import test_llm, test_via_llamacpp

__all__ = [
    "test_llm",
    "test_mention_number_of_values",
    "test_from_dataset",
    "test_via_llamacpp",
    "test_via_ollama",
]
