"""Custom exception classes for the LLM-LoRa training framework.

This module provides a hierarchy of specific exception classes to replace
generic Exception handling throughout the codebase, enabling better error
diagnosis, handling, and debugging.
"""


class LLMLoRaError(Exception):
    """Base exception class for all LLM-LoRa framework errors.

    All other custom exceptions in this framework inherit from this class,
    providing a common base for catching framework-specific errors.
    """


class ConfigurationError(LLMLoRaError):
    """Raised when configuration validation or loading fails.

    This includes:
    - Invalid configuration file format
    - Missing required configuration parameters
    - Invalid parameter values or ranges
    - Configuration compatibility issues
    """


class ModelLoadingError(LLMLoRaError):
    """Raised when model loading or initialization fails.

    This includes:
    - HuggingFace model not found or inaccessible
    - Model architecture incompatibility
    - Insufficient memory for model loading
    - Tokenizer loading failures
    - Quantization configuration errors
    """


class TrainingError(LLMLoRaError):
    """Raised when training process encounters failures.

    This includes:
    - Training data preparation failures
    - Training loop execution errors
    - Evaluation failures
    - Checkpoint saving/loading issues
    - GPU memory allocation errors during training
    """


class ConversionError(LLMLoRaError):
    """Raised when model format conversion fails.

    This includes:
    - GGUF conversion failures
    - RKLLM conversion failures
    - Model merging errors
    - Quantization process failures
    - Output file creation issues
    """


class DataProcessingError(LLMLoRaError):
    """Raised when data preparation or processing fails.

    This includes:
    - Dataset loading failures
    - Data preprocessing errors
    - Tokenization issues
    - Data format validation failures
    - Train/test split errors
    """


class ExperimentTrackingError(LLMLoRaError):
    """Raised when experiment tracking operations fail.

    This includes:
    - MLflow connection failures
    - Weights & Biases initialization errors
    - Metric logging failures
    - Artifact upload issues
    """


class ResourceManagementError(LLMLoRaError):
    """Raised when resource management operations fail.

    This includes:
    - GPU memory management errors
    - Disk space allocation failures
    - Temporary directory creation issues
    - Resource cleanup failures
    """
