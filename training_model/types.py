"""Type definitions and protocols for the LLM-LoRa training framework.

This module provides common type aliases, protocols, and type definitions
used throughout the codebase to improve type safety and IDE support.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, TypedDict, Union

from datasets import Dataset
from omegaconf import DictConfig
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Model and tokenizer type aliases
ModelType = Union[AutoModelForCausalLM, PreTrainedModel, Any]
TokenizerType = Union[AutoTokenizer, PreTrainedTokenizer]

# Configuration and data types
ConfigDict = dict[str, Any]


class TrainingResult(TypedDict):
    global_steps: int
    eval_loss: float


# Data processing types
DataPoint = dict[str, str]
TokenizedData = dict[str, Tensor | list[int]]
DatasetTuple = tuple[Dataset, Dataset]

# File and path types
PathLike = Union[str, Path]
ModelPath = Union[str, Path]
OutputPath = Union[str, Path]


# Training configuration types
class LoRaParams(TypedDict):
    r: int
    alpha: int
    dropout: float


class QuantizationConfig(TypedDict):
    enabled: bool
    qtype: str
    use_8bit: bool


# Conversion and processing results
class ConversionResult(TypedDict):
    success: bool
    output_path: str | None
    error_message: str | None


class ProcessingArtifacts(TypedDict):
    model_path: str
    checkpoint_path: str
    merged_path: str | None


# Function type definitions
TokenizeFunction = Callable[[str], TokenizedData]
PromptGenerator = Callable[[DataPoint], str]
DataPreparationFunction = Callable[[DictConfig, TokenizerType], DatasetTuple]


# Training strategy protocol
class TrainingStrategy(Protocol):
    """Protocol for different training strategies (SFT, GRPO, DPO)."""

    def train(
        self,
        model: ModelType,
        tokenizer: TokenizerType,
        config: DictConfig,
        train_data: Dataset,
        val_data: Dataset,
    ) -> TrainingResult:
        """Execute the training strategy."""
        ...


# Model converter protocol
class ModelConverter(Protocol):
    """Protocol for different model conversion formats."""

    def convert(
        self,
        model_path: PathLike,
        output_path: PathLike,
        **kwargs: Any,
    ) -> ConversionResult:
        """Convert model to specific format."""
        ...


# Resource manager protocol
class ResourceManager(Protocol):
    """Protocol for managing computational resources."""

    def __enter__(self) -> None:
        """Enter resource context."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit resource context with cleanup."""
        ...


# Evaluation types
EvaluationMetrics = dict[str, int | float | str]


class TestResult(TypedDict):
    metrics: EvaluationMetrics
    success: bool
    error_message: str | None


# Logging and tracking types
ExperimentConfig = dict[str, Any]
LoggingBackend = Union[str, None]  # Can be "wandb", "mlflow", or None


# Pipeline step results
class PipelineStepResult(TypedDict):
    step_name: str
    success: bool
    duration: float
    artifacts: dict[str, Any] | None
    error_message: str | None


# Context manager types
DirectoryContext = AbstractContextManager[None]
GPUMemoryContext = AbstractContextManager[None]


# Generation and sampling types
class GenerationConfig(TypedDict):
    temperature: float
    top_k: int
    top_p: float
    max_length: int
    do_sample: bool


# RKLLM specific types
class RKLLMConfig(TypedDict):
    target_platform: str
    quantization: str
    do_parallelize: bool
    hybrid_quantization: bool
    num_npu_core: int
