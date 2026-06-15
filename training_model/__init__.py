import json
import logging
import sys
from pathlib import Path
from typing import Optional  # noqa: F401

__version__ = "0.1.0"
__author__ = "Timur Komolov <komolov.timurka@mail.ru>"
__description__ = (
    "A comprehensive framework for "
    "training Large Language Models using "
    "LoRa (Low-Rank Adaptation) for memory-efficient "
    "fine-tuning"
)

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from evaluation.model_evaluation import dataset_to_json_for_test, test_llm

from .config_validation import get_validation_summary, validate_complete_config
from .logging_config import configure_logging
from .one_file_train import convert_to_gguf, convert_to_rkllm, main_train
from .optuna import optuna_optimize


def get_python_executable() -> str:
    """Возвращает путь к текущему исполняемому файлу Python."""
    return sys.executable


class LLMLoRaCLI:
    """CLI interface for LLM LoRa training pipeline using Fire."""

    def __init__(self) -> None:
        """Initialize CLI with default config directory."""
        self.config_dir = None
        self.cfg: DictConfig | None = None

    def _load_config(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
    ) -> None:
        """Load Hydra configuration programmatically."""
        if self.cfg is not None:
            return
        if config_dir is None:
            # Use relative path from current working directory
            config_dir = Path.cwd() / "conf"

        config_dir = str(Path(config_dir).resolve())

        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(config_name=config_name)

        # Set struct to False to allow dynamic key addition
        from omegaconf import OmegaConf

        OmegaConf.set_struct(cfg, False)
        OmegaConf.register_new_resolver("paths.venv_python_path", get_python_executable)

        self.cfg = cfg

    def _apply_overrides(self, cfg: DictConfig, overrides: dict) -> None:
        """Apply overrides to configuration, filtering Fire-specific params."""
        # Filter out Fire-specific parameters
        fire_params = {"help", "trace", "verbose"}
        filtered_overrides = {k: v for k, v in overrides.items() if k not in fire_params}

        for key, raw_value in filtered_overrides.items():
            value = raw_value
            # Convert string values to appropriate types
            if isinstance(value, str) and value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            elif isinstance(value, str) and value.replace(".", "").replace("-", "").isdigit():
                value = float(value) if "." in value else int(value)

            # Handle nested keys like model.learning_rate
            keys = key.split(".")
            target = cfg
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

    def train(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        **overrides: dict,
    ) -> None:
        """
        Start model training with specified configuration.

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            **overrides: Configuration overrides

        Example:
            python main.py train
            python main.py train model.train_steps=100
        """
        self._load_config(config_name, config_dir)
        self._apply_overrides(self.cfg, overrides)

        configure_logging(self.cfg.logging.log_level)
        logger = logging.getLogger(__name__)

        # Validate configuration before training
        logger.info("Validating configuration...")
        if not validate_complete_config(self.cfg):
            logger.error("Configuration validation failed - aborting training")
            raise ValueError("Configuration validation failed")

        # Use current working directory since get_original_cwd() requires Hydra decorator
        data_dir = Path.cwd() / self.cfg.paths.data_dir
        main_train(data_dir, self.cfg)
        logger.info("[SUCCESS] Training completed successfully!")

    def optimize(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        **overrides: dict,
    ) -> None:
        """
        Run Optuna hyperparameter optimization.

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            **overrides: Configuration overrides

        Example:
            python main.py optimize
            python main.py optimize --n_trials=50
        """
        self._load_config(config_name, config_dir)
        self._apply_overrides(self.cfg, overrides)

        configure_logging(self.cfg.logging.log_level)
        logger = logging.getLogger(__name__)

        # Auto-disable GGUF during Optuna trials to save time and disk space
        if self.cfg.optuna.enabled:
            logger.info("[OPTUNA] Auto-disabling GGUF conversion during optimization trials")
            logger.info(
                "[OPTUNA] (Only training time will be measured, "
                "GGUF will be created for best model separately)"
            )
            self.cfg.model.quant.convert_to_gguf = False

        # Validate configuration before optimization
        logger.info("Validating configuration...")
        if not validate_complete_config(self.cfg):
            logger.error("Configuration validation failed - aborting optimization")
            raise ValueError("Configuration validation failed")

        logger.info("[OPTUNA] Running Optuna hyperparameter optimization command")

        # Use current working directory since get_original_cwd() requires Hydra decorator
        data_dir = Path.cwd() / self.cfg.paths.data_dir
        optuna_optimize(data_dir, self.cfg)
        logger.info("[SUCCESS] Optuna optimization completed successfully!")

    def convert(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        gguf: bool = True,
        rkllm: bool = False,
        **overrides: dict,
    ) -> None:
        """
        Run only model conversion (GGUF and/or RKLLM).

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            gguf: Convert to GGUF format (default: True)
            rkllm: Convert to RKLLM format (default: False)
            **overrides: Configuration overrides

        Example:
            python main.py convert
            python main.py convert --gguf=True --rkllm=True
            python main.py convert --rkllm=True --target_platform=rk3588
        """
        self._load_config(config_name, config_dir)
        self._apply_overrides(self.cfg, overrides)

        configure_logging(self.cfg.logging.log_level)
        logger = logging.getLogger(__name__)

        conversions_performed = []

        if gguf:
            logger.info("[INFO] Converting to GGUF format...")
            # Use the merged model path as source
            model_path = self.cfg.paths.output_dir
            outfile = self.cfg.model.outfile

            convert_to_gguf(
                model_path=model_path,
                outfile=str(Path(model_path) / outfile),
                python_exe=self.cfg.paths.venv_python_path,
                outtype="f16",
                cfg=self.cfg,
            )
            conversions_performed.append("GGUF")

        if rkllm or self.cfg.get("model", {}).get("rkllm", {}).get("enabled", False):
            logger.info("[INFO] Converting to RKLLM format...")
            # Use the merged model path as source and create RKLLM output directory
            model_path = self.cfg.paths.output_dir
            rkllm_output_dir = Path(self.cfg.paths.output_dir) / "rkllm"
            rkllm_output_dir.mkdir(parents=True, exist_ok=True)

            # Get RKLLM parameters from config
            rkllm_config = self.cfg.get("model", {}).get("rkllm", {})
            target_platform = overrides.get(
                "target_platform",
                rkllm_config.get("target_platform", "rk3588"),
            )
            quantization = overrides.get(
                "quantization",
                rkllm_config.get("quantization", "w8a8"),
            )

            convert_to_rkllm(
                model_path=model_path,
                output_dir=rkllm_output_dir,
                target_platform=target_platform,
                quantization=quantization,
                do_parallelize=rkllm_config.get("do_parallelize", False),
                hybrid_quantization=rkllm_config.get("hybrid_quantization", False),
                num_npu_core=rkllm_config.get("num_npu_core", 1),
            )
            conversions_performed.append("RKLLM")

        if conversions_performed:
            logger.info(
                f"[SUCCESS] Model conversion completed: {', '.join(conversions_performed)}",
            )
        else:
            logger.warning(
                "[WARNING] No conversions performed. Use --gguf=True or --rkllm=True",
            )

    def test(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        manual_setup: bool = True,
        **overrides: dict,
    ) -> None:
        """
        Run model testing and evaluation with optional DeepEval metrics.

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            manual_setup: Wait for manual model loading in LM Studio (default: True)
            **overrides: Configuration overrides

        Example:
            python main.py test
            python main.py test --manual_setup=False
            python main.py test testing.deepeval_testing.enabled=true
        """
        self._load_config(config_name, config_dir)
        self._apply_overrides(self.cfg, overrides)

        configure_logging(self.cfg.logging.log_level)
        logger = logging.getLogger(__name__)

        # Import DeepEval integration
        from evaluation.game_evaluation import test_actions

        try:
            from evaluation.deepeval_integration import (
                test_game_context_appropriateness,
                test_russian_language_quality,
                test_unintended_answer_mention,
            )
        except ImportError:
            test_game_context_appropriateness = None
            test_russian_language_quality = None
            test_unintended_answer_mention = None

        # Prepare test dataset
        with Path(self.cfg.testing.test_dataset).open(encoding="utf-8") as file:
            test_dataset = json.load(file)
        dataset_to_json_for_test(test_dataset, self.cfg.testing.output_test_file)

        if manual_setup:
            input("[INFO] Load model into LM Studio and press Enter to continue...")

        # Build test functions list
        test_functions = [test_actions]

        # Add DeepEval metrics if enabled
        deepeval_cfg = self.cfg.get("testing", {}).get("deepeval_testing", {})
        if deepeval_cfg.get("enabled", False) and None not in [
            test_game_context_appropriateness,
            test_russian_language_quality,
            test_unintended_answer_mention,
        ]:
            requested_metrics = deepeval_cfg.get("metrics", [])
            eval_model_type = (
                self.cfg.get("deepeval", {}).get("evaluation_model", {}).get("type", "mistral")
            )
            logger.info(
                f"DeepEval testing enabled. Model type:"
                f" {eval_model_type}, Metrics: {requested_metrics}"
            )

            def wrapper_unintended_answer_mention(
                model_answer: str, correct_answer: str, **kwargs: dict
            ) -> bool:
                user_input = kwargs.get("user_input", "")
                return test_unintended_answer_mention(
                    self.cfg, user_input, model_answer, correct_answer
                )

            def wrapper_russian_language_quality(
                model_answer: str, correct_answer: str, **kwargs: dict
            ) -> bool:
                return test_russian_language_quality(self.cfg, model_answer)

            def wrapper_game_context_appropriateness(
                model_answer: str, correct_answer: str, **kwargs: dict
            ) -> bool:
                available_actions = kwargs.get("available_actions", [])
                user_input = kwargs.get("user_input")
                return test_game_context_appropriateness(
                    self.cfg, model_answer, available_actions, user_input
                )

            metric_mapping = {
                "unintended_answer_mention": wrapper_unintended_answer_mention,
                "russian_language_quality": wrapper_russian_language_quality,
                "game_context_appropriateness": wrapper_game_context_appropriateness,
            }

            for metric_name in requested_metrics:
                if metric_name in metric_mapping:
                    test_functions.append(metric_mapping[metric_name])
                    logger.info(f"✓ Added DeepEval metric: {metric_name}")

        logger.info("[INFO] Running model evaluation...")
        if len(test_functions) == 1:
            logger.info("Running action validation only")
        else:
            logger.info(
                f"Running {len(test_functions)}"
                f" test functions (action validation + "
                f"{len(test_functions)-1} DeepEval metrics)"
            )

        test_llm(
            self.cfg,
            path_test_dataset=self.cfg.testing.test_dataset,
            test_file=self.cfg.testing.output_test_file,
            test_func=test_functions,
        )
        logger.info("[SUCCESS] Testing completed successfully!")

    def pipeline(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        skip_test: bool = False,
        **overrides: dict,
    ) -> None:
        """
        Run complete pipeline: train -> convert -> test.

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            skip_test: Skip testing phase
            **overrides: Configuration overrides

        Example:
            python main.py pipeline
            python main.py pipeline optuna.enabled=true
            python main.py pipeline --skip_test=True
        """
        self._load_config(config_name, config_dir)
        self._apply_overrides(self.cfg, overrides)

        configure_logging(self.cfg.logging.log_level)
        logger = logging.getLogger(__name__)

        logger.info("[INFO] Starting complete LLM LoRa pipeline...")

        # Validate configuration before starting pipeline
        logger.info("Validating configuration...")
        if not validate_complete_config(self.cfg):
            logger.error("Configuration validation failed - aborting pipeline")
            raise ValueError("Configuration validation failed")

        # Training phase - use Optuna if enabled in config
        if self.cfg.optuna.enabled:
            logger.info(
                "[OPTUNA] Optuna optimization enabled - " "starting HPO study with training"
            )
            self.optimize(config_name, config_dir, **overrides)
        else:
            logger.info("[INFO] Standard training mode (Optuna disabled)")
            self.train(config_name, config_dir, **overrides)

        # Conversion phase
        # self.convert(config_name, config_dir, **overrides)

        # Testing phase
        if not skip_test and self.cfg.get("testing", {}).get("manual_lmstudio_test", False):
            self.test(config_name, config_dir, **overrides)

        logger.info("[SUCCESS] Complete pipeline finished successfully!")

    def convert_to_gguf(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        model_path: str | None = None,
        outfile: str | None = None,
        outtype: str = "f16",
        **overrides: dict,
    ) -> None:
        """
        Convert model to GGUF format.

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            model_path: Path to input model directory (default: from config)
            outfile: Output file path (default: from config)
            outtype: Output type specification (default: f16)
            **overrides: Configuration overrides

        Example:
            python main.py convert_to_gguf
            python main.py convert_to_gguf --model_path=models/trained_model
        """
        self._load_config(config_name, config_dir)
        self._apply_overrides(self.cfg, overrides)

        configure_logging(self.cfg.logging.log_level)
        logger = logging.getLogger(__name__)

        logger.info("[INFO] Converting to GGUF format...")

        # Use provided parameters or defaults from config
        model_path = model_path or self.cfg.paths.output_dir
        outfile = outfile or str(Path(model_path) / self.cfg.model.outfile)

        convert_to_gguf(
            model_path=model_path,
            outfile=outfile,
            python_exe=self.cfg.paths.venv_python_path,
            outtype=outtype,
            cfg=self.cfg,
        )
        logger.info("[SUCCESS] GGUF conversion completed successfully!")

    def convert_to_rkllm(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        model_path: str | None = None,
        output_dir: str | None = None,
        target_platform: str | None = None,
        quantization: str | None = None,
        do_parallelize: bool | None = None,
        hybrid_quantization: bool | None = None,
        num_npu_core: int | None = None,
        max_context: int = 4096,
        **overrides: dict,
    ) -> None:
        """
        Convert model to RKLLM format for Rockchip NPU.

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            model_path: Path to input model directory (default: from config)
            output_dir: Directory to save RKLLM model (default: from config)
            target_platform: Target Rockchip platform (default: from config)
            quantization: Quantization type (default: from config)
            do_parallelize: Enable model parallelization (default: from config)
            hybrid_quantization: Enable hybrid quantization (default: from config)
            num_npu_core: Number of NPU cores to use (default: from config)
            max_context: Maximum context length (default: 4096)
            **overrides: Configuration overrides

        Example:
            python main.py convert_to_rkllm
            python main.py convert_to_rkllm --model_path=models/trained_model
            python main.py convert_to_rkllm --target_platform=rk3588 --quantization=w8a8
        """
        self._load_config(config_name, config_dir)
        self._apply_overrides(self.cfg, overrides)

        configure_logging(self.cfg.logging.log_level)
        logger = logging.getLogger(__name__)

        logger.info("[INFO] Converting to RKLLM format...")

        # Get RKLLM config defaults
        rkllm_config = self.cfg.get("model", {}).get("rkllm", {})

        # Use provided parameters or defaults from config
        model_path = model_path or self.cfg.paths.output_dir
        output_dir = output_dir or str(Path(self.cfg.paths.output_dir) / "rkllm")
        target_platform = target_platform or rkllm_config.get("target_platform", "rk3588")
        quantization = quantization or rkllm_config.get("quantization", "w8a8")
        do_parallelize = (
            do_parallelize
            if do_parallelize is not None
            else rkllm_config.get("do_parallelize", False)
        )
        hybrid_quantization = (
            hybrid_quantization
            if hybrid_quantization is not None
            else rkllm_config.get("hybrid_quantization", False)
        )
        num_npu_core = (
            num_npu_core if num_npu_core is not None else rkllm_config.get("num_npu_core", 1)
        )

        result_path = convert_to_rkllm(
            model_path=model_path,
            output_dir=output_dir,
            target_platform=target_platform,
            quantization=quantization,
            do_parallelize=do_parallelize,
            hybrid_quantization=hybrid_quantization,
            num_npu_core=num_npu_core,
            max_context=max_context,
        )
        logger.info(f"[SUCCESS] RKLLM conversion completed: {result_path}")


__all__ = [
    "LLMLoRaCLI",
    "__author__",
    "__description__",
    "__version__",
    "configure_logging",
    "get_validation_summary",
    "main_train",
    "optuna_optimize",
    "validate_complete_config",
]
