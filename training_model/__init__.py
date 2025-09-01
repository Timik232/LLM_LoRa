import json
import logging
from pathlib import Path
from typing import Optional  # noqa: F401

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from evaluation.model_evaluation import dataset_to_json_for_test, test_llm

from .config_validation import get_validation_summary, validate_complete_config
from .logging_config import configure_logging
from .one_file_train import convert_to_gguf, convert_to_rkllm, main_train
from .optuna import optuna_optimize


class LLMLoRaCLI:
    """CLI interface for LLM LoRa training pipeline using Fire."""

    def __init__(self):
        """Initialize CLI with default config directory."""
        self.config_dir = None
        self.cfg = None

    def _load_config(
        self, config_name: str = "config", config_dir: str | None = None
    ) -> DictConfig:
        """Load Hydra configuration programmatically."""
        if config_dir is None:
            # Use relative path from current working directory
            config_dir = Path.cwd() / "conf"

        config_dir = str(Path(config_dir).resolve())

        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(config_name=config_name)

        # Set struct to False to allow dynamic key addition
        from omegaconf import OmegaConf

        OmegaConf.set_struct(cfg, False)

        return cfg

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

    def train(self, config_name: str = "config", config_dir: str | None = None, **overrides):
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
        cfg = self._load_config(config_name, config_dir)
        self._apply_overrides(cfg, overrides)

        configure_logging(logging.DEBUG)
        logger = logging.getLogger(__name__)

        # Validate configuration before training
        logger.info("Validating configuration...")
        if not validate_complete_config(cfg):
            logger.error("Configuration validation failed - aborting training")
            raise ValueError("Configuration validation failed")

        # Use current working directory since get_original_cwd() requires Hydra decorator
        data_dir = Path.cwd() / cfg.paths.data_dir
        main_train(data_dir, cfg)
        logger.info("[SUCCESS] Training completed successfully!")

    def optimize(
        self, config_name: str = "config", config_dir: str | None = None, **overrides
    ):
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
        cfg = self._load_config(config_name, config_dir)
        self._apply_overrides(cfg, overrides)

        configure_logging(logging.DEBUG)
        logger = logging.getLogger(__name__)

        # Validate configuration before optimization
        logger.info("Validating configuration...")
        if not validate_complete_config(cfg):
            logger.error("Configuration validation failed - aborting optimization")
            raise ValueError("Configuration validation failed")

        # Use current working directory since get_original_cwd() requires Hydra decorator
        data_dir = Path.cwd() / cfg.paths.data_dir
        optuna_optimize(data_dir, cfg)
        logger.info("[SUCCESS] Optuna optimization completed successfully!")

    def convert(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        gguf: bool = True,
        rkllm: bool = False,
        **overrides,
    ):
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
        cfg = self._load_config(config_name, config_dir)
        self._apply_overrides(cfg, overrides)

        configure_logging(logging.DEBUG)
        logger = logging.getLogger(__name__)

        conversions_performed = []

        if gguf:
            logger.info("[INFO] Converting to GGUF format...")
            # Use the merged model path as source
            model_path = cfg.paths.output_dir
            outfile = cfg.model.outfile

            convert_to_gguf(
                model_path=model_path,
                outfile=str(Path(model_path) / outfile),
                python_exe=cfg.paths.venv_python_path,
                outtype="f16",
                cfg=cfg,
            )
            conversions_performed.append("GGUF")

        if rkllm or cfg.get("model", {}).get("rkllm", {}).get("enabled", False):
            logger.info("[INFO] Converting to RKLLM format...")
            # Use the merged model path as source and create RKLLM output directory
            model_path = cfg.paths.output_dir
            rkllm_output_dir = Path(cfg.paths.output_dir) / "rkllm"

            # Get RKLLM parameters from config
            rkllm_config = cfg.get("model", {}).get("rkllm", {})
            target_platform = overrides.get(
                "target_platform", rkllm_config.get("target_platform", "rk3588")
            )
            quantization = overrides.get(
                "quantization", rkllm_config.get("quantization", "w8a8")
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
                f"[SUCCESS] Model conversion completed: {', '.join(conversions_performed)}"
            )
        else:
            logger.warning(
                "[WARNING] No conversions performed. Use --gguf=True or --rkllm=True"
            )

    def test(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        manual_setup: bool = True,
        **overrides,
    ):
        """
        Run model testing and evaluation.

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            manual_setup: Wait for manual model loading in LM Studio (default: True)
            **overrides: Configuration overrides

        Example:
            python main.py test
            python main.py test --manual_setup=False
        """
        cfg = self._load_config(config_name, config_dir)
        self._apply_overrides(cfg, overrides)

        configure_logging(logging.DEBUG)
        logger = logging.getLogger(__name__)

        # Prepare test dataset
        with Path(cfg.testing.test_dataset).open(encoding="utf-8") as file:
            test_dataset = json.load(file)
        dataset_to_json_for_test(test_dataset, cfg.testing.output_test_file)

        if manual_setup:
            input("[INFO] Load model into LM Studio and press Enter to continue...")

        logger.info("[INFO] Running model evaluation...")
        test_llm(
            cfg,
            path_test_dataset=cfg.testing.test_dataset,
            test_file=cfg.testing.output_test_file,
        )
        logger.info("[SUCCESS] Testing completed successfully!")

    def pipeline(
        self,
        config_name: str = "config",
        config_dir: str | None = None,
        use_optuna: bool = False,
        skip_test: bool = False,
        **overrides,
    ):
        """
        Run complete pipeline: train -> convert -> test.

        Args:
            config_name: Name of the config file (default: config)
            config_dir: Path to config directory (default: ./conf)
            use_optuna: Use Optuna optimization instead of standard training
            skip_test: Skip testing phase
            **overrides: Configuration overrides

        Example:
            python main.py pipeline
            python main.py pipeline --use_optuna=True
            python main.py pipeline --skip_test=True
        """
        cfg = self._load_config(config_name, config_dir)
        self._apply_overrides(cfg, overrides)

        configure_logging(logging.DEBUG)
        logger = logging.getLogger(__name__)

        logger.info("[INFO] Starting complete LLM LoRa pipeline...")

        # Validate configuration before starting pipeline
        logger.info("Validating configuration...")
        if not validate_complete_config(cfg):
            logger.error("Configuration validation failed - aborting pipeline")
            raise ValueError("Configuration validation failed")

        # Training phase
        if use_optuna:
            self.optimize(config_name, config_dir, **overrides)
        else:
            self.train(config_name, config_dir, **overrides)

        # Conversion phase
        self.convert(config_name, config_dir, **overrides)

        # Testing phase
        if not skip_test and cfg.get("testing", {}).get("manual_lmstudio_test", False):
            self.test(config_name, config_dir, **overrides)

        logger.info("[SUCCESS] Complete pipeline finished successfully!")


__all__ = [
    "LLMLoRaCLI",
    "configure_logging",
    "get_validation_summary",
    "main_train",
    "optuna_optimize",
    "validate_complete_config",
]
