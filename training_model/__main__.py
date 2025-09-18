import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from evaluation.model_evaluation import dataset_to_json_for_test, test_llm

from . import configure_logging, main_train, optuna_optimize


def main(cfg: DictConfig) -> None:
    """
    Main entry point for model training and testing workflow.

    This function performs the following steps:
    1. Configure logging
    2. Set up data directory path
    3. Run model training
    4. Optionally run model testing after manual model loading

    Args:
        cfg (DictConfig): Configuration dictionary from Hydra containing
                          paths, model settings, and other parameters.

    Returns:
        None

    Workflow:
    - Configures logging based on config log_level setting
    - Runs main training process
    - Prompts user to load model
    - Optionally runs model testing via LM Studio
    """
    configure_logging(cfg.logging.log_level)

    # Resolve data_dir path: if absolute use as-is, otherwise resolve relative to project root
    data_dir_config = Path(cfg.paths.data_dir)
    if data_dir_config.is_absolute():
        data_dir = data_dir_config
    else:
        # Find project root by going up from this file's location
        # training_model/__main__.py -> training_model/ -> project_root/
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / cfg.paths.data_dir

    # Ensure the data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    if cfg.training.use_optuna_optimize:
        optuna_optimize(data_dir, cfg)
    else:
        main_train(data_dir, cfg)

    if cfg.testing.manual_lmstudio_test:
        with Path(cfg.testing.test_dataset).open(encoding="utf-8") as file:
            test_dataset = json.load(file)
        dataset_to_json_for_test(test_dataset, cfg.testing.output_test_file)
        input("Load model into lmstudio and press Enter to continue...")
        test_llm(
            cfg,
            path_test_dataset=cfg.testing.test_dataset,
            test_file=cfg.testing.output_test_file,
        )


def legacy_main() -> None:
    """
    Legacy main function that loads config with Hydra decorator.
    Kept for backward compatibility but no longer used as primary entry point.
    """
    # Resolve repo root relative to this file to avoid CWD-dependent failures
    config_dir = (Path(__file__).resolve().parents[1] / "conf").resolve()

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="config")
        main(cfg)


if __name__ == "__main__":
    legacy_main()
