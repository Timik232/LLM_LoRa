import json
import logging
import os

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
    - Configures logging at DEBUG level
    - Runs main training process
    - Prompts user to load model
    - Optionally runs model testing via LM Studio
    """
    configure_logging(logging.DEBUG)
    # Use current working directory since get_original_cwd() requires Hydra decorator
    data_dir = os.path.join(os.getcwd(), cfg.paths.data_dir)
    if cfg.training.use_optuna_optimize:
        optuna_optimize(data_dir, cfg)
    else:
        main_train(data_dir, cfg)

    if cfg.testing.manual_lmstudio_test:
        with open(cfg.testing.test_dataset, "r", encoding="utf-8") as file:
            test_dataset = json.load(file)
        dataset_to_json_for_test(test_dataset, cfg.testing.output_test_file)
        input("Load model into lmstudio and press Enter to continue...")
        test_llm(
            cfg,
            path_test_dataset=cfg.testing.test_dataset,
            test_file=cfg.testing.output_test_file,
        )


def legacy_main():
    """
    Legacy main function that loads config with Hydra decorator.
    Kept for backward compatibility but no longer used as primary entry point.
    """
    config_dir = os.path.join(os.getcwd(), "conf")
    config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="config")
        main(cfg)


if __name__ == "__main__":
    legacy_main()
