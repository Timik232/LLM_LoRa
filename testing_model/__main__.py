import json
import logging

import hydra
from omegaconf import DictConfig

from training_model import configure_logging

from .test import dataset_to_json_for_test, test_llm


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def test_main(cfg: DictConfig) -> None:
    """
    Main entry point for testing workflow.

    This function performs the following steps:
    1. Configure logging
    2. Set up data directory path
    3. Run model testing via Ollama

    Args:
        cfg (DictConfig): Configuration dictionary from Hydra containing
                          paths, model settings, and other parameters.

    Returns:
        None

    Workflow:
    - Configures logging at DEBUG level
    - Runs main testing process
    """
    configure_logging(logging.DEBUG)
    if cfg.testing.test:
        with open(cfg.testing.test_dataset, "r", encoding="utf-8") as file:
            test_dataset = json.load(file)
        dataset_to_json_for_test(test_dataset, cfg.testing.output_test_file)
        input("Load model into lmstudio and press Enter to continue...")
        test_llm(
            cfg,
            path_test_dataset=cfg.testing.test_dataset,
            test_file=cfg.testing.output_test_file,
            use_ollama=True,
        )


if __name__ == "__main__":
    test_main()
