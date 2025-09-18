import json
from pathlib import Path

import hydra
import ollama
from omegaconf import DictConfig

from evaluation.model_evaluation import dataset_to_json_for_test, test_llm
from training_model import configure_logging


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
    - Configures logging based on config log_level setting
    - Runs main testing process
    """
    configure_logging(cfg.logging.log_level)
    if cfg.testing.test:
        with Path(cfg.testing.test_dataset).open(encoding="utf-8") as file:
            test_dataset = json.load(file)
        dataset_to_json_for_test(test_dataset, cfg.testing.output_test_file)
        client = ollama.Client()
        test_llm(
            cfg,
            path_test_dataset=cfg.paths.test_data,
            test_file=cfg.testing.output_test_file,
            use_ollama=True,
            ollama_client=client,
        )


if __name__ == "__main__":
    test_main()
