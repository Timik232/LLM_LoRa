import json
import logging
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from testing_model import test_via_llamacpp
from testing_model.test import dataset_to_json_for_test

from . import configure_logging, main_train


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
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
    data_dir = os.path.join(get_original_cwd(), cfg.paths.data_dir)
    main_train(data_dir, cfg)
    if cfg.testing.test:
        with open(cfg.testing.test_file, "r", encoding="utf-8") as file:
            test_dataset = json.load(file)
        dataset_to_json_for_test(test_dataset, cfg.testing.output_test_file)
        path_to_llm = os.path.join(
            cfg.paths.final_weights_path,
            cfg.model.version,
            cfg.model.outfile.replace(".gguf", f"{cfg.model.quant_postfix}.gguf"),
        )
        test_via_llamacpp(
            model_path=path_to_llm,
            test_dataset=cfg.testing.test_file,
            test_file=cfg.testing.output_test_file,
            temperature=0.7,
        )


if __name__ == "__main__":
    main()
