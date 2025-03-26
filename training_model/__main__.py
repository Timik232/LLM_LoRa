import logging
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from .logging_config import configure_logging
from .one_file_train import main_train


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    configure_logging(logging.DEBUG)
    data_dir = os.path.join(get_original_cwd(), cfg.paths.data_dir)
    main_train(data_dir, cfg)
    # next = input("Загрузите модель и нажмите Enter. Для пропуска введите 0\n")
    # if next == "0":
    #    return
    # test_via_lmstudio(
    #     cfg,
    #     test_dataset=os.path.join(data_dir, "dataset_ru.json"),
    #     test_file="train.json",
    # )
    # test_via_lmstudio(
    #     cfg,
    #     test_dataset=os.path.join(data_dir, "test_ru.json"),
    #     test_file="../test.json",
    # )


if __name__ == "__main__":
    main()
