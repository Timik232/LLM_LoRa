"""
Hyperparameter optimization script using Optuna and Hydra for your LLM training pipeline.

Place this file (e.g., `hpo_optuna.py`) at your project root. Adjust `TRAIN_MODULE` to the path of your training module (e.g., 'train' if your main file is `train.py`).
"""

import importlib
import logging

import optuna
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def objective(trial: optuna.Trial, data_dir: str, cfg: DictConfig) -> float:
    """Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object used to suggest hyperparameters.
        data_dir: Path to the training data directory.
        cfg: Initial configuration dictionary.

    Returns:
        float: Validation loss to minimize.

    Raises:
        ValueError: If 'eval_loss' is not present in the training metrics.
    """
    try:
        GlobalHydra.instance().clear()
        lr = trial.suggest_float("training.learning_rate", 1e-6, 5e-5, log=True)
        epochs = trial.suggest_float("training.num_train_epochs", 0.5, 2)
        weight_decay = trial.suggest_float("training.weight_decay", 0.0, 0.3)
        warmup_steps = trial.suggest_int("training.warmup_steps", 0, 500)

        with initialize(
            version_base="1.1",
            config_path="../conf",
            job_name=f"optuna_hpo_trial_{trial.number}",
        ):
            overrides = [
                f"training.learning_rate={lr}",
                f"training.num_train_epochs={epochs}",
                f"training.weight_decay={weight_decay}",
                f"training.warmup_steps={warmup_steps}",
            ]
            cfg: DictConfig = compose(config_name="config", overrides=overrides)

        TRAIN_MODULE = "training_model.one_file_train"
        module = importlib.import_module(TRAIN_MODULE)
        # Предполагается, что функция train возвращает словарь {'eval_loss': float}
        metrics: dict = module.main_train(data_dir, cfg)
        loss = metrics.get("eval_loss")
        if loss is None:
            raise ValueError("train(cfg) did not return 'eval_loss' in metrics")
        return float(loss)
    except Exception as e:
        logging.error(f"Trial #{trial.number} encountered an error: {e}")
        raise optuna.TrialPruned() from e


def optuna_optimize(data_dir: str, cfg: DictConfig) -> None:
    """Run Optuna hyperparameter optimization.

    Args:
        data_dir: Path to the training data directory.
        cfg: Initial configuration dictionary.
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3),
    )

    def run_trial(trial: optuna.Trial) -> float:
        """Wrapper to pass data_dir and cfg into the objective."""
        return objective(trial, data_dir, cfg)

    study.optimize(run_trial, n_trials=cfg.training.optuna_n_trials)

    logging.info("Best trial:")
    logging.info(f"  Loss: {study.best_value}")
    logging.info("  Params:")
    for key, val in study.best_params.items():
        logging.info(f"    {key}: {val}")

    with initialize(version_base="1.1", config_path="../conf", job_name="optuna_final"):
        best_overrides = [f"{k}={v}" for k, v in study.best_params.items()]
        best_cfg: DictConfig = compose(config_name="config", overrides=best_overrides)
        from omegaconf import OmegaConf

        OmegaConf.save(best_cfg, "best_config.yaml")
    logging.info("Saved best configuration to best_config.yaml")
