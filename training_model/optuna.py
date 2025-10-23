"""
Hyperparameter optimization script using Optuna and Hydra for your LLM training pipeline.

Place this file (e.g., `hpo_optuna.py`) at your project root.
 Adjust `TRAIN_MODULE` to the path of your training
 module (e.g., 'train' if your main file is `train.py`).
"""

import importlib
import logging

import optuna
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, data_dir: str, cfg: DictConfig) -> float:
    """Run a single Optuna objective.

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
        # Clear any existing Hydra instance
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()

        overrides = []

        if cfg.optuna.learning_rate.enabled:
            lr = trial.suggest_float(
                "training.learning_rate",
                cfg.optuna.learning_rate.min,
                cfg.optuna.learning_rate.max,
                log=cfg.optuna.learning_rate.log_scale,
            )
            overrides.append(f"training.learning_rate={lr}")

        if cfg.optuna.num_train_epochs.enabled:
            epochs = trial.suggest_float(
                "training.num_train_epochs",
                cfg.optuna.num_train_epochs.min,
                cfg.optuna.num_train_epochs.max,
            )
            overrides.append(f"training.num_train_epochs={epochs}")

        if cfg.optuna.weight_decay.enabled:
            weight_decay = trial.suggest_float(
                "training.weight_decay",
                cfg.optuna.weight_decay.min,
                cfg.optuna.weight_decay.max,
            )
            overrides.append(f"training.weight_decay={weight_decay}")

        if cfg.optuna.warmup_steps.enabled:
            warmup_steps = trial.suggest_int(
                "training.warmup_steps",
                cfg.optuna.warmup_steps.min,
                cfg.optuna.warmup_steps.max,
            )
            overrides.append(f"training.warmup_steps={warmup_steps}")

        with initialize(
            version_base="1.1",
            config_path="../conf",
            job_name=f"optuna_hpo_trial_{trial.number}",
        ):
            cfg: DictConfig = compose(config_name="config", overrides=overrides)

            train_module = "training_model.one_file_train"
            module = importlib.import_module(train_module)
            metrics: dict = module.main_train(data_dir, cfg)

        loss = metrics.get("eval_loss")
        if loss is None:
            raise ValueError("train(cfg) did not return 'eval_loss' in metrics")
        return float(loss)
    except Exception as e:
        logger.exception(
            "Trial #%s encountered an error: %s",
            trial.number,
            e,
        )
        raise optuna.TrialPruned from e


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

    study.optimize(run_trial, n_trials=cfg.optuna.n_trials)

    logger.info("Best trial:")
    logger.info(f"  Loss: {study.best_value}")
    logger.info("  Params:")
    for key, val in study.best_params.items():
        logger.info(f"    {key}: {val}")

    with initialize(version_base="1.1", config_path="../conf", job_name="optuna_final"):
        best_overrides = [f"{k}={v}" for k, v in study.best_params.items()]
        best_cfg: DictConfig = compose(config_name="config", overrides=best_overrides)
        OmegaConf.save(best_cfg, "best_config.yaml")
    logger.info("Saved best configuration to best_config.yaml")
