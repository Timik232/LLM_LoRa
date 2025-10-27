import json
import logging
from pathlib import Path

import hydra
import ollama
from omegaconf import DictConfig

from evaluation.game_evaluation import test_actions
from evaluation.model_evaluation import dataset_to_json_for_test, test_llm
from training_model import configure_logging

try:
    from evaluation.deepeval_integration import (
        test_game_context_appropriateness,
        test_game_context_appropriateness_async,
        test_russian_language_quality,
        test_russian_language_quality_async,
        test_unintended_answer_mention,
        test_unintended_answer_mention_async,
    )
except ImportError:
    test_game_context_appropriateness = None
    test_russian_language_quality = None
    test_unintended_answer_mention = None
    test_game_context_appropriateness_async = None
    test_russian_language_quality_async = None
    test_unintended_answer_mention_async = None


def _create_deepeval_test_functions(cfg: DictConfig) -> list:
    """Create list of DeepEval test functions based on config.

    Args:
        cfg (DictConfig): Hydra configuration

    Returns:
        list: List of test functions to execute
    """
    logger = logging.getLogger(__name__)
    test_functions = []

    deepeval_cfg = cfg.get("testing", {}).get("deepeval_testing", {})
    if not deepeval_cfg.get("enabled", False):
        logger.info("DeepEval testing disabled in config")
        return test_functions

    if None in [
        test_game_context_appropriateness,
        test_russian_language_quality,
        test_unintended_answer_mention,
    ]:
        logger.warning("DeepEval integration not available - skipping DeepEval metrics")
        return test_functions

    requested_metrics = deepeval_cfg.get("metrics", [])
    eval_model_type = (
        cfg.get("deepeval", {}).get("evaluation_model", {}).get("type", "mistral")
    )
    logger.info(
        f"DeepEval testing enabled. Model type: {eval_model_type}, "
        f"Metrics: {requested_metrics}"
    )

    async def wrapper_unintended_answer_mention(
        model_answer: str, correct_answer: str, **kwargs: dict
    ) -> tuple[bool, float, str]:
        """Async wrapper for test_unintended_answer_mention_async
        to match test_actions signature."""
        logger = logging.getLogger(__name__)
        try:
            from evaluation.deepeval_integration import _get_evaluation_model

            logger.debug("Initializing evaluation model for unintended_answer_mention metric")
            _get_evaluation_model(cfg)
            user_input = kwargs.get("user_input", "")
            if not user_input:
                logger.warning(
                    "WARNING: user_input is empty for unintended_answer_mention metric. "
                    "DeepEval will not be able to evaluate if the "
                    "model reveals answers to questions."
                )
            return await test_unintended_answer_mention_async(
                cfg, user_input, model_answer, correct_answer
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize evaluation model for unintended_answer_mention: {e}"
            )
            logger.error(
                "Check your DeepEval config and API credentials (MISTRAL_API, CUSTOM_API_KEY)"
            )
            raise

    async def wrapper_russian_language_quality(
        model_answer: str, correct_answer: str, **kwargs: dict
    ) -> tuple[bool, float, str]:
        """Async wrapper for test_russian_language_quality_async
        to match test_actions signature."""
        logger = logging.getLogger(__name__)
        try:
            from evaluation.deepeval_integration import _get_evaluation_model

            logger.debug("Initializing evaluation model for russian_language_quality metric")
            _get_evaluation_model(cfg)
            return await test_russian_language_quality_async(cfg, model_answer)
        except Exception as e:
            logger.error(
                f"Failed to initialize evaluation model for russian_language_quality: {e}"
            )
            logger.error(
                "Check your DeepEval config and API credentials (MISTRAL_API, CUSTOM_API_KEY)"
            )
            raise

    async def wrapper_game_context_appropriateness(
        model_answer: str, correct_answer: str, **kwargs: dict
    ) -> tuple[bool, float, str]:
        """Async wrapper for test_game_context_appropriateness_async
        to match test_actions signature."""
        try:
            from evaluation.deepeval_integration import _get_evaluation_model

            logger.debug(
                "Initializing evaluation model for game_context_appropriateness metric"
            )
            _get_evaluation_model(cfg)
            available_actions = kwargs.get("available_actions", [])
            user_input = kwargs.get("user_input")
            if not user_input:
                logger.warning(
                    "WARNING: user_input is empty for game_context_appropriateness metric. "
                    "DeepEval will not have full context for evaluation."
                )
            return await test_game_context_appropriateness_async(
                cfg, model_answer, available_actions, user_input
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize evaluation model for game_context_appropriateness: {e}"
            )
            logger.error(
                "Check your DeepEval config and API credentials (MISTRAL_API, CUSTOM_API_KEY)"
            )
            raise

    metric_mapping = {
        "unintended_answer_mention": wrapper_unintended_answer_mention,
        "russian_language_quality": wrapper_russian_language_quality,
        "game_context_appropriateness": wrapper_game_context_appropriateness,
    }

    for metric_name in requested_metrics:
        if metric_name in metric_mapping:
            metric_func = metric_mapping[metric_name]
            test_functions.append(metric_func)
            logger.info(f"✓ Added DeepEval metric: {metric_name}")
        else:
            logger.warning(f"✗ Unknown metric: {metric_name}")

    return test_functions


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def test_main(cfg: DictConfig) -> None:
    """
    Main entry point for testing workflow.

    This function performs the following steps:
    1. Configure logging
    2. Set up data directory path
    3. Build list of test functions (action validation + optional DeepEval metrics)
    4. Run model testing via Ollama
    5. Display final test summary

    Args:
        cfg (DictConfig): Configuration dictionary from Hydra containing
                          paths, model settings, and other parameters.

    Returns:
        None

    Workflow:
    - Configures logging based on config log_level setting
    - Loads DeepEval metrics if enabled in config
    - Runs testing process with configured test functions
    - Displays comprehensive test execution summary
    """
    configure_logging(cfg.logging.log_level)
    logger = logging.getLogger(__name__)

    if cfg.testing.test:
        with Path(cfg.testing.test_dataset).open(encoding="utf-8") as file:
            test_dataset = json.load(file)
        dataset_to_json_for_test(test_dataset, cfg.testing.output_test_file)
        client = ollama.Client()

        test_functions = [test_actions]
        deepeval_functions = _create_deepeval_test_functions(cfg)
        test_functions.extend(deepeval_functions)

        if len(test_functions) == 1:
            logger.info("Running action validation only")
        else:
            logger.info(
                f"Running {len(test_functions)} test "
                f"functions (action validation + {len(deepeval_functions)} DeepEval metrics)"
            )

        parallel_cfg = cfg.get("testing", {}).get("parallel_execution", {})
        if parallel_cfg.get("enabled", False):
            num_workers = parallel_cfg.get("num_workers", 4)
            logger.info(f"Parallel execution enabled with {num_workers} workers")
        else:
            logger.info("Sequential test execution enabled")

        test_data_path = Path(cfg.paths.data_dir) / cfg.testing.test_dataset.split("/")[-1]

        summaries = test_llm(
            cfg,
            path_test_dataset=str(test_data_path),
            test_file=cfg.testing.output_test_file,
            test_func=test_functions,
            use_ollama=True,
            ollama_client=client,
        )

        if summaries:
            logger.info("\n" + "=" * 60)
            logger.info("FINAL TEST EXECUTION SUMMARY")
            logger.info("=" * 60)
            for i, summary in enumerate(summaries):
                logger.info(
                    f"\nTest Function {i + 1}:"
                    f" {summary.test_functions[0] if summary.test_functions else 'Unknown'}"
                )
                logger.info(f"  Total Tests: {summary.total_tests}")
                logger.info(
                    f"  Passed: {summary.passed_tests} ({summary.success_rate * 100:.1f}%)"
                )
                logger.info(
                    f"  Failed: {summary.failed_tests} "
                    f"({(1 - summary.success_rate) * 100:.1f}%)"
                )
                if summary.metrics_per_function:
                    logger.info("  Detailed Metrics:")
                    for func_name, metrics in summary.metrics_per_function.items():
                        passed = metrics.get("passed", 0)
                        total = metrics.get("total", 0)
                        success_pct = (passed / total * 100) if total > 0 else 0
                        logger.info(
                            f"    - {func_name}: {passed}/{total} ({success_pct:.1f}%)"
                        )
            logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    test_main()
