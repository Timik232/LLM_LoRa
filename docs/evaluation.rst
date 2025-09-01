evaluation package
==================

The evaluation package provides comprehensive model evaluation capabilities for the LLM-LoRA framework. This package includes integration with DeepEval for advanced model assessment, specialized game evaluation metrics, and general model evaluation utilities.

Overview
--------

The evaluation framework offers:

* **DeepEval Integration**: Advanced evaluation metrics including faithfulness, answer relevancy, contextual precision, and bias detection
* **Game-specific Evaluation**: Specialized metrics for conversational AI and game-based interactions
* **Model Performance Assessment**: Comprehensive evaluation of fine-tuned models across multiple dimensions
* **Pytest Integration**: Structured testing framework with unit and integration test separation
* **Evaluation Metrics**: Support for BLEU, ROGUE, perplexity, and custom domain-specific metrics
* **Automated Testing**: Continuous evaluation pipeline for model quality assurance

Evaluation Workflow
-------------------

1. **Model Loading**: Load trained models for evaluation
2. **Dataset Preparation**: Prepare evaluation datasets with ground truth
3. **Metric Configuration**: Select appropriate evaluation metrics
4. **Evaluation Execution**: Run comprehensive model assessment
5. **Results Analysis**: Generate detailed evaluation reports
6. **Performance Comparison**: Compare models across different training configurations

Testing Structure
-----------------

The evaluation package follows a structured testing approach:

* **Unit Tests**: Individual function and component testing
* **Integration Tests**: End-to-end evaluation pipeline testing
* **Performance Tests**: Model performance benchmarking
* **Regression Tests**: Ensure model quality maintenance across updates

Submodules
----------

evaluation.deepeval\_integration module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration with DeepEval framework for advanced model evaluation metrics.

.. automodule:: evaluation.deepeval_integration
   :members:
   :show-inheritance:
   :undoc-members:

evaluation.game\_evaluation module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized evaluation metrics and methods for game-based conversational AI.

.. automodule:: evaluation.game_evaluation
   :members:
   :show-inheritance:
   :undoc-members:

evaluation.model\_evaluation module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

General model evaluation utilities and performance assessment tools.

.. automodule:: evaluation.model_evaluation
   :members:
   :show-inheritance:
   :undoc-members:

Evaluation Metrics
------------------

DeepEval Metrics
~~~~~~~~~~~~~~~~

* **Faithfulness**: Measures factual consistency with source material
* **Answer Relevancy**: Evaluates response relevance to input queries
* **Contextual Precision**: Assesses accuracy within given context
* **Contextual Recall**: Measures completeness of information retrieval
* **Bias Detection**: Identifies potential biases in model responses
* **Toxicity**: Detects harmful or inappropriate content

Game Evaluation Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

* **Dialogue Coherence**: Maintains conversation consistency
* **Character Consistency**: Preserves character traits and personality
* **Narrative Flow**: Evaluates story progression and continuity
* **Player Engagement**: Measures interaction quality and engagement

Standard Metrics
~~~~~~~~~~~~~~~~

* **BLEU Score**: Bilingual evaluation for text similarity
* **ROGUE Score**: Recall-oriented evaluation for summarization
* **Perplexity**: Language model uncertainty measurement
* **Token Accuracy**: Exact token matching evaluation

Usage Examples
--------------

Basic Model Evaluation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from evaluation.model_evaluation import evaluate_model
    from testing_model.models import CustomLocalModel, CustomMistralModel

    # Load and evaluate model
    results = evaluate_model(
        model_path="models/fine_tuned_model",
        test_dataset="data/test_set.json",
        metrics=["bleu", "rogue", "perplexity"]
    )

    # Use custom models for evaluation
    local_model = CustomLocalModel(model="custom-model", url="http://localhost:1234/v1/")
    mistral_model = CustomMistralModel(api_key="your-api-key")

DeepEval Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from evaluation.deepeval_integration import run_deepeval

    # Advanced evaluation with DeepEval
    eval_results = run_deepeval(
        model="models/chat_model",
        test_cases="data/evaluation_cases.json",
        metrics=["faithfulness", "bias", "toxicity"]
    )

Game-specific Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from evaluation.game_evaluation import evaluate_game_model

    # Game dialogue evaluation
    game_results = evaluate_game_model(
        model_path="models/game_character_model",
        scenarios="data/game_scenarios.json",
        character_profile="configs/character_config.yaml"
    )

Module contents
---------------

.. automodule:: evaluation
   :members:
   :show-inheritance:
   :undoc-members:
