Advanced Training Methods
=========================

The LLM-LoRA framework supports multiple advanced training approaches beyond traditional supervised fine-tuning. This document covers Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) methods for preference-based model alignment.

Overview
--------

Modern language model training has evolved beyond simple supervised learning to incorporate human preference data and alignment objectives. The framework provides three main training approaches:

* **Supervised Fine-tuning (SFT)**: Traditional instruction-response training
* **Direct Preference Optimization (DPO)**: Preference-based training without reinforcement learning
* **Group Relative Policy Optimization (GRPO)**: Advanced group-based preference learning

These methods enable training models that better align with human values and preferences while maintaining computational efficiency through LoRA adaptation.

Supervised Fine-tuning (SFT)
-----------------------------

Traditional Training Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SFT remains the foundation for most language model fine-tuning tasks. It uses instruction-response pairs to teach the model specific behaviors and capabilities.

**Key Characteristics:**

* Direct optimization on instruction-response pairs
* Stable and well-understood training dynamics
* Efficient with limited data
* Good baseline for further preference training

**Use Cases:**

* Task-specific adaptation (summarization, question-answering)
* Domain adaptation (medical, legal, technical)
* Instruction following capabilities
* Base model for preference methods

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    training:
      training_method: "sft"
      per_device_train_batch_size: 2
      gradient_accumulation_steps: 8
      learning_rate: 2e-5
      num_train_epochs: 3
      max_seq_length: 2048

    data:
      train_dataset: "data/sft_dataset.json"
      dataset_format: "instruction_response"

Data Format
~~~~~~~~~~~

SFT expects instruction-response pairs in JSON format:

.. code-block:: json

    [
      {
        "instruction": "Explain photosynthesis in simple terms.",
        "response": "Photosynthesis is how plants make food using sunlight..."
      },
      {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris."
      }
    ]

Direct Preference Optimization (DPO)
-------------------------------------

Preference Learning Without RL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DPO trains models directly on preference data without requiring a separate reward model or reinforcement learning. It optimizes the model to prefer chosen responses over rejected ones.

**Key Advantages:**

* No reward model required
* Stable training without RL complexity
* Direct optimization on preferences
* Computationally efficient
* Better alignment with human values

**Mathematical Foundation:**

DPO maximizes the likelihood of preferred responses while minimizing the likelihood of rejected responses using a reference model for regularization.

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    training:
      training_method: "dpo"
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 16
      learning_rate: 1e-5
      num_train_epochs: 1
      beta: 0.1  # Temperature parameter
      reference_model_path: "models/sft_base"

    data:
      preference_dataset: "data/preference_pairs.json"
      dataset_format: "preference_pairs"

Data Format
~~~~~~~~~~~

DPO requires preference pairs with chosen and rejected responses:

.. code-block:: json

    [
      {
        "prompt": "How should I invest my money?",
        "chosen": "Consider diversifying across stocks, bonds, and index funds based on your risk tolerance and timeline.",
        "rejected": "Put all your money in cryptocurrency - it's guaranteed to make you rich!"
      },
      {
        "prompt": "What's the best way to lose weight?",
        "chosen": "Focus on a balanced diet with regular exercise, aiming for 1-2 pounds per week weight loss.",
        "rejected": "Try this extreme crash diet that will help you lose 20 pounds in one week!"
      }
    ]

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

The DPO training process:

1. **Reference Model**: Use SFT model as reference for regularization
2. **Preference Loss**: Compute DPO loss using chosen/rejected pairs
3. **Beta Parameter**: Controls strength of regularization (typical: 0.1-0.5)
4. **Batch Processing**: Handle preference pairs efficiently in batches

.. code-block:: python

    # Simplified DPO training loop
    def dpo_train_step(model, ref_model, batch, beta=0.1):
        # Forward pass on chosen and rejected responses
        chosen_logits = model(batch['chosen_input'])
        rejected_logits = model(batch['rejected_input'])

        # Reference model logits (frozen)
        with torch.no_grad():
            ref_chosen = ref_model(batch['chosen_input'])
            ref_rejected = ref_model(batch['rejected_input'])

        # Compute DPO loss
        loss = compute_dpo_loss(chosen_logits, rejected_logits,
                               ref_chosen, ref_rejected, beta)
        return loss

Group Relative Policy Optimization (GRPO)
------------------------------------------

Advanced Preference Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRPO extends preference optimization by considering groups of responses and their relative preferences. This method provides more nuanced preference learning compared to pairwise methods.

**Key Features:**

* Group-based preference learning
* Multiple response ranking
* Improved alignment accuracy
* Better handling of preference ambiguity
* Suitable for complex preference structures

**Advantages over DPO:**

* Handles multiple preference levels
* More robust to noisy preferences
* Better generalization
* Improved alignment quality

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    training:
      training_method: "grpo"
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 32
      learning_rate: 5e-6
      num_train_epochs: 1
      beta: 0.2
      group_size: 4  # Number of responses per group
      reference_model_path: "models/sft_base"

    data:
      preference_dataset: "data/ranked_responses.json"
      dataset_format: "ranked_groups"

Data Format
~~~~~~~~~~~

GRPO uses ranked groups of responses:

.. code-block:: json

    [
      {
        "prompt": "Explain quantum computing",
        "responses": [
          {
            "text": "Quantum computing uses quantum mechanics...",
            "rank": 1
          },
          {
            "text": "Quantum computers are just fast computers...",
            "rank": 3
          },
          {
            "text": "Quantum computing leverages superposition and entanglement...",
            "rank": 2
          }
        ]
      }
    ]

Training Pipeline Comparison
----------------------------

Method Selection Guide
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Training Method Comparison
    :widths: 20 25 25 30
    :header-rows: 1

    * - Method
      - Data Requirements
      - Training Complexity
      - Best Use Cases
    * - SFT
      - Instruction-response pairs
      - Low
      - Task adaptation, base training
    * - DPO
      - Preference pairs
      - Medium
      - Alignment, safety, quality improvement
    * - GRPO
      - Ranked response groups
      - High
      - Complex preferences, advanced alignment

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Resource Requirements
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Method
      - Memory Usage
      - Training Time
      - Data Efficiency
      - Convergence Speed
    * - SFT
      - Low
      - Fast
      - High
      - Fast
    * - DPO
      - Medium
      - Medium
      - Medium
      - Medium
    * - GRPO
      - High
      - Slow
      - Low
      - Slow

Sequential Training Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal results, use a sequential training approach:

1. **SFT Base**: Train foundation model on instruction data
2. **DPO Alignment**: Apply preference optimization for basic alignment
3. **GRPO Refinement**: Fine-tune with complex preference structures (optional)

.. code-block:: bash

    # Sequential training pipeline
    # Step 1: SFT training
    python main.py train_model --config-name=sft_config

    # Step 2: DPO on SFT model
    python main.py train_model --config-name=dpo_config --model.base_model=models/sft_model

    # Step 3: GRPO refinement (optional)
    python main.py train_model --config-name=grpo_config --model.base_model=models/dpo_model

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

* **Quality over Quantity**: High-quality preferences are more valuable than large datasets
* **Diverse Preferences**: Include varied preference types and contexts
* **Balanced Data**: Ensure balanced representation of different preference categories
* **Validation Sets**: Maintain separate validation sets for each training stage

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~~

**SFT Parameters:**
- Learning rate: 1e-5 to 5e-5
- Batch size: Depends on available memory
- Epochs: 1-5 typically sufficient

**DPO Parameters:**
- Beta: 0.1-0.5 (higher values = stronger regularization)
- Learning rate: Lower than SFT (1e-6 to 1e-5)
- Reference model: Use best SFT checkpoint

**GRPO Parameters:**
- Group size: 3-8 responses per group
- Beta: 0.2-0.8 (typically higher than DPO)
- Learning rate: Very low (1e-7 to 5e-6)

Monitoring Training
~~~~~~~~~~~~~~~~~~~

Key metrics to track:

* **Loss Convergence**: Monitor training and validation loss
* **Preference Accuracy**: Track preference prediction accuracy
* **Model Quality**: Regular evaluation on held-out test sets
* **Alignment Metrics**: Use evaluation frameworks to assess alignment

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**DPO Training Issues:**
- Preference data imbalance
- Reference model quality
- Beta parameter tuning
- Convergence problems

**GRPO Training Issues:**
- Insufficient group diversity
- Ranking consistency
- Memory constraints
- Training instability

**Solutions:**
- Data preprocessing and validation
- Careful hyperparameter selection
- Gradient accumulation for memory management
- Regular checkpointing and validation

Advanced Configurations
-----------------------

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    training:
      training_method: "dpo"
      ddp_enabled: true
      num_gpus: 4
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 4

Custom Loss Functions
~~~~~~~~~~~~~~~~~~~~~

The framework supports custom loss implementations for specialized preference learning:

.. code-block:: python

    # Custom preference loss function
    class CustomPreferenceLoss(nn.Module):
        def __init__(self, beta=0.1):
            super().__init__()
            self.beta = beta

        def forward(self, chosen_logits, rejected_logits, ref_chosen, ref_rejected):
            # Implement custom preference loss
            pass

Integration with Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preference-trained models require specialized evaluation:

.. code-block:: bash

    # Evaluate preference-trained model
    python main.py evaluate_model \\
        --model_path=models/dpo_model \\
        --evaluation_type=preference \\
        --metrics=["preference_accuracy", "alignment_score"]
