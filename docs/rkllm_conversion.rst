RKLLM Conversion and NPU Deployment
====================================

The LLM-LoRA framework provides comprehensive support for converting trained models to RKLLM format for deployment on Rockchip NPU (Neural Processing Unit) devices. This enables efficient inference on edge devices with specialized AI acceleration hardware.

Overview
--------

RKLLM (Rockchip Large Language Model) is a specialized format optimized for Rockchip's NPU hardware. The conversion process transforms PyTorch models into a format that can leverage the high-performance, low-power inference capabilities of Rockchip devices.

**Key Features:**

* **NPU Acceleration**: Leverage dedicated AI hardware for faster inference
* **Edge Deployment**: Deploy models on resource-constrained devices
* **Power Efficiency**: Optimized for low power consumption
* **Platform Support**: Support for RK3588, RK3576, and other Rockchip SoCs
* **Quantization**: Automatic optimization for INT8/INT16 inference
* **Model Optimization**: Graph optimization and operator fusion

Supported Platforms
--------------------

Rockchip SoC Support
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Platform Compatibility
    :widths: 20 30 25 25
    :header-rows: 1

    * - Platform
      - NPU Performance
      - Memory Support
      - Typical Use Cases
    * - RK3588
      - 6 TOPS @ INT8
      - Up to 32GB
      - High-performance edge AI
    * - RK3576
      - 6 TOPS @ INT8
      - Up to 16GB
      - Compact AI devices
    * - RK3566
      - 1 TOP @ INT8
      - Up to 8GB
      - IoT and embedded AI
    * - RK3562
      - 1 TOP @ INT8
      - Up to 4GB
      - Cost-optimized AI solutions

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: RKLLM Performance Metrics
    :widths: 25 25 25 25
    :header-rows: 1

    * - Model Size
      - RK3588 (tokens/sec)
      - RK3576 (tokens/sec)
      - Power Usage (W)
    * - 1B parameters
      - ~50-80
      - ~40-60
      - 5-8W
    * - 3B parameters
      - ~25-40
      - ~20-30
      - 8-12W
    * - 7B parameters
      - ~10-20
      - ~8-15
      - 12-18W

RKLLM Conversion Workflow
-------------------------

Prerequisites
~~~~~~~~~~~~~

Before starting the conversion process, ensure you have:

* **RKLLM Toolkit**: Official Rockchip RKLLM conversion tools
* **Target Hardware**: Access to target Rockchip device for testing
* **Python Environment**: Compatible Python environment with required dependencies
* **Trained Model**: PyTorch model trained with the LLM-LoRA framework

Installation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Install RKLLM toolkit (requires Rockchip account)
    # Download from Rockchip developer portal
    pip install rkllm-toolkit

    # Additional dependencies
    pip install onnx onnxruntime
    pip install transformers torch

Conversion Process
~~~~~~~~~~~~~~~~~~

The conversion process involves several steps:

1. **Model Preparation**: Ensure model compatibility
2. **ONNX Export**: Convert PyTorch to ONNX intermediate format
3. **RKLLM Conversion**: Transform ONNX to RKLLM format
4. **Optimization**: Apply platform-specific optimizations
5. **Validation**: Test converted model accuracy

Configuration Setup
~~~~~~~~~~~~~~~~~~~~

Configure RKLLM conversion in your Hydra config:

.. code-block:: yaml

    conversion:
      convert_to_rkllm: true
      rkllm_config:
        target_platform: "rk3588"
        optimization_level: 2
        quantization: "int8"
        batch_size: 1
        sequence_length: 2048

    rkllm:
      toolkit_path: "path/to/rkllm-toolkit"
      output_format: "rkllm"
      enable_profiling: true
      optimization:
        graph_optimization: true
        operator_fusion: true
        memory_optimization: true

Using Fire CLI
~~~~~~~~~~~~~~

Convert models using the Fire CLI interface:

.. code-block:: bash

    # Basic RKLLM conversion
    python main.py convert_to_rkllm --model_path=models/trained_model

    # Conversion with specific platform
    python main.py convert_to_rkllm \\
        --model_path=models/trained_model \\
        --platform=rk3588 \\
        --quantization=int8

    # Advanced conversion with optimization
    python main.py convert_to_rkllm \\
        --model_path=models/trained_model \\
        --platform=rk3588 \\
        --optimization_level=2 \\
        --batch_size=1 \\
        --sequence_length=2048

Detailed Conversion Steps
-------------------------

Step 1: Model Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare the trained PyTorch model for conversion:

.. code-block:: python

    # Model preparation script
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def prepare_model_for_rkllm(model_path):
        # Load trained model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set model to evaluation mode
        model.eval()

        # Apply any necessary model modifications
        model = optimize_for_inference(model)

        return model, tokenizer

Step 2: ONNX Export
~~~~~~~~~~~~~~~~~~~

Convert the PyTorch model to ONNX format:

.. code-block:: python

    def export_to_onnx(model, tokenizer, output_path, sequence_length=2048):
        # Create dummy input
        dummy_input = torch.randint(0, tokenizer.vocab_size, (1, sequence_length))

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )

Step 3: RKLLM Conversion
~~~~~~~~~~~~~~~~~~~~~~~~

Use the RKLLM toolkit to convert the ONNX model:

.. code-block:: python

    from rkllm import RKLLM

    def convert_to_rkllm(onnx_path, output_path, config):
        # Initialize RKLLM converter
        rkllm = RKLLM(verbose=True)

        # Load and convert model
        ret = rkllm.load_onnx(
            model=onnx_path,
            inputs=['input_ids'],
            input_size_list=[[1, config['sequence_length']]],
            input_type_list=['int32']
        )

        if ret != 0:
            raise RuntimeError("Failed to load ONNX model")

        # Build RKLLM model
        ret = rkllm.build(
            do_quantization=config.get('quantization', True),
            optimization_level=config.get('optimization_level', 1),
            target_platform=config.get('target_platform', 'rk3588')
        )

        if ret != 0:
            raise RuntimeError("Failed to build RKLLM model")

        # Export model
        ret = rkllm.export_rkllm(output_path)

        if ret != 0:
            raise RuntimeError("Failed to export RKLLM model")

Optimization Settings
---------------------

Quantization Options
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Quantization Settings
    :widths: 25 35 40
    :header-rows: 1

    * - Precision
      - Description
      - Performance Impact
    * - INT8
      - 8-bit integer quantization
      - 4x speed improvement, minimal accuracy loss
    * - INT16
      - 16-bit integer quantization
      - 2x speed improvement, better accuracy
    * - FLOAT16
      - 16-bit floating point
      - 1.5x speed improvement, highest accuracy

Graph Optimization Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Optimization Levels
    :widths: 15 35 50
    :header-rows: 1

    * - Level
      - Optimizations Applied
      - Description
    * - 0
      - None
      - No optimization, fastest conversion
    * - 1
      - Basic optimizations
      - Standard operator fusion and constant folding
    * - 2
      - Advanced optimizations
      - Graph restructuring and memory optimization
    * - 3
      - Aggressive optimizations
      - Maximum performance, longer conversion time

Memory Optimization
~~~~~~~~~~~~~~~~~~~

Configure memory usage for optimal performance:

.. code-block:: yaml

    rkllm:
      memory_config:
        weight_memory_size: "1024MB"
        internal_memory_size: "256MB"
        io_memory_size: "128MB"
        optimization:
          enable_weight_sharing: true
          enable_activation_reuse: true
          memory_pool_optimization: true

Deployment and Runtime
----------------------

RKLLM Runtime Setup
~~~~~~~~~~~~~~~~~~~

Deploy the converted model on target hardware:

.. code-block:: python

    # RKLLM runtime inference
    from rkllm.api import RKLLM_Handle_t, rkllm_init, rkllm_inference, rkllm_destroy

    class RKLLMInference:
        def __init__(self, model_path):
            self.handle = RKLLM_Handle_t()
            ret = rkllm_init(self.handle, model_path, target_platform="rk3588")
            if ret != 0:
                raise RuntimeError("Failed to initialize RKLLM model")

        def generate(self, input_text, max_length=512):
            # Tokenize input
            input_ids = self.tokenize(input_text)

            # Run inference
            output_ids = rkllm_inference(
                self.handle,
                input_ids,
                max_length=max_length
            )

            # Decode output
            return self.detokenize(output_ids)

        def __del__(self):
            rkllm_destroy(self.handle)

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Monitor model performance during deployment:

.. code-block:: python

    # Performance monitoring
    import time
    import psutil

    def benchmark_rkllm_model(model, test_inputs, num_runs=10):
        latencies = []
        memory_usage = []

        for _ in range(num_runs):
            start_mem = psutil.Process().memory_info().rss

            start_time = time.time()
            output = model.generate(test_inputs)
            end_time = time.time()

            end_mem = psutil.Process().memory_info().rss

            latencies.append(end_time - start_time)
            memory_usage.append(end_mem - start_mem)

        return {
            'avg_latency': sum(latencies) / len(latencies),
            'avg_memory': sum(memory_usage) / len(memory_usage),
            'throughput': len(test_inputs) / sum(latencies)
        }

Docker Deployment
~~~~~~~~~~~~~~~~~

Deploy RKLLM models using Docker containers:

.. code-block:: dockerfile

    FROM rockchip/rkllm-runtime:latest

    # Copy model and runtime
    COPY models/model.rkllm /app/model.rkllm
    COPY inference_server.py /app/

    # Install dependencies
    RUN pip install fastapi uvicorn

    # Set environment variables
    ENV RKLLM_MODEL_PATH=/app/model.rkllm
    ENV TARGET_PLATFORM=rk3588

    # Expose port
    EXPOSE 8000

    # Start inference server
    CMD ["python", "/app/inference_server.py"]

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Conversion Failures:**
- Model architecture compatibility
- ONNX export issues
- Quantization problems
- Memory limitations

**Performance Issues:**
- Suboptimal quantization settings
- Insufficient optimization
- Memory constraints
- Platform-specific limitations

**Runtime Errors:**
- Model loading failures
- Inference crashes
- Memory allocation issues
- Hardware compatibility problems

Debugging Steps
~~~~~~~~~~~~~~~

1. **Verify Model Compatibility**:

   .. code-block:: bash

       python main.py validate_model --model_path=models/trained_model --target=rkllm

2. **Check ONNX Export**:

   .. code-block:: python

       import onnx
       model = onnx.load("model.onnx")
       onnx.checker.check_model(model)

3. **Monitor Conversion Process**:

   .. code-block:: bash

       python main.py convert_to_rkllm --model_path=models/trained_model --verbose=true

4. **Test on Target Hardware**:

   .. code-block:: bash

       # Deploy to target device and test
       scp model.rkllm user@target_device:/path/to/model/
       ssh user@target_device "cd /path/to/model && python test_inference.py"

Performance Optimization Tips
-----------------------------

Model Architecture
~~~~~~~~~~~~~~~~~~

* **Layer Optimization**: Reduce unnecessary layers and operations
* **Attention Optimization**: Optimize attention mechanisms for NPU
* **Activation Functions**: Use NPU-friendly activation functions

Quantization Strategy
~~~~~~~~~~~~~~~~~~~~

* **Calibration Data**: Use representative calibration dataset
* **Mixed Precision**: Apply different quantization levels to different layers
* **Accuracy Validation**: Validate accuracy after quantization

Deployment Optimization
~~~~~~~~~~~~~~~~~~~~~~~

* **Batch Size**: Optimize for target batch size requirements
* **Memory Layout**: Configure optimal memory layout for NPU
* **Concurrent Inference**: Enable multiple inference streams when possible

Integration Examples
--------------------

Web Service Integration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fastapi import FastAPI
    from rkllm_inference import RKLLMInference

    app = FastAPI()
    model = RKLLMInference("model.rkllm")

    @app.post("/generate")
    async def generate_text(prompt: str, max_length: int = 512):
        result = model.generate(prompt, max_length)
        return {"generated_text": result}

IoT Integration
~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    from rkllm_inference import RKLLMInference

    class IoTLanguageModel:
        def __init__(self, model_path):
            self.model = RKLLMInference(model_path)

        async def process_sensor_data(self, sensor_input):
            # Convert sensor data to text prompt
            prompt = self.format_sensor_prompt(sensor_input)

            # Generate response
            response = self.model.generate(prompt)

            # Process and return action
            return self.parse_action(response)

Edge AI Application
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class EdgeAIAssistant:
        def __init__(self, rkllm_model_path):
            self.model = RKLLMInference(rkllm_model_path)
            self.conversation_history = []

        def chat(self, user_input):
            # Build context with conversation history
            context = self.build_context(user_input)

            # Generate response
            response = self.model.generate(context)

            # Update conversation history
            self.update_history(user_input, response)

            return response

        def build_context(self, current_input):
            # Include recent conversation history
            context = ""
            for turn in self.conversation_history[-5:]:  # Last 5 turns
                context += f"User: {turn['user']}\\nAssistant: {turn['assistant']}\\n"
            context += f"User: {current_input}\\nAssistant:"
            return context
