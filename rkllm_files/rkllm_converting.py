#!/usr/bin/env python3
"""RKLLM Model Conversion Tool

Converts Hugging Face models to RKLLM format for Rockchip NPU deployment.
Supports both Hugging Face and GGUF model formats.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from rkllm.api import RKLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set PyTorch CUDA memory configuration for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert models to RKLLM format for Rockchip NPU deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to input model (Hugging Face directory or GGUF file)",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for RKLLM model file",
    )

    # RKLLM configuration
    parser.add_argument(
        "--target-platform",
        default="rk3588",
        choices=["rk3588", "rk3576", "rk3562", "rk3566"],
        help="Target Rockchip platform",
    )
    parser.add_argument(
        "--quantization",
        default="w8a8",
        choices=["w8a8", "w4a16", "w4a16_g128"],
        help="Quantization type",
    )
    parser.add_argument(
        "--num-npu-core",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of NPU cores to use",
    )
    parser.add_argument(
        "--do-parallelize",
        action="store_true",
        help="Enable model parallelization for larger models",
    )
    parser.add_argument(
        "--hybrid-quantization",
        action="store_true",
        help="Enable hybrid quantization",
    )

    # Model loading options
    parser.add_argument(
        "--model-format",
        choices=["huggingface", "gguf", "auto"],
        default="auto",
        help="Model format (auto-detected by default)",
    )
    parser.add_argument("--max-context", type=int, default=4096, help="Maximum context length")

    return parser.parse_args()


def detect_model_format(model_path: str) -> str:
    """Auto-detect model format based on path."""
    model_path = Path(model_path)

    if model_path.suffix.lower() == ".gguf":
        return "gguf"
    if model_path.is_dir() and (model_path / "config.json").exists():
        return "huggingface"

    # Default to huggingface if uncertain
    return "huggingface"


def load_model(rkllm: RKLLM, model_path: str, model_format: str) -> int:
    """Load model into RKLLM based on format."""
    logger.info(f"Loading {model_format} model from: {model_path}")

    if model_format == "gguf":
        return rkllm.load_gguf(model=model_path)
    if model_format == "huggingface":
        # Use CPU device for loading to avoid CUDA issues in container
        return rkllm.load_huggingface(model=model_path, device="cpu")
    raise ValueError(f"Unsupported model format: {model_format}")


def convert_model(args: argparse.Namespace) -> int:
    """Main conversion function."""
    logger.info("Starting RKLLM model conversion...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Target platform: {args.target_platform}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"NPU cores: {args.num_npu_core}")

    # Validate input model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return 1

    # Create output directory if needed
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect model format if needed
    model_format = args.model_format
    if model_format == "auto":
        model_format = detect_model_format(args.model_path)
        logger.info(f"Auto-detected model format: {model_format}")

    try:
        # Initialize RKLLM
        logger.info("Initializing RKLLM...")
        rkllm = RKLLM()

        # Load model
        ret = load_model(rkllm, args.model_path, model_format)
        if ret != 0:
            logger.error(f"Failed to load {model_format} model")
            return ret
        logger.info("Model loaded successfully")

        # Build model with quantization
        logger.info("Building model with RKLLM optimizations...")
        ret = rkllm.build(
            do_quantization=True,
            optimization_level=1,
            quantized_dtype=args.quantization,
            hybrid_rate=0.5 if args.hybrid_quantization else 0.0,
            max_context=args.max_context,
            quantized_algorithm="normal",
            target_platform=args.target_platform,
            num_npu_core=args.num_npu_core,
            extra_qparams=None,
        )

        if ret != 0:
            logger.error("Model build failed")
            return ret
        logger.info("Model built successfully")

        # Export model
        logger.info(f"Exporting RKLLM model to: {output_path}")
        ret = rkllm.export_rknn(str(output_path))
        if ret != 0:
            logger.error("Model export failed")
            return ret

        # Verify output file
        if not output_path.exists():
            logger.error(f"Output file was not created: {output_path}")
            return 1

        file_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info("RKNN conversion completed successfully!")
        logger.info(f"Output file: {output_path}")
        logger.info(f"File size: {file_size:.2f} MB")

        return 0

    except Exception as e:
        logger.exception(f"Conversion failed with error: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    return convert_model(args)


if __name__ == "__main__":
    sys.exit(main())
