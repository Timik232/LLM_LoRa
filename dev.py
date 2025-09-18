#!/usr/bin/env python3
"""
Development utility script for LLM-LoRa project.

This script provides convenient commands for common development tasks including
linting, formatting, testing, and project setup.

Usage:
    python dev.py <command> [options]

Available commands:
    setup       - Install dependencies and setup development environment
    format      - Format code with black and ruff
    lint        - Run linting with ruff
    test        - Run tests with pytest
    check       - Run all quality checks (format, lint, test)
    clean       - Clean up temporary files and caches
    docs        - Build documentation
    env-info    - Display environment information
    validate    - Validate configuration files
    help        - Show this help message

Examples:
    python dev.py setup
    python dev.py format
    python dev.py test --gpu
    python dev.py check
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def log_header(message: str) -> None:
    """Log a formatted header message."""
    logging.info("\n%s%s%s%s", Colors.BOLD, Colors.BLUE, "=" * 60, Colors.END)
    logging.info("%s%s%s%s", Colors.BOLD, Colors.BLUE, message.center(60), Colors.END)
    logging.info("%s%s%s%s\n", Colors.BOLD, Colors.BLUE, "=" * 60, Colors.END)


def log_success(message: str) -> None:
    """Log a success message."""
    logging.info("%s✅ %s%s", Colors.GREEN, message, Colors.END)


def log_error(message: str) -> None:
    """Log an error message."""
    logging.error("%s❌ %s%s", Colors.RED, message, Colors.END)


def log_warning(message: str) -> None:
    """Log a warning message."""
    logging.warning("%s⚠️  %s%s", Colors.YELLOW, message, Colors.END)


def log_info(message: str) -> None:
    """Log an info message."""
    logging.info("%si️  %s%s", Colors.BLUE, message, Colors.END)


def run_command(
    cmd: list[str],
    description: str,
    *,
    check: bool = True,
    cwd: Path | None = None,
) -> bool:
    """Run a command and handle output."""
    log_info(f"Running: {description}")
    logging.info("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=check, cwd=cwd, capture_output=False, text=True)
        success = result.returncode == 0
        if success:
            log_success(f"Completed: {description}")
        else:
            log_error(f"Failed: {description}")
    except subprocess.CalledProcessError as e:
        log_error(f"Command failed: {description}")
        log_error(f"Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        log_error(f"Command not found for: {description}")
        log_error("Make sure all dependencies are installed")
        return False
    else:
        return success


def setup_environment() -> bool:
    """Setup development environment."""
    log_header("Setting up Development Environment")

    project_root = Path(__file__).parent

    # Check if we're in a git repository
    if not (project_root / ".git").exists():
        log_warning("Not in a git repository")

    # Install dependencies with Poetry
    if not run_command(["poetry", "install", "--no-root"], "Installing dependencies"):
        return False

    # Install pre-commit hooks
    if not run_command(
        ["poetry", "run", "pre-commit", "install"],
        "Installing pre-commit hooks",
    ):
        return False

    # Verify CUDA availability
    cuda_check = [
        "python",
        "-c",
        "import torch; print(f'CUDA available: {torch.cuda.is_available()}')",
    ]
    run_command(["poetry", "run", *cuda_check], "Checking CUDA availability", check=False)

    log_success("Development environment setup completed!")
    return True


def format_code() -> bool:
    """Format code with black and ruff."""
    log_header("Formatting Code")

    success = True

    # Run ruff formatting
    if not run_command(
        ["poetry", "run", "ruff", "check", ".", "--fix", "--config=pyproject.toml"],
        "Running ruff formatter",
    ):
        success = False

    # Run black formatting
    if not run_command(["poetry", "run", "black", "."], "Running black formatter"):
        success = False

    if success:
        log_success("Code formatting completed!")
    else:
        log_error("Code formatting had issues")

    return success


def run_linting() -> bool:
    """Run linting with ruff."""
    log_header("Running Linting")

    success = run_command(
        ["poetry", "run", "ruff", "check", ".", "--config=pyproject.toml"],
        "Running ruff linter",
    )

    if success:
        log_success("Linting passed!")
    else:
        log_error("Linting found issues")

    return success


def run_tests(*, gpu: bool = False, unit: bool = False, integration: bool = False) -> bool:
    """Run tests with pytest."""
    log_header("Running Tests")

    cmd = ["poetry", "run", "pytest", "tests/", "-v"]

    # Add markers based on options
    markers = []
    if gpu:
        markers.append("gpu")
    if unit:
        markers.append("unit")
    if integration:
        markers.append("integration")

    if markers:
        cmd.extend(["-m", " or ".join(markers)])

    success = run_command(cmd, f"Running tests{' (GPU)' if gpu else ''}")

    if success:
        log_success("All tests passed!")
    else:
        log_error("Some tests failed")

    return success


def run_quality_checks() -> bool:
    """Run all quality checks."""
    log_header("Running Quality Checks")

    all_passed = True

    # Format code
    if not format_code():
        all_passed = False

    # Run linting
    if not run_linting():
        all_passed = False

    # Run tests
    if not run_tests():
        all_passed = False

    # Run pre-commit hooks
    if not run_command(
        ["poetry", "run", "pre-commit", "run", "--all-files"],
        "Running pre-commit hooks",
    ):
        all_passed = False

    if all_passed:
        log_success("All quality checks passed!")
    else:
        log_error("Some quality checks failed")

    return all_passed


def clean_project() -> None:
    """Clean up temporary files and caches."""
    log_header("Cleaning Project")

    project_root = Path(__file__).parent

    # Directories to clean
    clean_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "*.egg-info",
        "build",
        "dist",
        ".mypy_cache",
        "docs/_build",
    ]

    for pattern in clean_dirs:
        for path in project_root.rglob(pattern):
            if path.is_dir():
                log_info(f"Removing directory: {path}")
                try:
                    shutil.rmtree(path)
                except OSError as e:
                    log_warning(f"Could not remove {path}: {e}")

    # Files to clean
    clean_files = ["*.pyc", "*.pyo", ".coverage", "coverage.xml"]

    for pattern in clean_files:
        for path in project_root.rglob(pattern):
            if path.is_file():
                log_info(f"Removing file: {path}")
                try:
                    path.unlink()
                except OSError as e:
                    log_warning(f"Could not remove {path}: {e}")

    log_success("Project cleanup completed!")


def build_docs() -> bool:
    """Build documentation."""
    log_header("Building Documentation")

    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.exists():
        log_error("Documentation directory not found")
        return False

    # Build HTML documentation
    success = run_command(
        ["sphinx-build", "-b", "html", ".", "_build/html"],
        "Building HTML documentation",
        cwd=docs_dir,
    )

    if success:
        log_success("Documentation built successfully!")
        log_info(f"Documentation available at: {docs_dir / '_build' / 'html' / 'index.html'}")
    else:
        log_error("Documentation build failed")

    return success


def show_env_info() -> None:
    """Display environment information."""
    log_header("Environment Information")

    project_root = Path(__file__).parent

    # Python version
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logger.info("Python Version: %s", python_version)
    logger.info("Python Executable: %s", sys.executable)

    # Project info
    logger.info("Project Root: %s", project_root)
    logger.info("Working Directory: %s", Path.cwd())

    # Git info
    try:
        git_path = shutil.which("git")
        if git_path is None:
            raise FileNotFoundError("Git executable not found.")
        result = subprocess.run(
            [git_path, "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Git Branch: %s", result.stdout.strip())
        git_path = shutil.which("git")
        if git_path is None:
            raise FileNotFoundError("Git executable not found.")

        result = subprocess.run(
            [git_path, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Git Commit: %s", result.stdout.strip())
    except FileNotFoundError:
        logging.info("Git: Not available")

    # Virtual environment info
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        logger.info("Virtual Environment: %s", venv_path)
    else:
        logger.info("Virtual Environment: Not detected")

    # CUDA info
    if torch is not None:
        logger.info("PyTorch Version: %s", getattr(torch, "__version__", "unknown"))
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        logger.info("CUDA Available: %s", cuda_available)

        cuda_version = getattr(torch, "version", None)
        cuda_version_str = (
            getattr(cuda_version, "cuda", None) if cuda_version is not None else None
        )
        logger.info("CUDA Version: %s", cuda_version_str)

        if cuda_available:
            try:
                gpu_count = torch.cuda.device_count()
            except Exception:
                gpu_count = 0
            logger.info("GPU Count: %s", gpu_count)
            if gpu_count > 0:
                try:
                    logger.info("GPU Name: %s", torch.cuda.get_device_name(0))
                except Exception:
                    logger.info("GPU Name: unknown")
    else:
        logger.info("PyTorch: Not available")


def validate_config() -> None:
    """Validate configuration files."""
    log_header("Validating Configuration")

    project_root = Path(__file__).parent

    # Validate main config
    config_file = project_root / "conf" / "config.yaml"
    if config_file.exists():
        cmd = [
            "python",
            "-c",
            f"import yaml; yaml.safe_load(open('{config_file}')); "
            "print('✅ config.yaml is valid')",
        ]
        run_command(["poetry", "run", *cmd], "Validating config.yaml", check=False)
    else:
        log_warning("config.yaml not found")

    # Validate pyproject.toml
    pyproject_file = project_root / "pyproject.toml"
    if pyproject_file.exists():
        cmd = [
            "python",
            "-c",
            f"import tomllib; tomllib.load(open('{pyproject_file}', 'rb')); "
            "print('✅ pyproject.toml is valid')",
        ]
        run_command(["poetry", "run", *cmd], "Validating pyproject.toml", check=False)

    # Test imports of main modules
    modules = ["training_model", "evaluation", "testing_model"]
    for module in modules:
        cmd = ["python", "-c", f"import {module}; print('✅ {module} imports successfully')"]
        run_command(["poetry", "run", *cmd], f"Testing import of {module}", check=False)


def main() -> None:
    """Main entry point."""
    # Configure logging for colored output
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        description="Development utility script for LLM-LoRa project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "command",
        choices=[
            "setup",
            "format",
            "lint",
            "test",
            "check",
            "clean",
            "docs",
            "env-info",
            "validate",
            "help",
        ],
        help="Command to run",
    )

    # Test options
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests (for test command)")
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only (for test command)",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only (for test command)",
    )

    args = parser.parse_args()

    if args.command == "help":
        parser.print_help()
        return

    # Ensure we're in the project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    success = True

    if args.command == "setup":
        success = setup_environment()
    elif args.command == "format":
        success = format_code()
    elif args.command == "lint":
        success = run_linting()
    elif args.command == "test":
        success = run_tests(gpu=args.gpu, unit=args.unit, integration=args.integration)
    elif args.command == "check":
        success = run_quality_checks()
    elif args.command == "clean":
        clean_project()
    elif args.command == "docs":
        success = build_docs()
    elif args.command == "env-info":
        show_env_info()
    elif args.command == "validate":
        validate_config()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
