# Contributing to LLM-LoRa

We love your input! We want to make contributing to LLM-LoRa as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

**Examples of behavior that contributes to a positive environment:**
- Being respectful and inclusive
- Welcoming newcomers and helping them get started
- Being collaborative and constructive in discussions
- Focusing on what is best for the community
- Showing empathy towards other community members

**Examples of unacceptable behavior:**
- Harassment, trolling, or discriminatory language
- Personal attacks or political arguments
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

- Python 3.11-3.13
- Git
- NVIDIA GPU with CUDA support (for training)
- Docker (optional but recommended)

### Development Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR-USERNAME/LLM_LoRa.git
   cd LLM_LoRa
   ```

2. **Set up Development Environment**
   ```bash
   # Install Poetry
   pip install poetry

   # Install dependencies with development tools
   poetry install

   # Install pre-commit hooks
   poetry run pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run tests to verify everything works
   poetry run pytest tests/

   # Check code formatting
   poetry run black --check .
   poetry run ruff check .
   ```

4. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Development Process

We use [GitHub flow](https://guides.github.com/introduction/flow/index.html), so all code changes happen through Pull Requests.

### Workflow Overview

1. **Issue Discussion**: For major changes, open an issue first to discuss
2. **Fork & Branch**: Create a feature branch from `main`
3. **Develop**: Write code, tests, and documentation
4. **Test**: Ensure all tests pass and add new ones as needed
5. **Submit PR**: Open a pull request with clear description
6. **Review**: Address feedback from maintainers
7. **Merge**: Once approved, your PR will be merged

## How to Contribute

### 🐛 **Bug Reports**

**Before submitting a bug report:**
- Search existing issues to avoid duplicates
- Try to reproduce the bug with the latest version
- Check if the issue exists in a clean environment

**When submitting a bug report, include:**
- **System information**: OS, Python version, GPU details
- **Environment**: Docker vs local, dependencies versions
- **Reproduction steps**: Clear steps to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs/errors**: Full error messages and relevant logs
- **Configuration**: Relevant config files

### ✨ **Feature Requests**

**Before submitting a feature request:**
- Check if the feature already exists
- Search for existing feature requests
- Consider if it fits the project's goals

**When submitting a feature request:**
- **Clear title**: Concise description of the feature
- **Problem statement**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives**: Other solutions you've considered
- **Impact**: Who would benefit from this feature?

### 🔧 **Code Contributions**

**Types of contributions we're looking for:**
- Bug fixes
- Performance improvements
- New training methods or optimizations
- Documentation improvements
- Test coverage improvements
- Model format support (GGUF, RKLLM, etc.)
- Integration improvements (MLflow, Wandb, etc.)

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Update README, docstrings, and relevant docs
2. **Add Tests**: Include tests for new functionality
3. **Run Full Test Suite**: Ensure all tests pass
4. **Check Code Quality**: Run linting and formatting tools
5. **Update Changelog**: Add entry to CHANGELOG.md (if exists)

### PR Guidelines

**Title Format:**
- `feat: add support for new model architecture`
- `fix: resolve memory leak in training loop`
- `docs: update installation instructions`
- `test: add integration tests for GRPO training`
- `refactor: simplify configuration management`

**Description Template:**
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Changes Made
- List of specific changes
- Include technical details
- Mention any breaking changes

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] GPU tests completed (if applicable)

## Documentation
- [ ] Code comments updated
- [ ] README updated (if needed)
- [ ] API documentation updated (if needed)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, maintainers will merge

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# Line length: 95 characters (configured in pyproject.toml)
# Use Black for formatting
poetry run black .

# Use Ruff for linting
poetry run ruff check .
```

### Code Quality Tools

**Automatic Formatting:**
```bash
# Format code with Black
poetry run black .

# Sort imports
poetry run isort .
```

**Linting:**
```bash
# Run Ruff linter
poetry run ruff check .
poetry run ruff check --fix .  # Auto-fix issues
```

**Pre-commit Hooks:**
Pre-commit hooks automatically run on each commit:
- Black (formatting)
- Ruff (linting)
- Type checking with mypy
- YAML/JSON validation

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Files**: `snake_case.py`
- **Directories**: `snake_case`

### Documentation Style

- **Docstrings**: Use Google style docstrings
- **Comments**: Explain why, not what
- **Type hints**: Use type hints for function signatures

```python
def train_model(
    model_name: str,
    config: DictConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train a language model with LoRa fine-tuning.

    Args:
        model_name: Name of the base model from HuggingFace
        config: Hydra configuration object
        output_dir: Directory to save the trained model

    Returns:
        Dictionary containing training metrics and paths

    Raises:
        ValueError: If model_name is not supported
        RuntimeError: If GPU is not available but required
    """
    pass
```

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_training.py
│   └── test_config.py
├── integration/       # Integration tests
│   └── test_full_pipeline.py
├── fixtures/          # Test fixtures
│   └── sample_data.py
└── conftest.py       # pytest configuration
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest -m unit
poetry run pytest -m integration
poetry run pytest -m gpu  # Requires GPU

# Run with coverage
poetry run pytest --cov=training_model

# Run specific test file
poetry run pytest tests/unit/test_training.py
```

### Writing Tests

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **GPU tests**: Mark with `@pytest.mark.gpu`
- **Slow tests**: Mark with `@pytest.mark.slow`

```python
import pytest
from training_model.one_file_train import TrainingPipeline

class TestTrainingPipeline:
    @pytest.fixture
    def sample_config(self):
        return {
            "model_name": "microsoft/DialoGPT-small",
            "train_steps": 10,
        }

    def test_pipeline_initialization(self, sample_config):
        pipeline = TrainingPipeline(sample_config)
        assert pipeline.model_name == sample_config["model_name"]

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_training_step(self, sample_config):
        # Test that requires GPU and takes time
        pass
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and comments
2. **API Documentation**: Auto-generated from docstrings
3. **User Documentation**: README, guides, tutorials
4. **Developer Documentation**: Contributing, architecture docs

### Building Documentation

```bash
# Build Sphinx documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

### Documentation Guidelines

- **Clear and Concise**: Write for your audience
- **Examples**: Include code examples
- **Up-to-date**: Keep documentation current with code
- **Screenshots**: Include visuals when helpful

## Issue Reporting

### Bug Reports

Use the **Bug Report** template and include:

- **Environment details**: OS, Python, GPU, Docker version
- **Reproduction steps**: Minimal example to reproduce
- **Expected vs actual behavior**
- **Error messages**: Full stack traces
- **Configuration files**: Relevant config sections

### Security Issues

**Do not** open public issues for security vulnerabilities. Instead:
1. Email komolov.timurka@gmail.com
2. Include detailed description
3. Allow time for fix before disclosure

## Feature Requests

### Before Requesting

- Check existing issues and discussions
- Consider if it aligns with project goals
- Think about implementation complexity

### Feature Request Template

- **Problem Statement**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other solutions considered
- **Additional Context**: Screenshots, mockups, examples

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcase
- **Documentation**: Official docs and guides

### Getting Help

1. **Check Documentation**: README, docs/, and code comments
2. **Search Issues**: Look for existing solutions
3. **Ask Questions**: Use GitHub Discussions
4. **Report Bugs**: Use GitHub Issues with template

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributor graphs
- Special mentions for significant contributions

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Workflow

1. **Feature Freeze**: No new features for release
2. **Testing**: Comprehensive testing on release branch
3. **Documentation**: Update docs and changelog
4. **Release**: Tag version and publish
5. **Announcement**: Notify community of release

---

## Questions?

Don't hesitate to ask! We're here to help:

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Email**: For security issues or private matters

**Thank you for contributing to LLM-LoRa! 🚀**

---

*This document is adapted from [open-source contribution guidelines](https://github.com/nayafia/contributing-template) and best practices from major open-source projects.*
