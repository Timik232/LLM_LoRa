# Testing Guide for DPO and GRPO

This directory contains comprehensive tests for the DPO (Direct Preference Optimization) and GRPO (Group Relative Policy Optimization) training modules.

## Setup

1. Install test dependencies:
```bash
pip install -r tests/test_requirements.txt
```

2. Ensure you're in the project root directory:
```bash
cd T:\projects\LLM_LoRa\LLM_LoRa
```

## Running Tests

### Run all tests:
```bash
pytest tests/test_dpo_grpo.py -v
```

### Run specific test classes:
```bash
# Test only DPO functionality
pytest tests/test_dpo_grpo.py::TestDPOConfiguration -v
pytest tests/test_dpo_grpo.py::TestDPODataPreparation -v

# Test only GRPO functionality
pytest tests/test_dpo_grpo.py::TestGRPOConfiguration -v
pytest tests/test_dpo_grpo.py::TestGRPORewardFunction -v
pytest tests/test_dpo_grpo.py::TestGRPODataPreparation -v

# Test integration
pytest tests/test_dpo_grpo.py::TestTrainingIntegration -v
```

### Run with coverage:
```bash
pytest tests/test_dpo_grpo.py --cov=training_model --cov-report=html
```

### Run in parallel:
```bash
pytest tests/test_dpo_grpo.py -n auto
```

## Test Structure

### DPO Tests (`TestDPOConfiguration`, `TestDPODataPreparation`)
- Configuration validation
- Data format validation
- Data processing and preparation
- Error handling

### GRPO Tests (`TestGRPOConfiguration`, `TestGRPORewardFunction`, `TestGRPODataPreparation`)
- Configuration validation
- Reward function logic
- Data format validation
- Data processing and preparation
- Error handling

### Integration Tests (`TestTrainingIntegration`)
- Training pipeline integration
- Mock trainer interactions
- Configuration passing
- Memory management

### Data Format Tests (`TestDataFormatValidation`)
- Schema validation for DPO/GRPO data
- File format requirements
- Structure compliance

### Error Handling Tests (`TestErrorHandling`)
- Invalid configurations
- Missing data files
- Malformed data
- Exception scenarios

## Test Coverage

The test suite covers:
- ✅ Configuration validation
- ✅ Data preparation and processing
- ✅ Reward function logic (GRPO)
- ✅ Training integration
- ✅ Data format validation
- ✅ Error handling and edge cases
- ✅ Mock trainer interactions
- ✅ File I/O operations
- ✅ JSON parsing and validation

## Mocking Strategy

Tests use extensive mocking to avoid:
- Actual model loading (resource intensive)
- Network calls
- Large file I/O operations
- GPU operations
- External dependencies

Key mocked components:
- `DPOTrainer` and `GRPOTrainer` from TRL
- Model and tokenizer loading
- Hydra's `get_original_cwd()`
- File system operations where needed

## Debugging Failed Tests

1. **Run with detailed output:**
```bash
pytest tests/test_dpo_grpo.py -v -s --tb=long
```

2. **Run a specific failing test:**
```bash
pytest tests/test_dpo_grpo.py::TestClass::test_method -v -s
```

3. **Check test data:**
The tests create temporary directories and files. If tests fail, check that your data files match the expected format.

## Adding New Tests

When adding new functionality:

1. Add test methods to appropriate test classes
2. Use the provided fixtures for common data
3. Mock external dependencies
4. Test both success and failure scenarios
5. Update this README if needed

## Fixtures

The test file includes several fixtures:
- `sample_dpo_config`: Standard DPO configuration
- `sample_grpo_config`: Standard GRPO configuration
- `sample_dpo_data`: Sample DPO training data
- `sample_grpo_data`: Sample GRPO training data

Use these fixtures in new tests to maintain consistency.
