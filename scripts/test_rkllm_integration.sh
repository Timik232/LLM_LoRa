#!/bin/bash
# test_rkllm_integration.sh - Comprehensive RKLLM integration test script
# Tests all aspects of RKLLM functionality from installation to model conversion

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_LOG="/tmp/rkllm_integration_test.log"
CONTAINER_NAME="llm_lora-llm_training"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    local color="$2"
    shift 2
    echo -e "${color}[$(date '+%H:%M:%S')] [$level] $*${NC}" | tee -a "$TEST_LOG"
}

log_info() { log "INFO" "$BLUE" "$@"; }
log_success() { log "SUCCESS" "$GREEN" "$@"; }
log_warn() { log "WARN" "$YELLOW" "$@"; }
log_error() { log "ERROR" "$RED" "$@"; }

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# Test function wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log_info "Running test: $test_name"
    
    if $test_function; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        log_success "✓ $test_name PASSED"
        return 0
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        log_error "✗ $test_name FAILED"
        return 1
    fi
}

# Docker utility functions
docker_run_cmd() {
    local cmd="$1"
    docker run --rm --name "rkllm-test-$$" "$CONTAINER_NAME" bash -c "$cmd" 2>&1
}

docker_run_python() {
    local python_cmd="$1"
    docker_run_cmd "python -c \"$python_cmd\""
}

# Test 1: Container existence and basic functionality
test_container_exists() {
    log_info "Checking if container image exists..."
    if docker images | grep -q "$CONTAINER_NAME"; then
        log_success "Container image found: $CONTAINER_NAME"
        return 0
    else
        log_error "Container image not found: $CONTAINER_NAME"
        return 1
    fi
}

# Test 2: RKLLM package import test
test_rkllm_import() {
    log_info "Testing RKLLM package import..."
    local result
    result=$(docker_run_python "import rkllm; print('RKLLM import successful')")
    
    if echo "$result" | grep -q "RKLLM import successful"; then
        log_success "RKLLM package imported successfully"
        return 0
    else
        log_error "RKLLM package import failed"
        echo "$result" | tee -a "$TEST_LOG"
        return 1
    fi
}

# Test 3: RKLLM API import test
test_rkllm_api_import() {
    log_info "Testing RKLLM API import..."
    local result
    result=$(docker_run_python "from rkllm.api import RKLLM; print('RKLLM API import successful')")
    
    if echo "$result" | grep -q "RKLLM API import successful"; then
        log_success "RKLLM API imported successfully"
        return 0
    else
        log_error "RKLLM API import failed"
        echo "$result" | tee -a "$TEST_LOG"
        return 1
    fi
}

# Test 4: RKLLM initialization test
test_rkllm_initialization() {
    log_info "Testing RKLLM initialization..."
    local result
    result=$(docker_run_python "
from rkllm.api import RKLLM
try:
    rkllm = RKLLM()
    print('RKLLM initialization successful')
except Exception as e:
    print(f'RKLLM initialization failed: {e}')
    raise
")
    
    if echo "$result" | grep -q "RKLLM initialization successful"; then
        log_success "RKLLM initialized successfully"
        return 0
    else
        log_error "RKLLM initialization failed"
        echo "$result" | tee -a "$TEST_LOG"
        return 1
    fi
}

# Test 5: Dependencies verification
test_dependencies() {
    log_info "Testing RKLLM dependencies..."
    
    # Test rknn-toolkit2
    local rknn_result
    rknn_result=$(docker_run_python "import rknn_toolkit2; print(f'rknn-toolkit2 version: {rknn_toolkit2.__version__}')")
    
    if echo "$rknn_result" | grep -q "rknn-toolkit2 version:"; then
        log_success "rknn-toolkit2 dependency verified"
    else
        log_error "rknn-toolkit2 dependency missing"
        echo "$rknn_result" | tee -a "$TEST_LOG"
        return 1
    fi
    
    # Test other critical dependencies
    local deps=("torch" "numpy" "onnx")
    for dep in "${deps[@]}"; do
        if docker_run_python "import $dep; print('$dep OK')" | grep -q "$dep OK"; then
            log_success "$dep dependency verified"
        else
            log_warn "$dep dependency issue (may not be critical)"
        fi
    done
    
    return 0
}

# Test 6: Configuration validation test
test_config_validation() {
    log_info "Testing configuration validation functions..."
    local result
    result=$(docker_run_cmd "
cd /app && python -c \"
import sys
sys.path.append('.')
from training_model.one_file_train import validate_rkllm_config_params

# Test valid configuration
try:
    validate_rkllm_config_params('rk3588', 'w8a8', 1)
    print('Valid config validation: PASSED')
except Exception as e:
    print(f'Valid config validation: FAILED - {e}')
    sys.exit(1)

# Test invalid platform
try:
    validate_rkllm_config_params('invalid_platform', 'w8a8', 1)
    print('Invalid platform validation: FAILED - Should have raised error')
    sys.exit(1)
except ValueError:
    print('Invalid platform validation: PASSED')
except Exception as e:
    print(f'Invalid platform validation: FAILED - Wrong exception type: {e}')
    sys.exit(1)

print('Configuration validation tests completed successfully')
\"
")
    
    if echo "$result" | grep -q "Configuration validation tests completed successfully"; then
        log_success "Configuration validation functions working correctly"
        return 0
    else
        log_error "Configuration validation tests failed"
        echo "$result" | tee -a "$TEST_LOG"
        return 1
    fi
}

# Test 7: Enhanced error handling test
test_enhanced_error_handling() {
    log_info "Testing enhanced error handling..."
    local result
    result=$(docker_run_cmd "
cd /app && python -c \"
import sys
sys.path.append('.')
from training_model.one_file_train import safe_import_rkllm

try:
    RKLLM = safe_import_rkllm()
    print('Enhanced error handling test: PASSED - RKLLM imported successfully')
except ImportError as e:
    # Check if we get detailed diagnostic information
    error_msg = str(e)
    if 'DIAGNOSTIC INFORMATION' in error_msg:
        print('Enhanced error handling test: PASSED - Detailed diagnostics provided')
    else:
        print('Enhanced error handling test: FAILED - No detailed diagnostics')
        print(f'Error: {error_msg}')
        sys.exit(1)
except Exception as e:
    print(f'Enhanced error handling test: FAILED - Unexpected error: {e}')
    sys.exit(1)
\"
")
    
    if echo "$result" | grep -q "Enhanced error handling test: PASSED"; then
        log_success "Enhanced error handling working correctly"
        return 0
    else
        log_error "Enhanced error handling test failed"
        echo "$result" | tee -a "$TEST_LOG"
        return 1
    fi
}

# Test 8: Model conversion dry run test (using minimal test model)
test_model_conversion_dry_run() {
    log_info "Testing RKLLM model conversion with dry run..."
    
    # Create a minimal test model structure in the container
    local result
    result=$(docker_run_cmd "
cd /app && python -c \"
import os
import json
import sys
from pathlib import Path

# Create minimal test model directory
test_model_dir = Path('/tmp/test_model')
test_model_dir.mkdir(exist_ok=True)

# Create a minimal config.json
config = {
    'model_type': 'test',
    'vocab_size': 1000,
    'hidden_size': 768
}

with open(test_model_dir / 'config.json', 'w') as f:
    json.dump(config, f)

print(f'Test model directory created at: {test_model_dir}')

# Test our convert_to_rkllm function's validation
sys.path.append('.')
from training_model.one_file_train import validate_rkllm_config_params, safe_import_rkllm

try:
    # Test parameter validation
    validate_rkllm_config_params('rk3588', 'w8a8', 1)
    print('Parameter validation: PASSED')
    
    # Test RKLLM import (should work now)
    RKLLM = safe_import_rkllm()
    print('RKLLM import: PASSED')
    
    print('Model conversion dry run test: PASSED')
    
except Exception as e:
    print(f'Model conversion dry run test: FAILED - {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
\"
")
    
    if echo "$result" | grep -q "Model conversion dry run test: PASSED"; then
        log_success "Model conversion dry run completed successfully"
        return 0
    else
        log_error "Model conversion dry run failed"
        echo "$result" | tee -a "$TEST_LOG"
        return 1
    fi
}

# Test 9: Health check verification
test_health_check() {
    log_info "Testing container health check..."
    
    # Start container in background to test health check
    local container_id
    container_id=$(docker run -d --name "rkllm-health-test-$$" "$CONTAINER_NAME" sleep 30)
    
    # Wait a bit for health check to run
    sleep 5
    
    # Check health status
    local health_status
    health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "no-healthcheck")
    
    # Cleanup
    docker stop "$container_id" >/dev/null 2>&1 || true
    docker rm "$container_id" >/dev/null 2>&1 || true
    
    if [[ "$health_status" == "healthy" ]]; then
        log_success "Container health check is working and healthy"
        return 0
    elif [[ "$health_status" == "starting" ]]; then
        log_warn "Container health check is still starting (this is normal)"
        return 0
    else
        log_error "Container health check failed or not configured: $health_status"
        return 1
    fi
}

# Main test execution
main() {
    log_info "Starting RKLLM Integration Tests"
    log_info "Log file: $TEST_LOG"
    echo "=== RKLLM Integration Test Start ===" > "$TEST_LOG"
    echo "Timestamp: $(date)" >> "$TEST_LOG"
    
    # Run all tests
    run_test "Container Existence" test_container_exists
    run_test "RKLLM Package Import" test_rkllm_import
    run_test "RKLLM API Import" test_rkllm_api_import
    run_test "RKLLM Initialization" test_rkllm_initialization
    run_test "Dependencies Verification" test_dependencies
    run_test "Configuration Validation" test_config_validation
    run_test "Enhanced Error Handling" test_enhanced_error_handling
    run_test "Model Conversion Dry Run" test_model_conversion_dry_run
    run_test "Health Check Verification" test_health_check
    
    # Final report
    echo ""
    log_info "=== TEST RESULTS ==="
    log_info "Total tests: $TOTAL_TESTS"
    log_success "Passed: $TESTS_PASSED"
    if [[ $TESTS_FAILED -gt 0 ]]; then
        log_error "Failed: $TESTS_FAILED"
    else
        log_success "Failed: $TESTS_FAILED"
    fi
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "🎉 All RKLLM integration tests PASSED!"
        log_info "RKLLM is properly installed and functional in the Docker container."
        return 0
    else
        log_error "❌ Some RKLLM integration tests FAILED!"
        log_error "Check the test log for detailed information: $TEST_LOG"
        return 1
    fi
}

# Run main function
main "$@"