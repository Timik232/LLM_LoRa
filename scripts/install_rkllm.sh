#!/bin/bash
# install_rkllm.sh - Robust RKLLM toolkit installation with comprehensive error handling
# Based on design.md analysis and diagnostic findings

set -e  # Exit on any error

# Configuration
RKLLM_VERSIONS=("1.0.11" "1.0.10" "1.0.9" "1.0.8")
PYTHON_VERSION="cp311"
ARCH="linux_x86_64"
TMP_DIR="/tmp"
LOG_FILE="/tmp/rkllm_install.log"

# Logging function
log() {
    local level="$1"
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Error handling function
handle_error() {
    local exit_code=$?
    log "ERROR" "Installation failed with exit code $exit_code"
    log "ERROR" "Check log file: $LOG_FILE for detailed information"
    exit $exit_code
}

trap handle_error ERR

log "INFO" "Starting RKLLM toolkit installation"
log "INFO" "Python version: $PYTHON_VERSION, Architecture: $ARCH"

# Function to test RKLLM import
test_rkllm_import() {
    log "INFO" "Testing RKLLM import..."
    if python -c "import rkllm; print('RKLLM import successful')" 2>/dev/null; then
        log "INFO" "✓ RKLLM import test passed"
        return 0
    else
        log "WARN" "✗ RKLLM import test failed"
        return 1
    fi
}

# Function to test RKLLM API import
test_rkllm_api_import() {
    log "INFO" "Testing RKLLM API import..."
    if python -c "from rkllm.api import RKLLM; print('RKLLM API import successful')" 2>/dev/null; then
        log "INFO" "✓ RKLLM API import test passed"
        return 0
    else
        log "WARN" "✗ RKLLM API import test failed"
        return 1
    fi
}

# Function to install from wheel
install_from_wheel() {
    local version=$1
    local wheel_name="rkllm_toolkit-${version}-${PYTHON_VERSION}-${PYTHON_VERSION}-${ARCH}.whl"
    local url="https://github.com/airockchip/rknn-llm/releases/download/v${version}/${wheel_name}"
    local wheel_path="$TMP_DIR/$wheel_name"
    
    log "INFO" "Attempting to install RKLLM toolkit v${version} from wheel..."
    log "INFO" "URL: $url"
    
    # Download wheel
    if wget -q --timeout=30 --tries=2 "$url" -O "$wheel_path" 2>>"$LOG_FILE"; then
        log "INFO" "✓ Downloaded $wheel_name successfully ($(du -h "$wheel_path" | cut -f1))"
    else
        log "WARN" "✗ Failed to download $wheel_name"
        rm -f "$wheel_path"
        return 1
    fi
    
    # Verify download
    if [[ ! -f "$wheel_path" ]] || [[ ! -s "$wheel_path" ]]; then
        log "ERROR" "Downloaded wheel file is empty or missing"
        rm -f "$wheel_path"
        return 1
    fi
    
    # Install wheel
    log "INFO" "Installing wheel file..."
    if pip install "$wheel_path" --no-deps 2>>"$LOG_FILE"; then
        log "INFO" "✓ Wheel installation completed"
        rm -f "$wheel_path"
        
        # Test import
        if test_rkllm_import && test_rkllm_api_import; then
            log "INFO" "✓ RKLLM toolkit v${version} installation successful and verified"
            return 0
        else
            log "WARN" "✗ RKLLM toolkit installed but import failed"
            return 1
        fi
    else
        log "WARN" "✗ Failed to install wheel $wheel_name"
        rm -f "$wheel_path"
        return 1
    fi
}

# Function to install from source
install_from_source() {
    local repo_dir="$TMP_DIR/rknn-llm"
    
    log "INFO" "Attempting source installation from GitHub repository..."
    
    # Clean up any existing repo
    rm -rf "$repo_dir"
    
    # Clone repository
    if git clone https://github.com/airockchip/rknn-llm.git --depth 1 "$repo_dir" 2>>"$LOG_FILE"; then
        log "INFO" "✓ Repository cloned successfully"
    else
        log "WARN" "✗ Failed to clone repository"
        return 1
    fi
    
    # Check if toolkit directory exists
    if [[ ! -d "$repo_dir/rkllm-toolkit" ]]; then
        log "WARN" "✗ rkllm-toolkit directory not found in repository"
        rm -rf "$repo_dir"
        return 1
    fi
    
    # Install from source
    log "INFO" "Installing from source..."
    cd "$repo_dir/rkllm-toolkit"
    if pip install . --no-deps 2>>"$LOG_FILE"; then
        log "INFO" "✓ Source installation completed"
        cd "$TMP_DIR"
        rm -rf "$repo_dir"
        
        # Test import
        if test_rkllm_import && test_rkllm_api_import; then
            log "INFO" "✓ RKLLM toolkit source installation successful and verified"
            return 0
        else
            log "WARN" "✗ RKLLM toolkit source installed but import failed"
            return 1
        fi
    else
        log "WARN" "✗ Source installation failed"
        cd "$TMP_DIR"
        rm -rf "$repo_dir"
        return 1
    fi
}

# Function to verify dependencies
verify_dependencies() {
    log "INFO" "Verifying dependencies..."
    
    # Check rknn-toolkit2
    if python -c "import rknn_toolkit2; print(f'rknn-toolkit2 version: {rknn_toolkit2.__version__}')" 2>>"$LOG_FILE"; then
        log "INFO" "✓ rknn-toolkit2 is available"
    else
        log "WARN" "✗ rknn-toolkit2 not found - installing..."
        if pip install rknn-toolkit2 -i https://mirrors.aliyun.com/pypi/simple 2>>"$LOG_FILE"; then
            log "INFO" "✓ rknn-toolkit2 installed successfully"
        else
            log "ERROR" "✗ Failed to install rknn-toolkit2"
            return 1
        fi
    fi
    
    # Check other dependencies
    local deps=("torch" "numpy" "onnx")
    for dep in "${deps[@]}"; do
        if python -c "import $dep" 2>/dev/null; then
            log "INFO" "✓ $dep is available"
        else
            log "WARN" "✗ $dep not found"
        fi
    done
}

# Function to generate installation report
generate_report() {
    log "INFO" "Generating installation report..."
    
    echo "=== RKLLM Installation Report ===" >> "$LOG_FILE"
    echo "Timestamp: $(date)" >> "$LOG_FILE"
    echo "Python version: $(python --version)" >> "$LOG_FILE"
    echo "Pip version: $(pip --version)" >> "$LOG_FILE"
    
    if test_rkllm_import; then
        python -c "import rkllm; print(f'RKLLM location: {rkllm.__file__}')" 2>>"$LOG_FILE" || true
        python -c "import rkllm; print(f'RKLLM version: {getattr(rkllm, \"__version__\", \"unknown\")}')" 2>>"$LOG_FILE" || true
    fi
    
    echo "Installed packages:" >> "$LOG_FILE"
    pip list | grep -E "(rkllm|rknn)" >> "$LOG_FILE" || true
    echo "=== End Report ===" >> "$LOG_FILE"
}

# Main installation logic
main() {
    log "INFO" "RKLLM toolkit installation started"
    
    # Create log file
    touch "$LOG_FILE"
    
    # Check if already installed
    if test_rkllm_import && test_rkllm_api_import; then
        log "INFO" "✓ RKLLM toolkit is already installed and working"
        generate_report
        exit 0
    fi
    
    # Verify dependencies first
    verify_dependencies
    
    # Try wheel installation for each version
    log "INFO" "Attempting wheel installations..."
    for version in "${RKLLM_VERSIONS[@]}"; do
        if install_from_wheel "$version"; then
            log "INFO" "✓ Successfully installed RKLLM toolkit v$version via wheel"
            generate_report
            exit 0
        fi
        log "WARN" "Failed to install v$version, trying next version..."
    done
    
    # Fallback to source installation
    log "INFO" "All wheel installations failed, attempting source installation..."
    if install_from_source; then
        log "INFO" "✓ Successfully installed RKLLM toolkit from source"
        generate_report
        exit 0
    fi
    
    # All methods failed
    log "ERROR" "All installation methods failed"
    log "ERROR" "Please check the log file: $LOG_FILE for detailed information"
    generate_report
    exit 1
}

# Run main function
main "$@"