#!/bin/bash
# easy-oww Setup Script
# Automates virtual environment creation and dependency installation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Check Python version
check_python() {
    print_header "Checking Python Version"

    # Try to find a compatible Python version (3.7-3.11)
    PYTHON_CMD=""

    # First, check if default python3 is compatible
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
        PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 7 ] && [ "$PYTHON_MINOR" -le 11 ]; then
            PYTHON_CMD="python3"
            print_success "Python $PYTHON_VERSION detected"
            return
        elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -gt 11 ]; then
            print_warning "Python $PYTHON_VERSION detected (too new, requires 3.7-3.11)"
            print_info "Searching for compatible Python version..."
        else
            print_error "Python $PYTHON_VERSION detected (too old, requires 3.7-3.11)"
            echo "Please install Python 3.7-3.11 from https://www.python.org/downloads/"
            exit 1
        fi
    else
        print_info "python3 not found, searching for compatible version..."
    fi

    # Try to find specific Python versions (prefer newer compatible versions)
    for version in 3.11 3.10 3.9 3.8 3.7; do
        if command -v python${version} &> /dev/null; then
            PYTHON_CMD="python${version}"
            PYTHON_VERSION=$(${PYTHON_CMD} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
            print_success "Found compatible Python $PYTHON_VERSION at $(which ${PYTHON_CMD})"
            return
        fi
    done

    # No compatible Python found
    print_error "No compatible Python version found (requires 3.7-3.11)"
    echo ""
    echo "Please install a compatible Python version:"
    echo "  - On macOS with Homebrew: brew install python@3.11"
    echo "  - Download from: https://www.python.org/downloads/"
    exit 1
}

# Create virtual environment
create_venv() {
    print_header "Creating Virtual Environment"

    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf venv
        else
            print_info "Using existing virtual environment"
            return
        fi
    fi

    print_info "Creating virtual environment with ${PYTHON_CMD}..."
    ${PYTHON_CMD} -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_header "Upgrading pip"
    print_info "Upgrading pip to latest version..."
    pip install --upgrade pip -q
    print_success "pip upgraded successfully"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"

    print_info "Installing easy-oww and dependencies..."
    print_warning "This may take a few minutes..."

    pip install -e . -q

    print_success "All dependencies installed"
}

# Fix entry point script for editable install
fix_entry_point() {
    print_info "Fixing entry point script..."

    SCRIPT_PATH="venv/bin/easy-oww"
    PYTHON_PATH="$(pwd)/venv/bin/python3"

    if [ -f "$SCRIPT_PATH" ]; then
        cat > "$SCRIPT_PATH" << ENTRY
#!${PYTHON_PATH}
# -*- coding: utf-8 -*-
import re
import sys
import os

# Add the project root to sys.path for editable install
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from easy_oww.cli.main import cli

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\\\\.pyw|\\\\.exe)?$', '', sys.argv[0])
    sys.exit(cli())
ENTRY
        chmod +x "$SCRIPT_PATH"
        print_success "Entry point fixed"
    fi
}

# Download OpenWakeWord models
download_oww_models() {
    print_header "Downloading OpenWakeWord Models"

    print_info "Downloading required model files..."
    python3 -c "from openwakeword import utils; utils.download_models(['alexa'])" 2>/dev/null

    if [ $? -eq 0 ]; then
        print_success "OpenWakeWord models downloaded"
    else
        print_warning "Failed to download models automatically"
        print_info "You can download them later with:"
        echo "   python3 -c \"from openwakeword import utils; utils.download_models(['alexa'])\""
    fi
}

# Install dev dependencies
install_dev_dependencies() {
    print_header "Installing Development Dependencies (Optional)"

    read -p "Install development dependencies (pytest, etc.)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing dev dependencies..."
        pip install pytest pytest-mock -q
        print_success "Development dependencies installed"
    else
        print_info "Skipping development dependencies"
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"

    if command -v easy-oww &> /dev/null; then
        VERSION=$(easy-oww --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
        print_success "easy-oww is installed (version $VERSION)"
    else
        print_warning "easy-oww command not found in PATH"
        print_info "You can run it with: python -m easy_oww.cli.main"
    fi

    # Test imports
    print_info "Testing Python imports..."
    if python -c "from easy_oww.audio import AudioRecorder; from easy_oww.tts import PiperTTS; from easy_oww.training import TrainingConfig; from easy_oww.testing import ModelDetector" 2>/dev/null; then
        print_success "All modules import successfully"
    else
        print_error "Some modules failed to import"
        exit 1
    fi
}

# Show next steps
show_next_steps() {
    print_header "Setup Complete!"

    echo "To start using easy-oww:"
    echo ""
    echo "1. Activate the virtual environment:"
    echo -e "   ${GREEN}source venv/bin/activate${NC}"
    echo ""
    echo "2. Initialize your workspace:"
    echo -e "   ${GREEN}easy-oww init${NC}"
    echo ""
    echo "3. Download required datasets:"
    echo -e "   ${GREEN}easy-oww download --required-only${NC}"
    echo ""
    echo "4. Download TTS voices:"
    echo -e "   ${GREEN}easy-oww download-voices${NC}"
    echo ""
    echo "5. Create your first project:"
    echo -e "   ${GREEN}easy-oww create my_wake_word --wake-word \"hey assistant\"${NC}"
    echo ""
    echo "For more information:"
    echo "  - README.md - Complete guide"
    echo "  - INSTALLATION.md - Detailed installation instructions"
    echo "  - QUICKSTART.md - Step-by-step tutorial"
    echo ""
    print_info "Remember to activate the virtual environment before using easy-oww:"
    echo -e "   ${GREEN}source venv/bin/activate${NC}"
    echo ""
}

# Main installation flow
main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                                        ║${NC}"
    echo -e "${BLUE}║         easy-oww Setup Script          ║${NC}"
    echo -e "${BLUE}║                                        ║${NC}"
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo ""

    check_python
    create_venv
    activate_venv
    upgrade_pip
    install_dependencies
    fix_entry_point
    download_oww_models
    install_dev_dependencies
    verify_installation
    show_next_steps
}

# Run main function
main
