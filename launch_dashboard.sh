#!/bin/bash

# =================================================================
# MAY PASOK BA - Dashboard Launcher Script (v4)
# =================================================================
# This script intelligently handles virtual environment setup, installs
# dependencies from an existing requirements.txt, ensures a .env
# file is present for API keys, and launches the dashboard.
#
# Usage: ./launch_dashboard.sh
# =================================================================

set -e  # Exit on any error

# --- Configuration ---
VENV_NAME="may_pasok_ba_env"
DASHBOARD_FILE="dashboard.py" # Assumes your streamlit file is named this
ENV_FILE=".env"               # The environment file with your API key
REQUIREMENTS_FILE="requirements.txt"
PORT="8501"
PYTHON_CMD="python3" # Default python command

# --- Colors for Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Helper Functions ---
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() {
    echo -e "${CYAN}"
    echo "=================================================="
    echo "      🌧️  MAY PASOK BA DASHBOARD LAUNCHER 🌧️"
    echo "=================================================="
    echo -e "${NC}"
}
command_exists() { command -v "$1" >/dev/null 2>&1; }

# --- Core Script Functions ---

check_prerequisites() {
    print_status "Checking prerequisites..."

    # 1. Check for Python
    if ! command_exists python3 && ! command_exists python; then
        print_error "Python is not installed or not in PATH! Please install Python 3.7+."
        exit 1
    fi
    command_exists python3 && PYTHON_CMD="python3" || PYTHON_CMD="python"
    print_success "Using $($PYTHON_CMD --version) ✓"

    # 2. Check for the main dashboard file
    if [ ! -f "$DASHBOARD_FILE" ]; then
        print_error "Dashboard file '$DASHBOARD_FILE' not found!"
        exit 1
    fi
    print_success "Found dashboard file ($DASHBOARD_FILE) ✓"

    # 3. Check for the .env file (critical for API keys)
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file '$ENV_FILE' not found!"
        print_warning "Please create a '$ENV_FILE' file with your API key."
        exit 1
    fi
    print_success "Found environment file ($ENV_FILE) ✓"
}

setup_venv() {
    if [ -d "$VENV_NAME" ]; then
        print_status "Virtual environment '$VENV_NAME' already exists."
    else
        print_status "Creating virtual environment '$VENV_NAME'..."
        $PYTHON_CMD -m venv "$VENV_NAME"
        print_success "Virtual environment created successfully ✓"
    fi

    # Activate the virtual environment
    print_status "Activating virtual environment..."
    if [ -f "$VENV_NAME/bin/activate" ]; then
        source "$VENV_NAME/bin/activate"
    elif [ -f "$VENV_NAME/Scripts/activate" ]; then # For Windows Git Bash/WSL
        source "$VENV_NAME/Scripts/activate"
    else
        print_error "Could not find activation script! Try recreating the environment."
        exit 1
    fi
    print_success "Virtual environment activated ✓"
}

install_dependencies() {
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_warning "No '$REQUIREMENTS_FILE' found. Skipping dependency installation."
        print_warning "If the app fails, create a '$REQUIREMENTS_FILE' and run this script again."
        return
    fi

    print_status "Checking and installing packages from '$REQUIREMENTS_FILE'..."
    # Pip is efficient and will skip already-installed packages.
    # This is simpler and more reliable than a manual Python check.
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r "$REQUIREMENTS_FILE"

    if [ $? -eq 0 ]; then
        print_success "Dependencies are up to date ✓"
    else
        print_error "Failed to install dependencies!"
        exit 1
    fi
}

launch_streamlit() {
    print_status "Launching Streamlit dashboard..."
    print_status "Dashboard will be available at: http://localhost:$PORT"
    print_status "Press Ctrl+C to stop the server"
    echo
    print_success "🚀 Starting May Pasok Ba Dashboard..."
    echo
    streamlit run "$DASHBOARD_FILE" --server.port=$PORT --server.headless=true
}

# --- Main Execution Logic ---
main() {
    print_header

    # Step 1: Check for Python, dashboard.py, and .env file
    check_prerequisites

    # Step 2: Set up and activate the virtual environment
    setup_venv

    # Step 3: Install dependencies if requirements.txt exists
    install_dependencies

    # Step 4: Launch the application
    launch_streamlit
}

# Run main function
main