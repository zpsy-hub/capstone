#!/bin/bash

# =================================================================
# MAY PASOK BA - Dashboard Launcher Script
# =================================================================
# This script handles virtual environment setup, dependency installation,
# and launching the Streamlit dashboard automatically.
# 
# Usage: ./launch_dashboard.sh
# =================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="may_pasok_ba_env"
PYTHON_VERSION="python3"
DASHBOARD_FILE="dashboard.py"
PORT="8501"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}"
    echo "=================================================="
    echo "     🌧️  MAY PASOK BA DASHBOARD LAUNCHER 🌧️"
    echo "=================================================="
    echo -e "${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH!"
        print_error "Please install Python 3.7+ and try again."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    MAJOR_VER=$(echo $PYTHON_VER | cut -d. -f1)
    MINOR_VER=$(echo $PYTHON_VER | cut -d. -f2)
    
    if [ "$MAJOR_VER" -eq 3 ] && [ "$MINOR_VER" -ge 7 ]; then
        print_success "Python $($PYTHON_CMD --version 2>&1) found ✓"
    else
        print_error "Python 3.7+ is required. Found: $($PYTHON_CMD --version 2>&1)"
        exit 1
    fi
}

# Function to setup virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment '$VENV_NAME' already exists."
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing virtual environment..."
            rm -rf "$VENV_NAME"
        else
            print_status "Using existing virtual environment..."
            return 0
        fi
    fi
    
    print_status "Creating virtual environment '$VENV_NAME'..."
    $PYTHON_CMD -m venv "$VENV_NAME"
    
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created successfully ✓"
    else
        print_error "Failed to create virtual environment!"
        exit 1
    fi
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [ -f "$VENV_NAME/bin/activate" ]; then
        # Unix/Linux/macOS
        source "$VENV_NAME/bin/activate"
        print_success "Virtual environment activated ✓"
    elif [ -f "$VENV_NAME/Scripts/activate" ]; then
        # Windows (Git Bash)
        source "$VENV_NAME/Scripts/activate"
        print_success "Virtual environment activated ✓"
    else
        print_error "Could not find activation script!"
        print_error "Please check if virtual environment was created properly."
        exit 1
    fi
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    python -m pip install --upgrade pip
    if [ $? -eq 0 ]; then
        print_success "Pip upgraded successfully ✓"
    else
        print_warning "Pip upgrade failed, but continuing..."
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing required packages..."
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        print_status "Found requirements.txt, installing from file..."
        pip install -r requirements.txt
    else
        print_status "Installing individual packages..."
        pip install streamlit pandas numpy plotly feedparser
    fi
    
    if [ $? -eq 0 ]; then
        print_success "All dependencies installed successfully ✓"
    else
        print_error "Failed to install dependencies!"
        exit 1
    fi
}

# Function to check if dashboard file exists
check_dashboard_file() {
    print_status "Checking for dashboard file..."
    
    if [ ! -f "$DASHBOARD_FILE" ]; then
        print_error "Dashboard file '$DASHBOARD_FILE' not found!"
        print_error "Please make sure the file exists in the current directory."
        exit 1
    fi
    
    print_success "Dashboard file found ✓"
}

# Function to create requirements.txt if it doesn't exist
create_requirements() {
    if [ ! -f "requirements.txt" ]; then
        print_status "Creating requirements.txt file..."
        cat > requirements.txt << EOF
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
feedparser>=6.0.10
EOF
        print_success "requirements.txt created ✓"
    fi
}

# Function to launch Streamlit
launch_streamlit() {
    print_status "Launching Streamlit dashboard..."
    print_status "Dashboard will be available at: http://localhost:$PORT"
    print_status "Press Ctrl+C to stop the server"
    echo
    print_success "🚀 Starting May Pasok Ba Dashboard..."
    echo
    
    # Launch Streamlit
    streamlit run "$DASHBOARD_FILE" --server.port=$PORT --server.headless=true
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi
    print_success "Cleanup completed ✓"
}

# Function to display help
show_help() {
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -p, --port     Specify port (default: 8501)"
    echo "  -c, --clean    Remove virtual environment and exit"
    echo "  --no-browser   Don't open browser automatically"
    echo
    echo "Examples:"
    echo "  $0                    # Standard launch"
    echo "  $0 -p 8502          # Launch on port 8502"
    echo "  $0 -c               # Clean virtual environment"
    echo
}

# Function to clean virtual environment
clean_env() {
    print_status "Cleaning virtual environment..."
    if [ -d "$VENV_NAME" ]; then
        rm -rf "$VENV_NAME"
        print_success "Virtual environment removed ✓"
    else
        print_warning "No virtual environment found to clean."
    fi
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -c|--clean)
            clean_env
            ;;
        --no-browser)
            export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    # Set trap for cleanup
    trap cleanup EXIT INT TERM
    
    print_header
    
    # Step 1: Check Python
    check_python
    
    # Step 2: Check dashboard file
    check_dashboard_file
    
    # Step 3: Create requirements.txt if needed
    create_requirements
    
    # Step 4: Setup virtual environment
    setup_venv
    
    # Step 5: Activate virtual environment
    activate_venv
    
    # Step 6: Upgrade pip
    upgrade_pip
    
    # Step 7: Install dependencies
    install_dependencies
    
    # Step 8: Launch Streamlit
    launch_streamlit
}

# Run main function
main "$@"