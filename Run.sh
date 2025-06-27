echo "Running the project..."

# --- IMPORTANT: Navigate to the script's directory first ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
echo "Navigating to project directory: ${SCRIPT_DIR}"
cd "${SCRIPT_DIR}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change to script directory."
    exit 1
fi

# --- 0. Determine Python Command to Use ---
PYTHON_VERSION_ARG="${1}" # Get the argument passed to the script

if [ -z "${PYTHON_VERSION_ARG}" ]; then
    # No version specified, use default 'python' command
    PYTHON_CMD="python"
    echo "No specific Python version requested. Using default 'python' command on your system."
    echo "Checking default Python version..."
    if command -v "${PYTHON_CMD}" &> /dev/null; then
        DEFAULT_PYTHON_VERSION=$("${PYTHON_CMD}" --version 2>&1)
        echo "Default system Python version: ${DEFAULT_PYTHON_VERSION}"
        if [[ "${DEFAULT_PYTHON_VERSION}" != *Python\ 3.13* ]]; then
            echo "Warning: Your default Python is not 3.13. It's recommended to use Python 3.12 or 3.13 for this project."
            echo "You can specify a version like: ./setup_project.sh 3.12"
        fi
    else
        echo "Error: Default 'python' command not found. Please ensure Python is installed and in your PATH."
        exit 1
    fi
else
    # Version specified, construct command (e.g., python3.12, python3.13)
    PYTHON_CMD="python${PYTHON_VERSION_ARG}"
    echo "Attempting to use Python version: ${PYTHON_VERSION_ARG} (command: ${PYTHON_CMD})"
    if ! command -v "${PYTHON_CMD}" &> /dev/null; then
        echo "Error: Python command '${PYTHON_CMD}' not found."
        echo "Please ensure Python ${PYTHON_VERSION_ARG} is installed and its executable is in your system's PATH."
        exit 1
    fi
fi

# 1. Activate Virtual Environment
echo "2. Activating virtual environment..."
# Check if the Windows-style activate script exists, otherwise use the Linux-style
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment activate script not found in either venv/Scripts or venv/bin."
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment. Please check your setup."
    exit 1
fi
echo "Virtual environment activated. Current Python version in venv:"
python --version # Display the active Python version within the venv

# 2. Run streamlit server
echo "2. Running streamlit server..."
streamlit run app.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to run streamlit server."
    exit 1
fi
