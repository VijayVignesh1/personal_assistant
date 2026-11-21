#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$VIRTUAL_ENV"
PYTHON="$VIRTUAL_ENV/bin/python"

echo "=================================="
echo " Personal Assistant Installation"
echo "=================================="

# Ensure pip is available
echo "Checking pip..."

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "Installing pip..."
    "$PYTHON" -m ensurepip --upgrade
fi

# Upgrade pip

echo "Upgrading pip..."
"$PYTHON" -m pip install --upgrade pip

# Install UV inside virtual environment

echo "Installing UV..."
"$PYTHON" -m pip install uv

# Install Personal Assistant

echo "Installing Personal Assistant..."
"$VENV_DIR/bin/uv" pip install \
    --python "$PYTHON" \
    $PROJECT_ROOT --no-deps


# Install additional dependencies
echo "Installing additional dependencies..."
"$PYTHON" -m pip install psutil py-cpuinfo pyyaml --no-deps


echo "=================================="
echo " Installation completed!"
echo "=================================="

echo ""
echo "To install Personal Assistant, run:"
echo "personal-assistant-install"

echo ""
echo "To install Personal Assistant in developer mode, run:"
echo "personal-assistant-install --dev"