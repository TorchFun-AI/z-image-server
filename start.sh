#!/bin/bash

# Z-Image-Turbo API Server Startup Script

cd "$(dirname "$0")/server"

# Activate virtual environment if exists
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
    echo "Virtual environment activated"
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found, using system Python"
fi

echo "Starting Z-Image-Turbo API Server..."
echo ""

python api_server.py
