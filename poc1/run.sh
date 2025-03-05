#!/bin/bash
# API keys should be set in a .env file (copy .env.example to .env and add your API keys)
# Check if .env file exists and load it
if [ -f ../.env ]; then
    set -a
    source ../.env
    set +a
    echo "Loaded API keys from .env file"
else
    echo "WARNING: No .env file found. Please create one with your API keys."
    echo "You can copy .env.example to .env and add your API keys there."
fi

# Change to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if debug mode requested
DEBUG_FLAG=""
if [ "$1" == "--debug" ]; then
    DEBUG_FLAG="--debug"
    echo "Running in debug mode"
fi

# Check if setup is needed
if [ ! -f data/financial.db ]; then
    echo "Database not found. Running setup first..."
    python main.py --setup $DEBUG_FLAG
fi

# Run the chatbot
python main.py --run $DEBUG_FLAG
