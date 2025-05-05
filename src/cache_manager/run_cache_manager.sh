#!/bin/bash

# Get the directory where this script is located
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Cache Manager trial from directory: $DIR"
echo "Python version:"
python3 --version

# Run the Python script
echo "Running Python script ..."
python3 "$DIR/cache_manager.py"

# Capture the exit code
exit_code=$?

echo "Python script completed with exit code: $exit_code"

# Return the same exit code
exit $exit_code
