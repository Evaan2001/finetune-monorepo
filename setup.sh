#!/bin/bash
# Script to create the Finetune Monorepo project structure

set -e  # Exit on any error

echo "Creating Finetune Monorepo directory structure..."

# Create top-level files
touch bazel_requirements_lock.txt
touch .bazelrc
touch .gitignore
touch BUILD
touch MODULE.bazel

# Create src directory and subdirectories
mkdir -p src/{cache_manager,training}

# Create files in src/cache_manager
touch src/cache_manager/BUILD
touch src/cache_manager/cache_manager.py
touch src/cache_manager/run_cache_manager.sh

# Create files in src/training
touch src/training/BUILD
touch src/training/train_model.py
touch src/training/run_training.sh

# Make all shell scripts executable
find . -name "*.sh" -exec chmod +x {} \;
echo "Made all shell scripts executable"

echo "Finetune Monorepo directory structure created successfully!"
echo "Directory structure:"
find . -type f | sort

EOF

echo "Setup complete! Your Finetune Monorepo is ready for development."
