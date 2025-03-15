#/bin/bash

# Check if project name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 project_name"
    echo "Example: $0 titanic"
    exit 1
fi

# Store project name from command line
PROJECT_NAME=$1

# Build and publish the project
echo "Building project: $PROJECT_NAME"
pyb install_dependencies && pyb && pyb publish

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Build failed. Please check the errors above."
    exit 1
fi

# Uninstall existing package and install the new one
echo "Uninstalling existing package and installing new version..."
pip uninstall -y "$PROJECT_NAME" && pip install "target/dist/${PROJECT_NAME}-0.1.0/dist/${PROJECT_NAME}-0.1.0-py3-none-any.whl"

echo "Successfully installed $PROJECT_NAME package"