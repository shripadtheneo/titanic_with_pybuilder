#!/bin/bash
# Initialize repository variable
REPOSITORY="dev"

# Check if project name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 project_name [version] [upload]"
    echo "Example: $0 titanic 0.2.1 upload"
    echo "Parameters:"
    echo "  project_name: Name of the project"
    echo "  version: (Optional) Version of the package (default: 0.1.0)"
    echo "  upload: (Optional) If 'upload' is specified, upload to PyPI"
    exit 1
fi

# Store project name from command line
PROJECT_NAME=$1

# Set version (default to 0.1.0 if not provided)
VERSION=${2:-"0.1.0"}
UPLOAD=0

# Get ENV from the 4th argument if provided
if [ $# -ge 4 ]; then
    ENV=$4
    REPOSITORY="$ENV"
    echo "Using repository: $REPOSITORY"
fi

# Check if upload is requested
if [ $# -ge 3 ] && [ "$3" = "upload" ]; then
    UPLOAD=1
    
    # Check if twine is installed
    if ! pip show twine > /dev/null 2>&1; then
        echo "Twine not found. Installing twine..."
        pip install twine
    fi
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        echo "AWS CLI not found. Please install it to use CodeArtifact authentication."
        echo "Installation guide: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
        exit 1
    fi

    # Use AWS CodeArtifact login for twine
    echo "Setting up AWS CodeArtifact authentication..."
    aws codeartifact login --tool twine --repository $REPOSITORY --domain ml-repository --domain-owner 127214153984 --region ap-northeast-1
    
    if [ $? -ne 0 ]; then
        echo "Failed to authenticate with CodeArtifact. Check your AWS credentials and permissions."
        exit 1
    fi
    
    echo "Successfully configured CodeArtifact authentication."
fi

# Update version in build.py if it exists
if [ -f build.py ]; then
    echo "Updating version in build.py..."
    sed -i.bak "s/version = \"[0-9\.]*\"/version = \"$VERSION\"/" build.py
    rm -f build.py.bak
fi

# Build and publish the project
echo "Building project: $PROJECT_NAME (version $VERSION)"
pyb install_dependencies && pyb && pyb publish

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Build failed. Please check the errors above."
    exit 1
fi

# Uninstall existing package and install the new one
echo "Uninstalling existing package and installing new version..."
pip uninstall -y "$PROJECT_NAME" && pip install "target/dist/${PROJECT_NAME}-${VERSION}/dist/${PROJECT_NAME}-${VERSION}-py3-none-any.whl"

echo "Successfully installed $PROJECT_NAME package (version $VERSION)"

# Upload to PyPI if requested
if [ $UPLOAD -eq 1 ]; then
    echo "Uploading package to PyPI..."
    twine upload -r codeartifact --skip-existing "target/dist/${PROJECT_NAME}-${VERSION}/dist/${PROJECT_NAME}-${VERSION}-py3-none-any.whl" "target/dist/${PROJECT_NAME}-${VERSION}/dist/${PROJECT_NAME}-${VERSION}.tar.gz"
    
    # Check if upload was successful
    if [ $? -ne 0 ]; then
        echo "Upload to PyPI failed. Please check the errors above."
        exit 1
    fi
    
    echo "Successfully uploaded $PROJECT_NAME (version $VERSION) to PyPI"
fi

# Print usage instructions
echo ""
echo "You can now use this package by importing it in Python:"
echo "  import $PROJECT_NAME"
echo ""
echo "Or run the CLI tool:"
echo "  ${PROJECT_NAME} --help"