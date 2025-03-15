# Use Python 3.11.8 as base image
FROM python:3.11.8-slim

RUN pip install pybuilder
# Set working directory in the container
WORKDIR /app

# Copy requirements file (if exists)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

RUN pyb install_dependencies && pyb && pyb publish
RUN pip install target/dist/titanic-0.1.0/dist/titanic-0.1.0-py3-none-any.whl


# Command to run when the container starts
CMD ["python", "app.py"]