# Dockerfile

# Use a single, simple base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install all the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# THE FIX: Use the correct --bind argument for Gunicorn and add a long timeout
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "app:app"]