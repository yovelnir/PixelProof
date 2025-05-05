#!/bin/bash

# Exit on any error
set -e

CONTAINER_NAME="pixelproof-backend"
IMAGE_NAME="pixelproof-backend"

# Function for cleanup
cleanup() {
    echo "Cleaning up any existing containers..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Cleanup first
cleanup

# Change to the backend directory
cd "$(dirname "$0")/backend"

echo "Building the backend Docker image..."
docker build -t $IMAGE_NAME .

echo "Starting the backend container..."
docker run --name $CONTAINER_NAME \
    -p 5000:5000 \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/uploads:/app/uploads" \
    -e PYTHONPATH=/app \
    $IMAGE_NAME

# Register cleanup function for Ctrl+C
trap cleanup INT TERM

# Note: The container will remain running in the foreground
# Press Ctrl+C to stop it 