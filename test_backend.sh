#!/bin/bash

# Exit on any error
set -e

CONTAINER_NAME="pixelproof-backend-test"
IMAGE_NAME="pixelproof-backend-test"

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

echo "===== BUILDING TEST IMAGE ====="
docker build -t $IMAGE_NAME .

# Ensure we have a test image available
if [ ! -d "test_images" ]; then
    echo "Creating test_images directory..."
    mkdir -p test_images
    
    # If no test images, create a message
    echo "Please add test images to the test_images directory" > test_images/README.txt
fi

echo "===== RUNNING TEST CONTAINER ====="

# Find the first image in test_images directory
TEST_IMAGE=$(ls test_images/*.jpg test_images/*.png test_images/*.jpeg 2>/dev/null | head -1)

if [ -z "$TEST_IMAGE" ]; then
    echo "No test images found. Please add some test images to the test_images directory."
    exit 1
fi

echo "Using test image: $TEST_IMAGE"

# Test the SRM filter implementation first
echo "===== TESTING SRM FILTER APPLICATION ====="
docker run --name "${CONTAINER_NAME}-srm" \
    -v "$(pwd)/test_images:/app/test_images" \
    $IMAGE_NAME \
    python test_srm.py --image "/app/test_images/$(basename "$TEST_IMAGE")"

# Clean up the SRM test container
docker rm "${CONTAINER_NAME}-srm" 2>/dev/null || true

# If a model file is specified and exists, test with the model too
if [ -d "models" ]; then
    MODEL_FILE=$(ls models/*.keras models/*.h5 2>/dev/null | head -1)
    if [ -n "$MODEL_FILE" ]; then
        echo ""
        echo "===== TESTING MODEL PREDICTION ====="
        echo "Using model: $MODEL_FILE"
        
        docker run --name "${CONTAINER_NAME}-model" \
            -v "$(pwd)/models:/app/models" \
            -v "$(pwd)/test_images:/app/test_images" \
            $IMAGE_NAME \
            python test_srm.py --image "/app/test_images/$(basename "$TEST_IMAGE")" --model "/app/models/$(basename "$MODEL_FILE")"
            
        # Clean up the model test container
        docker rm "${CONTAINER_NAME}-model" 2>/dev/null || true
    else
        echo "No model files found in the models directory. Skipping model prediction test."
    fi
else
    echo "No models directory found. Skipping model prediction test."
fi

# Now run the full API test if requested
if [ "$1" == "--full-test" ]; then
    echo ""
    echo "===== RUNNING FULL API TEST ====="
    docker run --name $CONTAINER_NAME \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/test_images:/app/test_images" \
        $IMAGE_NAME \
        python test_models.py --image "/app/test_images/$(basename "$TEST_IMAGE")" --models-dir "/app/models"
        
    # Cleanup the full test container
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

echo "===== TEST COMPLETED =====" 