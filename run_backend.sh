#!/bin/bash

# Enhanced backend runner with evaluation and caching support
# This script is now a wrapper around the Python runner

# Exit on any error
set -e

# Get the directory of this script
SCRIPT_DIR="$(dirname "$0")"

# Display help if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "PixelProof Backend Runner"
    echo "========================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script now supports advanced evaluation and caching options."
    echo "For detailed options, use:"
    echo "  python backend/run_backend.py --help"
    echo ""
    echo "Quick Examples:"
    echo "  $0                          # Run with automatic evaluation/caching"
    echo "  $0 --docker                 # Run with Docker (legacy behavior)"
    echo "  $0 --evaluate-only          # Just run evaluation and exit"
    echo "  $0 --serve-only             # Start server with cached weights"
    echo "  $0 --force-evaluation       # Force fresh evaluation"
    echo "  $0 --cache-info             # Show cache information"
    echo ""
    exit 0
fi

# Check for legacy Docker-only mode
if [[ "$1" == "--docker-legacy" ]]; then
    echo "Running legacy Docker mode..."
    
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
    cd "$SCRIPT_DIR/backend"
    
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
    
    exit 0
fi

# Default: use the enhanced Python runner
echo "Using enhanced backend runner with evaluation and caching support..."
echo "For detailed options, run: $0 --help"
echo ""

# Pass all arguments to the Python runner
python "$SCRIPT_DIR/backend/run_backend.py" "$@" 