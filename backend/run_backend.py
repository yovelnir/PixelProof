#!/usr/bin/env python3
"""
Enhanced backend runner with model evaluation and weight caching support.
"""
import argparse
import sys
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PixelProof Backend Runner')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--evaluate-only', action='store_true',
                           help='Run model evaluation and save weights, then exit')
    mode_group.add_argument('--serve-only', action='store_true',
                           help='Start server using cached weights (no evaluation)')
    mode_group.add_argument('--docker', action='store_true',
                           help='Run using Docker container')
    
    # Evaluation options
    eval_group = parser.add_argument_group('evaluation options')
    eval_group.add_argument('--force-evaluation', action='store_true',
                           help='Force model evaluation even if cache exists')
    eval_group.add_argument('--no-cache', action='store_true',
                           help='Disable weight caching')
    
    # Cache management
    cache_group = parser.add_argument_group('cache management')
    cache_group.add_argument('--cache-info', action='store_true',
                            help='Show cache information and exit')
    cache_group.add_argument('--clear-cache', action='store_true',
                            help='Clear weight cache and exit')
    
    # Server options
    server_group = parser.add_argument_group('server options')
    server_group.add_argument('--port', type=int, default=5000,
                             help='Port to run the server on (default: 5000)')
    server_group.add_argument('--host', default='0.0.0.0',
                             help='Host to bind the server to (default: 0.0.0.0)')
    
    return parser.parse_args()

def run_docker_backend():
    """Run the backend using Docker."""
    logger.info("Running backend with Docker...")
    
    container_name = "pixelproof-backend"
    image_name = "pixelproof-backend"
    
    # Determine correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If this script is in backend/, go up one level for project root
    if os.path.basename(script_dir) == "backend":
        project_root = os.path.dirname(script_dir)
        backend_dir = script_dir
    else:
        project_root = script_dir
        backend_dir = os.path.join(script_dir, "backend")
    
    if not os.path.exists(backend_dir):
        logger.error(f"Backend directory not found: {backend_dir}")
        return False
    
    # Cleanup existing container
    logger.info("Cleaning up any existing containers...")
    subprocess.run(["docker", "stop", container_name], 
                  capture_output=True, check=False)
    subprocess.run(["docker", "rm", container_name], 
                  capture_output=True, check=False)
    
    # Build image
    logger.info("Building Docker image...")
    result = subprocess.run(["docker", "build", "-t", image_name, "."], 
                          cwd=backend_dir, check=False)
    if result.returncode != 0:
        logger.error("Failed to build Docker image")
        return False
    
    # Run container
    logger.info("Starting Docker container...")
    models_path = os.path.join(backend_dir, "models")
    uploads_path = os.path.join(backend_dir, "uploads")
    
    cmd = [
        "docker", "run", "--name", container_name,
        "-p", "5000:5000",
        "-v", f"{models_path}:/app/models",
        "-v", f"{uploads_path}:/app/uploads",
        "-e", "PYTHONPATH=/app",
        image_name
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Stopping container...")
        subprocess.run(["docker", "stop", container_name], check=False)
        subprocess.run(["docker", "rm", container_name], check=False)
    
    return True

def run_native_backend(args):
    """Run the backend natively with Python."""
    logger.info("Running backend natively...")
    
    # Determine backend directory based on where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If this script is in backend/, then backend_dir is current directory
    # If this script is in root/, then backend_dir is backend/
    if os.path.basename(script_dir) == "backend":
        backend_dir = script_dir
    else:
        backend_dir = os.path.join(script_dir, "backend")
    
    if not os.path.exists(backend_dir):
        logger.error(f"Backend directory not found: {backend_dir}")
        return False
    
    # Build command
    cmd = [sys.executable, "app.py"]
    
    # Add arguments
    if args.force_evaluation:
        cmd.append("--force-evaluation")
    if args.no_cache:
        cmd.append("--no-cache")
    if args.evaluate_only:
        cmd.append("--evaluate")
    if args.cache_info:
        cmd.append("--cache-info")
    if args.clear_cache:
        cmd.append("--clear-cache")
    
    cmd.extend(["--host", args.host, "--port", str(args.port)])
    
    # Run the backend
    try:
        result = subprocess.run(cmd, cwd=backend_dir, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        logger.info("Backend stopped by user")
        return True

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle cache-only operations
    if args.cache_info or args.clear_cache:
        return run_native_backend(args)
    
    # Handle evaluation-only mode
    if args.evaluate_only:
        logger.info("Running evaluation-only mode...")
        return run_native_backend(args)
    
    # Handle serve-only mode
    if args.serve_only:
        logger.info("Running serve-only mode (using cached weights)...")
        # Force use of cache and disable evaluation
        args.no_cache = False
        args.force_evaluation = False
        return run_native_backend(args)
    
    # Handle Docker mode
    if args.docker:
        return run_docker_backend()
    
    # Default: run native backend with evaluation if needed
    logger.info("Running backend with automatic evaluation/caching...")
    return run_native_backend(args)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 