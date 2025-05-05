#!/usr/bin/env python3
"""
Test script to verify model loading and prediction capabilities.
This script directly loads models and tests their predictions without going through the API.
"""
import os
import sys
import logging
import argparse
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_test")

# Add the current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import model classes
try:
    from models.ensemble_model import EnsembleModel
    from models.srm_cnn_model import SRMCNNModel
    from utils.image_processing import load_and_preprocess_image, apply_srm_filters_tf
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    sys.exit(1)

def test_single_model(model_path, image_path, latent_size=64):
    """Test a single SRM-CNN model."""
    logger.info(f"Testing single model: {model_path} with latent size {latent_size}")
    
    # Create and load model
    model = SRMCNNModel(latent_size=latent_size)
    success = model.load(model_path)
    
    if not success:
        logger.error(f"Failed to load model from {model_path}")
        return
    
    # Make sure image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return
    
    # Log image details
    try:
        with Image.open(image_path) as img:
            logger.info(f"Test image: {image_path}, size: {img.size}, mode: {img.mode}")
    except Exception as e:
        logger.error(f"Failed to open image: {str(e)}")
    
    # Test with direct prediction
    logger.info("Testing direct model prediction...")
    try:
        # Preprocess the image
        preprocessed = model.preprocess(image_path)
        logger.info(f"Preprocessed image shape: {preprocessed.shape}")
        
        # Direct prediction
        pred = model.predict(preprocessed)
        logger.info(f"Raw prediction: {pred}")
        
        # Analyze
        result = model.analyze(image_path)
        logger.info(f"Analysis result: {result}")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")

def test_ensemble(models_dir, image_path):
    """Test the ensemble model with all available models."""
    logger.info(f"Testing ensemble with models from: {models_dir}")
    
    # Create ensemble
    ensemble = EnsembleModel()
    
    # Check models directory
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return
    
    # Find model files
    model_files = [f for f in os.listdir(models_dir) 
                  if f.endswith(('.keras', '.h5')) and os.path.isfile(os.path.join(models_dir, f))]
    
    if not model_files:
        logger.error(f"No model files found in {models_dir}")
        return
    
    logger.info(f"Found model files: {', '.join(model_files)}")
    
    # Load each model and add to ensemble
    for filename in model_files:
        path = os.path.join(models_dir, filename)
        latent_size = 64 if "256" in filename else 32
        
        model = SRMCNNModel(latent_size=latent_size)
        if model.load(path):
            ensemble.add_model(model)
            logger.info(f"Added {model.model_name} to ensemble")
    
    # Check if any models were loaded
    if not ensemble.models:
        logger.error("No models were successfully loaded")
        return
    
    # Test ensemble prediction
    logger.info(f"Testing ensemble with {len(ensemble.models)} models")
    
    # Make sure image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return
    
    try:
        # Analyze with ensemble
        result = ensemble.analyze(image_path)
        logger.info(f"Ensemble result: {result}")
        
        # Check if prediction is neutral
        if 0.49 <= result["probability"] <= 0.51:
            logger.warning("WARNING: Neutral prediction (close to 0.5) detected!")
            logger.warning("This may indicate the model is using fallback values")
        
    except Exception as e:
        logger.error(f"Error during ensemble prediction: {str(e)}")

def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(description="Test deepfake detection models")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--models-dir", default="models", help="Directory containing model files")
    parser.add_argument("--single-model", help="Test a specific model file")
    parser.add_argument("--latent-size", type=int, default=64, help="Latent size for single model test")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DEEPFAKE DETECTION MODEL TEST")
    print("=" * 80)
    
    if args.single_model:
        test_single_model(args.single_model, args.image, args.latent_size)
    else:
        test_ensemble(args.models_dir, args.image)
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 