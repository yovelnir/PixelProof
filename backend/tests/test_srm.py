#!/usr/bin/env python3
"""
Test script to specifically check the SRM filter application and tensor shape handling.
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
logger = logging.getLogger("srm_test")

# Add the current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import utilities
try:
    from utils.image_processing import load_and_preprocess_image, apply_srm_filters
    from models.srm_cnn_model import SRMCNNModel
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    sys.exit(1)

def test_srm_filter(image_path):
    """
    Test the SRM filter application process on an image.
    
    Args:
        image_path (str): Path to the test image
    """
    logger.info(f"Testing SRM filter on image: {image_path}")
    
    # Load and preprocess the image
    try:
        img = load_and_preprocess_image(image_path)
        logger.info(f"Loaded image shape: {img.shape}")
        
        # Apply SRM filters
        filtered_img = apply_srm_filters(img)
        logger.info(f"Filtered image shape: {filtered_img.shape}")
        logger.info(f"Filtered image min: {np.min(filtered_img)}, max: {np.max(filtered_img)}")
        
        # Add batch dimension for model input
        batched_img = np.expand_dims(filtered_img, axis=0)
        logger.info(f"Batched image shape: {batched_img.shape}")
        
        # Success!
        logger.info("SRM filter application successful")
        
        return batched_img
    except Exception as e:
        logger.error(f"Error in SRM filter application: {str(e)}")
        return None

def test_model_prediction(image_path, model_path, latent_size=64):
    """
    Test model prediction on an image.
    
    Args:
        image_path (str): Path to the test image
        model_path (str): Path to the model file
        latent_size (int): Latent size for the model
    """
    logger.info(f"Testing model prediction with latent size {latent_size}")
    logger.info(f"Model path: {model_path}")
    
    # Create model
    model = SRMCNNModel(latent_size=latent_size)
    
    # Load model
    success = model.load(model_path)
    if not success:
        logger.error("Failed to load model")
        return
    
    # Preprocess image
    try:
        # Use model's preprocess method
        preprocessed = model.preprocess(image_path)
        logger.info(f"Preprocessed shape from model: {preprocessed.shape}")
        
        # Make prediction
        prediction = model.predict(preprocessed)
        logger.info(f"Prediction: {prediction}")
        
        # Success!
        return prediction
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        return None

def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(description="Test SRM filter application and model prediction")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--model", help="Path to model file for prediction test")
    parser.add_argument("--latent-size", type=int, default=64, help="Latent size for model")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SRM FILTER AND MODEL PREDICTION TEST")
    print("=" * 80)
    
    # Test SRM filter
    print("\nTesting SRM filter application...")
    srm_result = test_srm_filter(args.image)
    
    if srm_result is not None:
        print("✓ SRM filter test passed")
    else:
        print("✗ SRM filter test failed")
    
    # Test model prediction if model path provided
    if args.model:
        print("\nTesting model prediction...")
        pred_result = test_model_prediction(args.image, args.model, args.latent_size)
        
        if pred_result is not None:
            print(f"✓ Model prediction test passed: {pred_result}")
        else:
            print("✗ Model prediction test failed")
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 