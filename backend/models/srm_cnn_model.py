import os
import sys
import numpy as np
import tensorflow as tf
from .base_model import BaseModel
from utils.image_processing import load_and_preprocess_image, apply_srm_filters
import logging

# Set recursion limit higher
sys.setrecursionlimit(3000)

# Configure logging
logger = logging.getLogger(__name__)

class SRMCNNModel(BaseModel):
    """SRM-CNN model for deepfake detection.
    
    This model applies SRM filters to images and feeds them through a CNN.
    All images are preprocessed to 256x256 regardless of internal representation.
    """
    
    def __init__(self, latent_size=64, invert_labels=False):
        """Initialize SRM-CNN model for deepfake detection.
        
        Args:
            latent_size (int): Size of the latent representation, not input size
            invert_labels (bool): Whether to invert labels (for models trained with fake=1, real=0)
        """
        super().__init__(model_name=f"SRM-CNN-L{latent_size}")
        self.latent_size = latent_size
        # All images are processed at 256x256 regardless of latent size
        self.image_size = 256
        self.model = None
        self.invert_labels = invert_labels  # Flag to handle label inversion
    
    def load(self, model_path):
        """Load a trained model from file.
        
        Args:
            model_path (str): Path to the saved model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        logger.info(f"Loading {self.model_name} from {model_path}...")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Error: Model file {model_path} does not exist")
            return False
        
        try:
            # Simple approach with compile=False to avoid custom optimizer issues
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model {self.model_name}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess(self, image_path):
        """Preprocess image for model input.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            np.ndarray: Preprocessed image ready for model input
        """
        # logger.info(f"Preprocessing image {image_path}")
        
        # Always preprocess to 256x256 regardless of latent size
        img = load_and_preprocess_image(image_path, target_size=(self.image_size, self.image_size))
        
        # Apply SRM filters to get noise features - this should return shape (256, 256, 15) without batch dimension
        filtered_img = apply_srm_filters(img)
        
        # Log shape and value range to verify proper preprocessing
        # logger.info(f"Filtered image shape: {filtered_img.shape}, min: {np.min(filtered_img)}, max: {np.max(filtered_img)}")
        
        # Explicitly add the batch dimension to match model expectations (None, 256, 256, 15)
        # 'None' in the model shape means any batch size, so we use 1
        filtered_img_batch = np.expand_dims(filtered_img, axis=0)
        
        # logger.info(f"Final preprocessed tensor shape: {filtered_img_batch.shape}")
        
        return filtered_img_batch
    
    def predict(self, processed_image):
        """Run inference on preprocessed image.
        
        Args:
            processed_image: Preprocessed image tensor
            
        Returns:
            float: Probability of image being REAL (0-1), where 1=Real and 0=Fake
            
        Raises:
            ValueError: If prediction fails or model is not loaded
        """
        if self.model is None:
            logger.error("Error: Model not loaded!")
            raise ValueError("Model not loaded")
        
        # Make prediction
        try:
            # logger.info(f"Running prediction with model {self.model_name}")
            # Ensure input shape matches model expectations
            expected_shape = self.model.input_shape
            actual_shape = processed_image.shape
            
            # logger.info(f"Model expects shape: {expected_shape}, got: {actual_shape}")
            
            # Check if shapes match except for batch dimension
            if len(expected_shape) == 4 and len(actual_shape) == 4:
                if expected_shape[1:] == actual_shape[1:]:
                    # Shapes match except for possibly batch dimension
                    # logger.info("Shape dimensions match except for batch size")
                    pass
                else:
                    # Try to reshape to match expected format
                    logger.warning(f"Shape mismatch, attempting to reshape: {actual_shape} to match {expected_shape}")
                    processed_image = processed_image.reshape((-1, expected_shape[1], expected_shape[2], expected_shape[3]))
                    # logger.info(f"Reshaped to: {processed_image.shape}")
            
            # The model is designed to accept 'None' as batch size, which means it can handle any batch size
            # When TensorFlow shows (None, 256, 256, 15), it means the model accepts any number of samples
            # Our input is (1, 256, 256, 15) which should be compatible
            
            # Use with batch_size=1 explicitly
            prediction = self.model(processed_image, training=False).numpy()
            
            # Log raw prediction value
            # logger.info(f"Raw prediction: {prediction}")
            
            # Ensure we're getting a value between 0 and 1
            if isinstance(prediction, np.ndarray) and prediction.size > 0:
                pred_value = float(prediction.flatten()[0])
            else:
                logger.error(f"Unexpected prediction format: {prediction}")
                raise ValueError(f"Unexpected prediction format: {prediction}")
                
            # logger.info(f"Prediction value: {pred_value} (1=Real, 0=Fake)")
            
            # Handle out-of-range predictions
            if pred_value < 0 or pred_value > 1:
                logger.warning(f"Prediction out of range [0,1]: {pred_value}, clamping to range")
                pred_value = max(0, min(1, pred_value))
                
            return pred_value
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def analyze(self, image_path):
        """Analyze an image for deepfake detection.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Analysis results with probabilities
            
        Raises:
            ValueError: If analysis fails for any reason
        """
        if self.model is None:
            logger.error(f"Model {self.model_name} is not loaded!")
            raise ValueError("Model not loaded")
        
        try:
            # logger.info(f"Analyzing image with {self.model_name}: {image_path}")
            
            # Preprocess image using standardized 256x256 size
            preprocessed_image = self.preprocess(image_path)
            
            # Make prediction with extra error handling
            try:
                # Don't assume the output shape - use the more flexible predict method
                real_prob = self.predict(preprocessed_image)
                
                # Log the prediction
                # logger.info(f"{self.model_name} prediction: {real_prob}")
            except Exception as pred_error:
                logger.error(f"Error during prediction: {str(pred_error)}")
                # Do not return default values - propagate the error
                raise ValueError(f"Prediction error: {str(pred_error)}")
            
            # Handle label inversion if needed
            if self.invert_labels:
                # If model was trained with inverted labels (fake=1, real=0)
                # then raw output is probability of being FAKE
                fake_probability = real_prob
                real_probability = 1.0 - real_prob
            else:
                # Standard convention: raw output is probability of being REAL
                real_probability = real_prob
                fake_probability = 1.0 - real_prob
            
            # Determine prediction based on 0.5 threshold
            is_real = real_probability > 0.25
            prediction = "real" if is_real else "fake"
            
            # Calculate confidence (distance from 0.5 threshold, scaled to [0,1])
            confidence = abs(real_probability - 0.5) * 2
            
            result = {
                "probability": float(fake_probability),  # Probability of being fake (for consistency with API)
                "prediction": prediction,
                "confidence": float(confidence)
            }
            
            # logger.info(f"Analysis result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            # Do not return default values - propagate the error
            raise ValueError(f"Analysis failed: {str(e)}") 