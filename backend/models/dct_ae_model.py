import os
import sys
import numpy as np
import tensorflow as tf
from .base_model import BaseModel
from utils.image_processing import load_and_preprocess_image, apply_srm_filters
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DCTAutoencoderModel(BaseModel):
    """DCT model with autoencoder preprocessing for deepfake detection.
    
    This model:
    1. Applies SRM filters to the input image
    2. Passes filtered image through an autoencoder to get a latent representation
    3. Applies DCT to the latent representation
    4. Uses the transformed features for classification
    """
    
    def __init__(self, latent_size=256):
        """Initialize DCT+Autoencoder model for deepfake detection.
        
        Args:
            latent_size (int): Size of the latent representation from the autoencoder
        """
        super().__init__(model_name=f"DCT-AE-L{latent_size}")
        self.latent_size = latent_size
        self.image_size = 256  # Fixed standard size for preprocessing
        self.model = None      # Classification model
        self.encoder = None    # Encoder part of autoencoder
        
    def load(self, model_path):
        """Load the DCT classifier model and autoencoder.
        
        Args:
            model_path (str): Path to the DCT model file
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Determine autoencoder path based on latent size
        models_dir = os.path.dirname(model_path)
        ae_dir = os.path.join(models_dir, 'ae')
        ae_path = os.path.join(ae_dir, f"autoencoder_L{self.latent_size}.keras")
        
        if not os.path.exists(ae_path):
            logger.error(f"Autoencoder model not found: {ae_path}")
            return False
            
        try:
            # Load the DCT classifier model
            logger.info(f"Loading DCT model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Log model summary
            self.model.summary(print_fn=logger.info)
            
            # Load the autoencoder model
            logger.info(f"Loading autoencoder from {ae_path}")
            autoencoder = tf.keras.models.load_model(ae_path)
            
            # Create encoder model by extracting the encoder_output layer
            # The encoder's output is the layer named 'encoder_output' (as defined in build_autoencoder)
            encoder_output = None
            for layer in autoencoder.layers:
                if layer.name == 'encoder_output':
                    encoder_output = layer
                    break
            
            if encoder_output is not None:
                # Create a new model with just the encoder part
                self.encoder = tf.keras.models.Model(
                    inputs=autoencoder.input,
                    outputs=encoder_output.output
                )
                logger.info(f"Created encoder model with output shape: {self.encoder.output_shape}")
            else:
                # Fallback method if the expected layer name isn't found
                logger.warning("Could not find 'encoder_output' layer, trying alternative extraction method")
                # Extract encoder layers based on output dimension
                encoder_layers = []
                encoder_found = False
                
                for layer in autoencoder.layers:
                    encoder_layers.append(layer)
                    # Stop when we find the layer whose output shape matches the latent dimension
                    if hasattr(layer, 'output_shape') and layer.output_shape:
                        # For 3D outputs (Conv/Pool layers), we need to check the last dimension
                        if len(layer.output_shape) == 4 and layer.output_shape[-1] == self.latent_size:
                            logger.info(f"Found potential encoder bottleneck: {layer.name} with shape {layer.output_shape}")
                            encoder_found = True
                            break
                
                if encoder_found:
                    self.encoder = tf.keras.models.Model(
                        inputs=autoencoder.input,
                        outputs=encoder_layers[-1].output
                    )
                    logger.info(f"Created encoder model with output shape: {self.encoder.output_shape}")
                else:
                    logger.error("Failed to extract encoder part from autoencoder")
                    return False
            
            # Display input and output shapes
            logger.info(f"Models loaded successfully.")
            logger.info(f"Encoder input shape: {self.encoder.input_shape}, output shape: {self.encoder.output_shape}")
            logger.info(f"DCT model input shape: {self.model.input_shape}, output shape: {self.model.output_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def preprocess(self, image_path):
        """Preprocess image through SRM filters, autoencoder, and DCT.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            np.ndarray: Preprocessed features ready for classification
        """
        logger.info(f"Preprocessing image with SRM+Autoencoder+DCT: {image_path}")
        
        # 1. Load and preprocess image to standard size
        img = load_and_preprocess_image(image_path, target_size=(self.image_size, self.image_size))
        
        # 2. Apply SRM filters
        filtered_img = apply_srm_filters(img)
        filtered_img_batch = np.expand_dims(filtered_img, axis=0)
        
        # 3. Apply encoder to get latent representation
        if self.encoder is None:
            logger.error("Encoder model not loaded!")
            raise ValueError("Encoder model not loaded")
        
        encoded_features = self.encoder.predict(filtered_img_batch)
        logger.info(f"Raw encoded features shape: {encoded_features.shape}")
        
        # 4. Flatten the encoded features if they're not already flat
        # The encoder output is likely 8x8x256 based on the architecture
        if len(encoded_features.shape) > 2:
            # Flatten all dimensions except the batch dimension
            latent_features = encoded_features.reshape(encoded_features.shape[0], -1)
            logger.info(f"Flattened latent features shape: {latent_features.shape}")
        else:
            latent_features = encoded_features
        
        # 5. Apply DCT to latent features
        # Use scipy's DCT for 1D data
        from scipy.fftpack import dct
        latent_dct = dct(latent_features, axis=1, norm='ortho')
        logger.info(f"DCT features shape: {latent_dct.shape}")
        
        # 6. Standardize features (similar to the evaluation script)
        # Note: In production, you'd use a pre-fit scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(latent_dct)
        
        logger.info(f"Final preprocessed features shape: {scaled_features.shape}")
        return scaled_features
    
    def predict(self, processed_features):
        """Run inference on preprocessed features.
        
        Args:
            processed_features: Preprocessed features
            
        Returns:
            float: Probability of image being REAL (0-1), where 1 = real, 0 = fake
            
        Raises:
            ValueError: If prediction fails or model is not loaded
        """
        if self.model is None:
            logger.error("Error: Classification model not loaded!")
            raise ValueError("Classification model not loaded")
        
        try:
            logger.info(f"Running prediction with model {self.model_name}")
            
            # Get expected input shape from model
            expected_shape = self.model.input_shape
            actual_shape = processed_features.shape
            
            logger.info(f"Model expects shape: {expected_shape}, got: {actual_shape}")
            
            # Handle shape mismatches if needed
            if expected_shape[1:] != actual_shape[1:]:
                logger.warning(f"Shape mismatch. Model expects: {expected_shape}, got: {actual_shape}")
                # Try to reshape if possible
                try:
                    processed_features = processed_features.reshape((-1,) + expected_shape[1:])
                    logger.info(f"Reshaped to: {processed_features.shape}")
                except Exception as reshape_error:
                    logger.error(f"Failed to reshape: {str(reshape_error)}")
                    raise ValueError(f"Input shape mismatch: {actual_shape} vs expected {expected_shape}")
            
            # Make prediction
            prediction = self.model.predict(processed_features)
            
            # Process prediction result
            if isinstance(prediction, np.ndarray) and prediction.size > 0:
                if prediction.shape[1] == 1:
                    # Binary classification with single output neuron
                    # NOTE: In our convention, 1 = real, 0 = fake
                    pred_value = float(prediction.flatten()[0])
                else:
                    # Multi-class with softmax
                    # NOTE: In our convention, index 1 = real, index 0 = fake
                    pred_value = float(prediction[0, 1])
                
                logger.info(f"Raw prediction value: {pred_value} (1 = real, 0 = fake)")
                
                # Ensure prediction is within [0,1] range
                if pred_value < 0 or pred_value > 1:
                    logger.warning(f"Prediction out of range [0,1]: {pred_value}, clamping to range")
                    pred_value = max(0, min(1, pred_value))
                    
                return pred_value
            else:
                logger.error(f"Unexpected prediction format: {prediction}")
                raise ValueError(f"Unexpected prediction format: {prediction}")
                
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
        if self.model is None or self.encoder is None:
            logger.error(f"Model {self.model_name} is not fully loaded!")
            raise ValueError("Model not fully loaded")
        
        try:
            logger.info(f"Analyzing image with {self.model_name}: {image_path}")
            
            # Process image through the full pipeline
            preprocessed_features = self.preprocess(image_path)
            
            # Make prediction
            prediction = self.predict(preprocessed_features)
            
            # Calculate confidence (distance from 0.5 threshold, scaled to [0,1])
            confidence = abs(float(prediction - 0.5)) * 2
            
            # Determine if the image is classified as real
            # NOTE: In our convention, 1 = real, 0 = fake
            is_real = prediction > 0.5
            
            # Prepare result - adjust to match rest of system's output format
            # where "probability" is likelihood of being fake
            fake_probability = 1.0 - prediction
            
            result = {
                "probability": float(fake_probability),  # Probability of being fake
                "prediction": "real" if is_real else "fake",
                "confidence": confidence
            }
            
            logger.info(f"DCT+Autoencoder analysis result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during DCT+Autoencoder analysis: {str(e)}")
            raise ValueError(f"Analysis failed: {str(e)}") 