import os
import sys
import numpy as np
import tensorflow as tf
from .base_model import BaseModel
from utils.image_processing import load_and_preprocess_image, extract_dct_features, apply_srm_filters_tf
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DCTModel(BaseModel):
    """DCT-based model for deepfake detection.
    
    This model follows the EXACT training pipeline from create_dct_dataset function:
    
    Training Pipeline (create_dct_dataset):
    1. img = tf.io.read_file(path)
    2. img = tf.image.decode_jpeg(img, channels=3)
    3. img = tf.image.resize(img, [256, 256])
    4. img = tf.cast(img, tf.float32) / 255.0
    5. filtered_img = apply_srm_filters_tf(img[tf.newaxis, ...])[0]
    6. latent = encoder(filtered_img[tf.newaxis, ...], training=False)[0]
    7. latent_flat = tf.reshape(latent, [-1])
    8. dct_features = tf.signal.dct(tf.expand_dims(latent_flat, 0), type=2, norm='ortho')[0]
    9. dct_features = tf.math.l2_normalize(dct_features, axis=0)
    
    Model Architecture (build_dct_classifier):
    - Input: DCT features (normalized)
    - Dense layers with ReLU, Dropout, BatchNorm
    - Output: Dense(1, activation='sigmoid') - probability of being REAL
    - Loss: binary_crossentropy
    - Labels: 0=fake, 1=real
    """
    
    def __init__(self, block_size=8, num_coefficients=64, use_encoder=False, latent_size=256, invert_labels=False):
        """Initialize DCT model for deepfake detection.
        
        Args:
            block_size (int): Size of DCT blocks (typically 8) - kept for compatibility
            num_coefficients (int): Number of DCT coefficients to keep per block - kept for compatibility
            use_encoder (bool): Whether to use the autoencoder encoder
            latent_size (int): Size of the latent representation for the autoencoder
            invert_labels (bool): Whether to invert labels (for models trained with fake=1, real=0)
        """
        model_name = f"DCT-B{block_size}-C{num_coefficients}"
        if use_encoder:
            model_name = f"DCT-AE-L{latent_size}"
            
        super().__init__(model_name=model_name)
        self.block_size = block_size
        self.num_coefficients = num_coefficients
        self.image_size = 256  # Fixed standard size for preprocessing
        self.use_encoder = use_encoder
        self.latent_size = latent_size
        self.model = None
        self.encoder = None
        self.invert_labels = invert_labels  # Flag to handle label inversion
    
    def load(self, model_path):
        """Load the DCT model and optionally load the encoder.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        logger.info(f"Loading DCT model from {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Error: Model file {model_path} does not exist")
            return False
        
        try:
            # Load the classification model
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Model loaded successfully. Input shape: {}, Output shape: {}".format(
                self.model.input_shape, self.model.output_shape))
            
            # Extract expected input size from model to determine correct latent size
            expected_input_size = self.model.input_shape[1]  # Get feature dimension
            
            if self.use_encoder:
                # Determine which autoencoder to use based on expected input size
                if expected_input_size == 8192:
                    # 8192 = 8*8*128, so we need L128 autoencoder
                    autoencoder_path = os.path.join(os.path.dirname(model_path), "ae", "autoencoder_L128.h5")
                    if not os.path.exists(autoencoder_path):
                        autoencoder_path = os.path.join(os.path.dirname(model_path), "ae", "autoencoder_L128.keras")
                    self.latent_size = 128
                elif expected_input_size == 16384:
                    # 16384 = 8*8*256, so we need L256 autoencoder  
                    autoencoder_path = os.path.join(os.path.dirname(model_path), "ae", "autoencoder_L256.h5")
                    if not os.path.exists(autoencoder_path):
                        autoencoder_path = os.path.join(os.path.dirname(model_path), "ae", "autoencoder_L256.keras")
                    self.latent_size = 256
                else:
                    logger.error(f"Unexpected input size {expected_input_size}, cannot determine autoencoder")
                    return False
                
                logger.info(f"Using autoencoder from {autoencoder_path} for latent size {self.latent_size}")
                
                if not os.path.exists(autoencoder_path):
                    logger.error(f"Autoencoder file not found: {autoencoder_path}")
                    return False
                
                # Load autoencoder and extract encoder
                autoencoder = tf.keras.models.load_model(autoencoder_path, compile=False)
                
                # Create encoder model exactly as in training code
                try:
                    # Extract encoder exactly as in training: Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder_output').output)
                    self.encoder = tf.keras.models.Model(
                        inputs=autoencoder.input, 
                        outputs=autoencoder.get_layer('encoder_output').output
                    )
                    self.encoder.trainable = False  # Match training code: encoder.trainable = False
                except Exception as e:
                    logger.error(f"Failed to extract encoder using 'encoder_output' layer: {str(e)}")
                    return False
                
                logger.info(f"Created encoder model with output shape: {self.encoder.output_shape}")
                logger.info(f"Encoder input shape: {self.encoder.input_shape}, output shape: {self.encoder.output_shape}")
                
                # Update model name with correct latent size
                self.model_name = f"DCT-AE-L{self.latent_size} ({os.path.basename(model_path)})"
                
                return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess(self, image_path):
        """Preprocess image using the EXACT same pipeline as the training code.
        
        This follows the exact sequence from create_dct_dataset function:
        1. Load image with tf.io.read_file
        2. Decode JPEG with 3 channels
        3. Resize to [256, 256]
        4. Cast to float32 and normalize to [0,1]
        5. Apply SRM filters to get 15 channels
        6. Pass through encoder
        7. Flatten latent representation
        8. Apply DCT with type=2, norm='ortho'
        9. Apply L2 normalization
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            np.ndarray: Preprocessed DCT features ready for classification
        """
        if self.use_encoder:
            if self.encoder is None:
                logger.error("Encoder model not loaded!")
                raise ValueError("Encoder model not loaded")
            
            # Step 1-4: Load and preprocess image using image_processing utility
            img = load_and_preprocess_image(image_path, target_size=(256, 256))
            
            # Step 5: Apply SRM filters to get 15 channels using TensorFlow function directly (matches training)
            filtered_img = apply_srm_filters_tf(img)  # shape: (256, 256, 15)
            
            # Step 6: Pass SRM-filtered image into encoder
            latent = self.encoder(filtered_img[tf.newaxis, ...], training=False)[0]  # shape: (8, 8, latent_size)
            
            # Step 7: Flatten latent representation exactly as in training
            latent_flat = tf.reshape(latent, [-1])  # shape: (flattened_size,)
            
            # Step 8: Apply DCT exactly as in training code
            dct_features = tf.signal.dct(tf.expand_dims(latent_flat, 0), type=2, norm='ortho')[0]
            
            # Step 9: Apply L2 normalization exactly as in training code
            dct_features = tf.math.l2_normalize(dct_features, axis=0)
            
            # Convert to numpy and add batch dimension for model prediction
            features_batch = np.expand_dims(dct_features.numpy(), axis=0)
            
            return features_batch
        else:
            # For non-encoder models, use original DCT-based pipeline
            img = load_and_preprocess_image(image_path, target_size=(self.image_size, self.image_size))
            dct_features = extract_dct_features(
                img, 
                block_size=self.block_size,
                num_coefficients=self.num_coefficients
            )
            
            # Add batch dimension for model compatibility
            features_batch = np.expand_dims(dct_features, axis=0)
            
            return features_batch
    
    def predict(self, processed_features):
        """Run inference on preprocessed features.
        
        Args:
            processed_features: Preprocessed DCT features
            
        Returns:
            np.ndarray: Raw prediction output
            
        Raises:
            ValueError: If prediction fails or model is not loaded
        """
        if self.model is None:
            logger.error("Error: Model not loaded!")
            raise ValueError("Model not loaded")
        
        try:
            # logger.info(f"Running prediction with model {self.model_name}")
            
            # Get expected input shape from model
            expected_shape = self.model.input_shape
            actual_shape = processed_features.shape
            
            # logger.info(f"Model expects shape: {expected_shape}, got: {actual_shape}")
            
            # Handle shape mismatches if needed
            if expected_shape[1:] != actual_shape[1:]:
                logger.warning(f"Shape mismatch. Model expects: {expected_shape}, got: {actual_shape}")
                # Try to reshape if possible
                try:
                    processed_features = processed_features.reshape((-1,) + expected_shape[1:])
                    # logger.info(f"Reshaped to: {processed_features.shape}")
                except Exception as reshape_error:
                    logger.error(f"Failed to reshape: {str(reshape_error)}")
                    raise ValueError(f"Input shape mismatch: {actual_shape} vs expected {expected_shape}")
            
            # Make prediction
            raw_prediction = self.model.predict(processed_features, verbose=0)
            # logger.info(f"Raw model prediction: {raw_prediction}")
            
            return raw_prediction
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def analyze(self, image_path):
        """Analyze an image for deepfake detection using the exact training pipeline.
        
        The trained model uses:
        - binary_crossentropy loss
        - Single sigmoid output neuron
        - Labels: 0 = fake, 1 = real
        - Output: probability of being REAL
        
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
        
        if self.use_encoder and self.encoder is None:
            logger.error(f"Model {self.model_name} requires encoder but none is loaded!")
            raise ValueError("Encoder not loaded")
        
        try:
            # Preprocess image using exact training pipeline
            preprocessed_features = self.preprocess(image_path)
            
            # Make prediction
            raw_prediction = self.predict(preprocessed_features)
            
            # Process prediction results - model trained with single sigmoid output
            if isinstance(raw_prediction, np.ndarray):
                # The training code uses single sigmoid output: Dense(1, activation='sigmoid')
                # This outputs the probability of being REAL (label 1) for standard models
                raw_prob = float(raw_prediction.flatten()[0])
                
                # Handle label inversion if needed
                if self.invert_labels:
                    # If model was trained with inverted labels (fake=1, real=0)
                    # then raw output is probability of being FAKE
                    fake_prob = raw_prob
                    real_prob = 1.0 - raw_prob
                else:
                    # Standard convention: raw output is probability of being REAL
                    real_prob = raw_prob
                    fake_prob = 1.0 - raw_prob
                
                # Determine prediction based on 0.5 threshold
                prediction = "real" if real_prob > 0.15 else "fake"
                
                # Calculate confidence (distance from 0.5 threshold, scaled to [0,1])
                confidence = abs(real_prob - 0.5) * 2
                
            else:
                logger.error(f"Unexpected prediction format: {raw_prediction}")
                raise ValueError(f"Unexpected prediction format: {raw_prediction}")
            
            result = {
                "probability": float(fake_prob),  # Probability of being fake (for consistency with API)
                "prediction": prediction,
                "confidence": float(confidence)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during DCT analysis: {str(e)}")
            raise ValueError(f"Analysis failed: {str(e)}") 