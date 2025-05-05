import os
import sys
import numpy as np
import tensorflow as tf
from .base_model import BaseModel
from utils.image_processing import load_and_preprocess_image, extract_dct_features, apply_srm_filters
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define SRM filters exactly as in Colab for consistency
SRM_FILTERS = np.array([
    [[[0, 0, -1, 0, 0],
      [0, -1, 2, -1, 0],
      [-1, 2, 4, 2, -1],
      [0, -1, 2, -1, 0],
      [0, 0, -1, 0, 0]]],
    [[[-1, 2, -2, 2, -1],
      [2, -6, 8, -6, 2],
      [-2, 8, -12, 8, -2],
      [2, -6, 8, -6, 2],
      [-1, 2, -2, 2, -1]]],
    [[[2, -1, 0, -1, 2],
      [-1, -2, 3, -2, -1],
      [0, 3, 0, 3, 0],
      [-1, -2, 3, -2, -1],
      [2, -1, 0, -1, 2]]],
    [[[0, 0, 0, 0, 0],
      [1, -2, 1, -2, 1],
      [0, 0, 0, 0, 0],
      [-1, 2, -1, 2, -1],
      [0, 0, 0, 0, 0]]],
    [[[1, -4, 6, -4, 1],
      [-4, 16, -24, 16, -4],
      [6, -24, 36, -24, 6],
      [-4, 16, -24, 16, -4],
      [1, -4, 6, -4, 1]]],
], dtype=np.float32)

# Colab-style SRM filter application
def apply_srm_filters_colab_style(image):
    """Apply SRM filters exactly as done in Colab"""
    image = tf.image.resize(image, [256, 256])
    channels = tf.split(image, num_or_size_splits=3, axis=-1)
    srm_filters_tf = tf.constant(np.transpose(SRM_FILTERS, (2, 3, 1, 0)), dtype=tf.float32)
    feature_maps = [tf.nn.conv2d(channel, srm_filters_tf, strides=1, padding='SAME') for channel in channels]
    return tf.concat(feature_maps, axis=-1)

class DCTModel(BaseModel):
    """DCT-based model for deepfake detection.
    
    This model applies DCT (Discrete Cosine Transform) block processing to images
    and extracts frequency domain features for analysis.
    
    NOTE: This version can optionally use an autoencoder to extract latent features
    before applying DCT.
    """
    
    def __init__(self, block_size=8, num_coefficients=64, use_encoder=False, latent_size=256):
        """Initialize DCT model for deepfake detection.
        
        Args:
            block_size (int): Size of DCT blocks (typically 8)
            num_coefficients (int): Number of DCT coefficients to keep per block
            use_encoder (bool): Whether to use the autoencoder encoder
            latent_size (int): Size of the latent representation for the autoencoder
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
        
        # Track known real and fake examples for comparison
        self.example_vectors = []
        self.scaler = None
        self.scaler_fitted = False
    
    def load(self, model_path):
        """Load the model from a file.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
        try:
            logger.info(f"Loading DCT model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Log model summary
            self.model.summary(print_fn=logger.info)
            
            # Display input and output shapes
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
            
            logger.info(f"Model loaded successfully. Input shape: {input_shape}, Output shape: {output_shape}")
            
            # If we're using the encoder, load it
            if self.use_encoder:
                # Determine autoencoder path based on latent size
                models_dir = os.path.dirname(model_path)
                ae_dir = os.path.join(models_dir, 'ae')
                ae_path = os.path.join(ae_dir, f"autoencoder_L{self.latent_size}.keras")
                
                if not os.path.exists(ae_path):
                    logger.error(f"Autoencoder model not found: {ae_path}")
                    return False
                
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
                
                logger.info(f"Encoder input shape: {self.encoder.input_shape}, output shape: {self.encoder.output_shape}")
                
                # Initialize StandardScaler for feature normalization
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.example_vectors = []
                self.scaler_fitted = False
                
                # Get test images for scaler initialization - use images from test_images folder
                project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(model_path)), '..'))
                test_images_dir = os.path.join(project_root, 'backend', 'test_images')
                
                if os.path.exists(test_images_dir):
                    logger.info(f"Found test_images directory at {test_images_dir}")
                    # Get all image files from the test_images directory
                    test_image_files = []
                    for filename in os.listdir(test_images_dir):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            test_image_files.append(os.path.join(test_images_dir, filename))
                    
                    if test_image_files:
                        logger.info(f"Found {len(test_image_files)} test images for scaler initialization")
                        # Use at most 5 images for initialization (to keep it fast)
                        for i, img_path in enumerate(test_image_files[:5]):
                            logger.info(f"Processing test image {i+1}/{min(5, len(test_image_files))}: {os.path.basename(img_path)}")
                            try:
                                # Extract features from test image
                                srm_features, latent, latent_flat, latent_dct = self._extract_features(img_path)
                                self.example_vectors.append(latent_dct[0])
                                logger.info(f"Added example vector #{len(self.example_vectors)} from {os.path.basename(img_path)}")
                            except Exception as e:
                                logger.warning(f"Error processing test image {img_path}: {str(e)}")
                        
                        # Fit the scaler with the collected example vectors
                        if len(self.example_vectors) >= 2:
                            self.scaler.fit(self.example_vectors)
                            self.scaler_fitted = True
                            logger.info(f"Successfully fitted scaler with {len(self.example_vectors)} test images")
                        else:
                            logger.warning("Not enough valid test images for scaler initialization")
                    else:
                        logger.warning("No image files found in test_images directory")
                else:
                    logger.warning(f"Test images directory not found at {test_images_dir}")
                    logger.warning("Will initialize scaler with first analyzed images")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _extract_features(self, image_path):
        """Extract features from an image in exactly the same way as Colab.
        
        Args:
            image_path (str): Path to image
            
        Returns:
            tuple: (srm_features, latent, latent_flat, latent_dct)
        """
        # Load image using TensorFlow API exactly as in Colab
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
        
        # Apply SRM filters (TensorFlow-based)
        img_batch = img[tf.newaxis, ...]  # Add batch dimension
        srm_features = apply_srm_filters_colab_style(img_batch)
        
        # Apply encoder to get latent representation
        latent = self.encoder.predict(srm_features, verbose=0)
        
        # Flatten the latent features
        latent_flat = latent.reshape(latent.shape[0], -1)
        
        # Apply DCT
        from scipy.fftpack import dct
        latent_dct = dct(latent_flat, axis=1, norm='ortho')
        
        return srm_features, latent, latent_flat, latent_dct
    
    def preprocess_colab_style(self, image_path):
        """Preprocess image using the exact same pipeline as in Colab.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Preprocessed features and intermediate representations
        """
        logger.info(f"Preprocessing image Colab-style: {image_path}")
        
        # Extract features using the same process as in Colab
        srm_features, latent, latent_flat, latent_dct = self._extract_features(image_path)
        
        logger.info(f"SRM features shape: {srm_features.shape}")
        logger.info(f"Latent features shape: {latent.shape}")
        logger.info(f"Flattened latent shape: {latent_flat.shape}")
        logger.info(f"DCT features shape: {latent_dct.shape}")
        
        # Add to example vectors if we don't have many yet
        if len(self.example_vectors) < 10:  # Limit how many we collect
            self.example_vectors.append(latent_dct[0])
            logger.info(f"Added example vector #{len(self.example_vectors)}")
        
        # Handle scaler based on available example vectors
        if not self.scaler_fitted and len(self.example_vectors) >= 2:
            # We have enough examples now, fit the scaler
            self.scaler.fit(self.example_vectors)
            self.scaler_fitted = True
            logger.info(f"Fitted scaler with {len(self.example_vectors)} examples")
            scaled_features = self.scaler.transform(latent_dct)
            logger.info(f"Scaled features using fitted scaler")
        elif self.scaler_fitted:
            # Use the already fitted scaler
            scaled_features = self.scaler.transform(latent_dct)
            logger.info(f"Scaled features using existing scaler")
        else:
            # Not enough examples yet, use a temporary scaler
            logger.warning(f"No scaler fitted yet and only {len(self.example_vectors)} example(s) available")
            logger.info("Using synthetic sample approach for first-time scaling")
            from sklearn.preprocessing import StandardScaler
            temp_scaler = StandardScaler()
            
            # Create a slightly modified copy of the sample for proper statistics
            noise_factor = 1e-5
            noise = np.random.normal(0, noise_factor, latent_dct[0].shape)
            synthetic_sample = latent_dct[0] + noise
            
            # Stack the original and synthetic sample
            fit_samples = np.vstack([latent_dct[0], synthetic_sample])
            temp_scaler.fit(fit_samples)
            
            # Apply the scaler to the original sample
            scaled_features = temp_scaler.transform(latent_dct)
            logger.info("Used synthetic sample scaling as fallback")
        
        logger.info(f"Scaled features shape: {scaled_features.shape}")
        
        return {
            'srm_features': srm_features,
            'latent': latent,
            'latent_flat': latent_flat,
            'latent_dct': latent_dct,
            'scaled_features': scaled_features
        }
    
    def preprocess(self, image_path):
        """Preprocess image for model input.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            np.ndarray: Preprocessed image features ready for model input
        """
        logger.info(f"Preprocessing image with DCT: {image_path}")
        
        if self.use_encoder:
            # Use Colab-style preprocessing for encoder-based models
            preprocessed = self.preprocess_colab_style(image_path)
            return preprocessed['scaled_features']
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
            logger.info(f"DCT features shape: {features_batch.shape}")
            
            return features_batch
    
    def predict(self, processed_features):
        """Run inference on preprocessed features.
        
        Args:
            processed_features: Preprocessed DCT features
            
        Returns:
            dict: Raw prediction output with probabilities for each class
            
        Raises:
            ValueError: If prediction fails or model is not loaded
        """
        if self.model is None:
            logger.error("Error: Model not loaded!")
            raise ValueError("Model not loaded")
        
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
            
            # Make prediction with verbose=0 to match Colab
            raw_prediction = self.model.predict(processed_features, verbose=0)
            logger.info(f"Raw model prediction: {raw_prediction}")
            
            return raw_prediction
                
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
        
        if self.use_encoder and self.encoder is None:
            logger.error(f"Model {self.model_name} requires encoder but none is loaded!")
            raise ValueError("Encoder not loaded")
        
        try:
            logger.info(f"Analyzing image with {self.model_name}: {image_path}")
            
            # Preprocess image to extract DCT features
            preprocessed_features = self.preprocess(image_path)
            
            # Make prediction
            raw_prediction = self.predict(preprocessed_features)
            
            # Process prediction results differently based on output shape
            if isinstance(raw_prediction, np.ndarray):
                if len(raw_prediction.shape) > 1 and raw_prediction.shape[1] > 1:
                    # Multi-class output from model (softmax)
                    # CRITICAL FIX: Based on example Colab code, the DCT models use:
                    # Class labels = ['Fake', 'Real'] (index 0 = Fake, index 1 = Real)
                    fake_prob = float(raw_prediction[0, 0])
                    real_prob = float(raw_prediction[0, 1])
                    logger.info(f"Multi-class output: fake prob (index 0): {fake_prob}, real prob (index 1): {real_prob}")
                    
                    # Get predicted class using argmax - exactly like Colab example
                    predicted_class_idx = np.argmax(raw_prediction[0])
                    class_labels = ['fake', 'real']  # lowercase to match our system's prediction format
                    prediction = class_labels[predicted_class_idx]
                    is_real = (predicted_class_idx == 1)  # 1 = Real
                    logger.info(f"Predicted class: index {predicted_class_idx} = {prediction}")
                    
                    # For consistency with ensemble, return fake probability
                    probability = fake_prob
                else:
                    # Single output neuron - interpret as Fake probability directly
                    fake_prob = float(raw_prediction.flatten()[0])
                    logger.info(f"Single output neuron: {fake_prob} (interpreted as fake prob)")
                    
                    is_real = fake_prob < 0.5  # Less than 0.5 means real
                    prediction = "real" if is_real else "fake"
                    probability = fake_prob
            else:
                logger.error(f"Unexpected prediction format: {raw_prediction}")
                raise ValueError(f"Unexpected prediction format: {raw_prediction}")
            
            # Calculate confidence (distance from decision boundary)
            if len(raw_prediction.shape) > 1 and raw_prediction.shape[1] > 1:
                # For multi-class, confidence is the difference between the two class probabilities
                confidence = abs(real_prob - fake_prob)
            else:
                # For single output, confidence is the distance from 0.5 threshold, scaled to [0,1]
                confidence = abs(fake_prob - 0.5) * 2
            
            result = {
                "probability": float(probability),  # Probability of being fake
                "prediction": prediction,
                "confidence": float(confidence)
            }
            
            logger.info(f"DCT analysis result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during DCT analysis: {str(e)}")
            raise ValueError(f"Analysis failed: {str(e)}") 