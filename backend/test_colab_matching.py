import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === SRM Filters ===
srm_filters = np.array([
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

# === SRM Filter Application - EXACTLY as in Colab ===
def apply_srm_filters_tf(image):
    """Apply SRM filters to image"""
    image = tf.image.resize(image, [256, 256])
    channels = tf.split(image, num_or_size_splits=3, axis=-1)
    srm_filters_tf = tf.constant(np.transpose(srm_filters, (2, 3, 1, 0)), dtype=tf.float32)
    feature_maps = [tf.nn.conv2d(channel, srm_filters_tf, strides=1, padding='SAME') for channel in channels]
    return tf.concat(feature_maps, axis=-1)

# === Preprocess Image for Encoder - EXACTLY as in Colab ===
def preprocess_image_for_encoder(img_path):
    """Preprocess image exactly as in Colab"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    srm_features = apply_srm_filters_tf(img[tf.newaxis, ...])
    return srm_features

def setup_colab_style_classifier(model_dir, example_images):
    """Set up classifier like in Colab notebook"""
    # Load autoencoder and extract encoder
    ae_path = os.path.join(model_dir, 'ae', 'autoencoder_L256.keras')
    classifier_path = os.path.join(model_dir, 'dct_kfold_noEarlyStopping_SGD_model.h5')
    
    logger.info(f"Loading autoencoder from {ae_path}")
    autoencoder = load_model(ae_path)
    
    # Get encoder output layer
    encoder_output = None
    for layer in autoencoder.layers:
        if layer.name == 'encoder_output':
            encoder_output = layer
            break
    
    if encoder_output:
        encoder = Model(inputs=autoencoder.input, outputs=encoder_output.output)
    else:
        # Fallback if layer name not found
        logger.warning("Could not find 'encoder_output' layer, will fail")
        return None, None
    
    logger.info(f"Loading classifier from {classifier_path}")
    classifier = load_model(classifier_path)
    
    # Create scaler with example images
    example_vectors = []
    for img_path in example_images:
        logger.info(f"Processing example image: {img_path}")
        srm_feat = preprocess_image_for_encoder(img_path)
        latent = encoder.predict(srm_feat, verbose=0)
        latent_flat = latent.reshape(1, -1)
        latent_dct = dct(latent_flat, axis=-1, norm='ortho')
        example_vectors.append(latent_dct[0])
    
    scaler = StandardScaler()
    scaler.fit(example_vectors)
    logger.info(f"Scaler fit with {len(example_vectors)} example vectors")
    
    return encoder, classifier, scaler

def classify_image_colab_style(img_path, encoder, classifier, scaler):
    """Classify image exactly like in Colab notebook"""
    logger.info(f"Classifying image: {img_path}")
    
    # Process exactly like in Colab
    srm_feat = preprocess_image_for_encoder(img_path)
    logger.info(f"SRM features shape: {srm_feat.shape}")
    
    latent = encoder.predict(srm_feat, verbose=0)
    logger.info(f"Latent features shape: {latent.shape}")
    
    latent_flat = latent.reshape(1, -1)
    logger.info(f"Flattened latent shape: {latent_flat.shape}")
    
    latent_dct = dct(latent_flat, axis=-1, norm='ortho')
    logger.info(f"DCT features shape: {latent_dct.shape}")
    
    latent_scaled = scaler.transform(latent_dct)
    logger.info(f"Scaled features shape: {latent_scaled.shape}")
    
    # Get raw prediction
    raw_prediction = classifier.predict(latent_scaled, verbose=0)
    logger.info(f"Raw prediction: {raw_prediction}")
    
    # Get class prediction using argmax
    predicted_class_idx = np.argmax(raw_prediction[0])
    class_labels = ['Fake', 'Real']
    prediction = class_labels[predicted_class_idx]
    
    logger.info(f"Predicted class: {prediction} (index {predicted_class_idx})")
    logger.info(f"Prediction probabilities: {raw_prediction}")
    
    return {
        'prediction': prediction,
        'class_idx': predicted_class_idx,
        'probabilities': raw_prediction[0].tolist(),
        'fake_prob': float(raw_prediction[0, 0]),
        'real_prob': float(raw_prediction[0, 1])
    }

def main():
    parser = argparse.ArgumentParser(description='Test DCT model with Colab-style pipeline')
    parser.add_argument('image_path', help='Path to the image to classify')
    parser.add_argument('--model-dir', default='/app/models', 
                        help='Directory containing model files (default: /app/models)')
    parser.add_argument('--example-real', default=None, 
                        help='Example real image for scaler (uses image_path if not specified)')
    parser.add_argument('--example-fake', default=None, 
                        help='Example fake image for scaler (uses image_path if not specified)')
    
    args = parser.parse_args()
    
    # Set up classifier
    logger.info(f"Using model directory: {args.model_dir}")
    
    # If example images aren't provided, use the input image twice
    if not args.example_real or not args.example_fake:
        logger.info("No example images provided, using input image for scaler fitting")
        example_images = [args.image_path, args.image_path]
    else:
        example_images = [args.example_real, args.example_fake]
        logger.info(f"Using example images: {example_images}")
    
    encoder, classifier, scaler = setup_colab_style_classifier(args.model_dir, example_images)
    
    if not encoder or not classifier or not scaler:
        logger.error("Failed to set up classification pipeline")
        return
    
    # Classify image
    result = classify_image_colab_style(args.image_path, encoder, classifier, scaler)
    
    # Print result
    print("\n" + "="*50)
    print(f"COLAB-STYLE CLASSIFICATION RESULT FOR {os.path.basename(args.image_path)}")
    print(f"Prediction: {result['prediction']}")
    print(f"Fake probability: {result['fake_prob']}")
    print(f"Real probability: {result['real_prob']}")
    print("="*50)

if __name__ == "__main__":
    main() 