import numpy as np
import tensorflow as tf
from PIL import Image
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define SRM filters (5 filters from the training code)
SRM_FILTERS = np.array([
    # Filter 1: Laplacian-High Boost
    [[[0, 0, -1, 0, 0],
      [0, -1, 2, -1, 0],
      [-1, 2, 4, 2, -1],
      [0, -1, 2, -1, 0],
      [0, 0, -1, 0, 0]]],

    # Filter 2: Edge & Noise Enhancer
    [[[-1, 2, -2, 2, -1],
      [2, -6, 8, -6, 2],
      [-2, 8, -12, 8, -2],
      [2, -6, 8, -6, 2],
      [-1, 2, -2, 2, -1]]],

    # Filter 3: Diagonal Residual Capture
    [[[2, -1, 0, -1, 2],
      [-1, -2, 3, -2, -1],
      [0, 3, 0, 3, 0],
      [-1, -2, 3, -2, -1],
      [2, -1, 0, -1, 2]]],

    # Filter 4: Vertical Edge Residuals
    [[[0, 0, 0, 0, 0],
      [1, -2, 1, -2, 1],
      [0, 0, 0, 0, 0],
      [-1, 2, -1, 2, -1],
      [0, 0, 0, 0, 0]]],

    # Filter 5: High Frequency Noise Extractor
    [[[1, -4, 6, -4, 1],
      [-4, 16, -24, 16, -4],
      [6, -24, 36, -24, 6],
      [-4, 16, -24, 16, -4],
      [1, -4, 6, -4, 1]]],
], dtype=np.float32)

# Convert SRM filters to TensorFlow format
SRM_FILTERS_TF = tf.constant(np.transpose(SRM_FILTERS, (2, 3, 1, 0)), dtype=tf.float32)  # (5, 5, 1, 5)

def apply_srm_filters_tf(image):
    """Apply SRM filters using TensorFlow.
    
    Args:
        image: Tensor of shape [height, width, 3] or numpy array
        
    Returns:
        numpy.ndarray of shape [height, width, 15] with SRM filter responses
    """
    # Log input shape
    input_shape = None
    if isinstance(image, np.ndarray):
        input_shape = image.shape
    else:
        input_shape = tf.shape(image).numpy()
    
    logger.info(f"SRM filter input shape: {input_shape}")
    
    # Convert to tensor if numpy array
    if isinstance(image, np.ndarray):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Add batch dimension if needed
    has_batch_dim = len(tf.shape(image)) == 4
    if not has_batch_dim:
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
    
    # Split into R, G, B channels
    channels = tf.split(image, num_or_size_splits=3, axis=-1)  # List of (batch, height, width, 1)
    feature_maps = []
    
    for channel in channels:
        # Apply all 5 filters to this channel
        fm = tf.nn.conv2d(channel, SRM_FILTERS_TF, strides=1, padding='SAME')  # (batch, height, width, 5)
        feature_maps.append(fm)
    
    # Concatenate feature maps: (batch, height, width, 15)
    result = tf.concat(feature_maps, axis=-1)
    
    # Convert to numpy - will be easier to manipulate shapes
    result_np = result.numpy()
    
    # Log shape after filtering
    logger.info(f"SRM filter result shape (with batch): {result_np.shape}")
    
    # Remove batch dimension if it wasn't there originally
    if not has_batch_dim:
        result_np = result_np[0]  # Remove batch dimension
    
    logger.info(f"Final SRM filter result shape: {result_np.shape}")
    
    return result_np

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load an image from disk and preprocess it exactly like the training code.
    
    Args:
        image_path (str): Path to image file
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image tensor ready for SRM filtering
    """
    try:
        # Read the image file using PIL (more reliable than TF for various formats)
        with Image.open(image_path) as img:
            logger.info(f"Loaded image: {image_path}, original size: {img.size}, mode: {img.mode}")
            
            # Convert to RGB mode in case of RGBA or other formats
            if img.mode != 'RGB':
                img = img.convert('RGB')
                logger.info(f"Converted image to RGB mode")
            
            # Resize to target size
            img = img.resize(target_size)
            
            # Convert to numpy array and normalize to [0,1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            logger.info(f"Preprocessed image shape: {img_array.shape}")
            
            return img_array
            
    except Exception as e:
        # Fallback to TensorFlow loading
        logger.warning(f"PIL loading failed, trying TensorFlow: {str(e)}")
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        
        img_array = img.numpy()
        logger.info(f"TF preprocessed image shape: {img_array.shape}")
        
        return img_array

def apply_srm_filters(image_array):
    """Apply SRM filters to an image that's already loaded as a numpy array.
    
    Args:
        image_array (numpy.ndarray): Image array with shape [height, width, 3]
        
    Returns:
        numpy.ndarray: Feature maps after applying SRM filters with shape [height, width, 15]
    """
    # Ensure correct input shape
    if len(image_array.shape) == 4:
        # If input has batch dimension, use first image only
        logger.warning(f"Input has 4 dimensions, using only first image: {image_array.shape}")
        image_array = image_array[0]
    
    if image_array.shape[-1] != 3:
        logger.error(f"Invalid image shape: {image_array.shape}, last dimension should be 3 (RGB)")
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image_array.shape}")
        
    # Convert numpy array to tensor and apply SRM filters 
    filtered_img = apply_srm_filters_tf(image_array)
    
    # Check the output shape
    if len(filtered_img.shape) != 3 or filtered_img.shape[-1] != 15:
        logger.error(f"Invalid output shape: {filtered_img.shape}, expected [height, width, 15]")
        raise ValueError(f"Invalid SRM filter output shape: {filtered_img.shape}")
    
    return filtered_img 

def apply_dct_transform(image_array):
    """Apply 2D Discrete Cosine Transform (DCT) to an image.
    
    Args:
        image_array (numpy.ndarray): Image array with shape [height, width, 3]
        
    Returns:
        numpy.ndarray: DCT coefficients with shape [height, width, 3]
    """
    # Ensure correct input shape
    if len(image_array.shape) == 4:
        # If input has batch dimension, use first image only
        logger.warning(f"Input has 4 dimensions, using only first image: {image_array.shape}")
        image_array = image_array[0]
    
    if image_array.shape[-1] != 3:
        logger.error(f"Invalid image shape: {image_array.shape}, last dimension should be 3 (RGB)")
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image_array.shape}")
    
    # Apply DCT to each channel separately
    dct_coeffs = np.zeros_like(image_array, dtype=np.float32)
    
    for c in range(3):  # Process each channel (R,G,B)
        # Apply 2D DCT channel-wise using TensorFlow
        channel = tf.constant(image_array[:, :, c], dtype=tf.float32)
        dct_channel = tf.signal.dct(tf.signal.dct(channel, type=2, norm='ortho'), 
                                   type=2, norm='ortho', axis=0)
        dct_coeffs[:, :, c] = dct_channel.numpy()
    
    logger.info(f"DCT transform shape: {dct_coeffs.shape}")
    return dct_coeffs

def extract_dct_features(image_array, block_size=8, num_coefficients=64):
    """Extract DCT features from image using block processing.
    
    Args:
        image_array (numpy.ndarray): Image array with shape [height, width, 3]
        block_size (int): Size of DCT blocks (typically 8x8 for JPEG)
        num_coefficients (int): Number of coefficients to keep per block (max block_size²)
        
    Returns:
        numpy.ndarray: DCT features suitable for model input
    """
    # Ensure input is the right shape
    if len(image_array.shape) == 4:
        image_array = image_array[0]  # Remove batch dimension if present
    
    # Get image dimensions
    height, width, channels = image_array.shape
    
    # Make sure image dimensions are multiples of block_size
    # by padding if necessary
    pad_h = (block_size - height % block_size) % block_size
    pad_w = (block_size - width % block_size) % block_size
    
    if pad_h > 0 or pad_w > 0:
        # Pad the image to make dimensions multiples of block_size
        image_array = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        height, width, _ = image_array.shape
        logger.info(f"Padded image to shape: {image_array.shape}")
    
    # Calculate number of blocks
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    
    # Determine how many coefficients to keep per block and channel
    if num_coefficients > block_size * block_size:
        num_coefficients = block_size * block_size
        logger.warning(f"Limiting coefficients to {num_coefficients}")
    
    # Prepare output array for DCT features
    # Shape: [num_blocks_h * num_blocks_w, num_coefficients, channels]
    feature_vector = np.zeros((num_blocks_h * num_blocks_w, num_coefficients, channels), dtype=np.float32)
    
    # Process each block
    block_idx = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Extract block
            block = image_array[i:i+block_size, j:j+block_size, :]
            
            # Apply DCT to each channel of the block
            for c in range(channels):
                # Apply 2D DCT to the block
                channel_block = tf.constant(block[:, :, c], dtype=tf.float32)
                dct_block = tf.signal.dct(tf.signal.dct(channel_block, type=2, norm='ortho'), 
                                         type=2, norm='ortho', axis=0)
                
                # Convert to numpy array and flatten in zigzag order (approximation)
                # This prioritizes low-frequency coefficients which carry more information
                dct_flat = _zigzag_flatten(dct_block.numpy(), num_coefficients)
                
                # Store the DCT coefficients
                feature_vector[block_idx, :, c] = dct_flat
            
            block_idx += 1
    
    # Reshape to create final feature tensor: (num_blocks, num_coefficients*channels)
    final_features = feature_vector.reshape(num_blocks_h * num_blocks_w, -1)
    
    logger.info(f"Extracted DCT features with shape: {final_features.shape}")
    return final_features

def _zigzag_flatten(block, num_coefficients):
    """Flatten a square block in zigzag order to prioritize low-frequency coefficients.
    
    Args:
        block (numpy.ndarray): Square block of DCT coefficients
        num_coefficients (int): Number of coefficients to keep
        
    Returns:
        numpy.ndarray: Flattened coefficients in zigzag order
    """
    # Get block size
    block_size = block.shape[0]
    
    # Initialize output array
    result = np.zeros(num_coefficients, dtype=np.float32)
    
    # Simple zigzag implementation
    index = 0
    for i in range(2 * block_size - 1):
        if i < block_size:
            # Upper triangular matrix
            z_min = 0
            z_max = i
        else:
            # Lower triangular matrix
            z_min = i - block_size + 1
            z_max = block_size - 1
        
        # Determine zigzag direction
        zigzag_up = (i % 2 == 0)
        
        for z in range(z_min, z_max + 1):
            if zigzag_up:
                y = i - z
                x = z
            else:
                y = z
                x = i - z
            
            # Store coefficient and increment index
            if index < num_coefficients:
                result[index] = block[y, x]
                index += 1
            else:
                # We've collected enough coefficients
                return result
    
    return result 