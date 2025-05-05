import tensorflow as tf
import os

def load_model_direct(model_path, model_name="model"):
    """Load a model without any fallback mechanisms.
    
    Args:
        model_path (str): Path to the model file
        model_name (str): Name of the model for logging
        
    Returns:
        model: The loaded model
        
    Raises:
        Exception: If model loading fails
    """
    print(f"Loading {model_name} from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model {model_name} loaded successfully")
    return model

def check_tf_gpu():
    """Check if TensorFlow can see and use GPUs.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow sees {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
        return True
    else:
        print("No GPUs detected. Running on CPU.")
        return False 