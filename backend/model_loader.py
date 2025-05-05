"""
Model loader utility to handle import issues and setup the ensemble model.
"""
import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the model classes
try:
    from models.ensemble_model import EnsembleModel
    from models.srm_cnn_model import SRMCNNModel
    from models.dct_model import DCTModel
    from models.dct_ae_model import DCTAutoencoderModel
except ImportError as e:
    logger.error(f"Error importing model classes: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def load_models(models_dir=None):
    """Initialize and load all models.
    
    Args:
        models_dir (str): Directory containing model files
        
    Returns:
        EnsembleModel: Initialized ensemble model with all available models loaded
    """
    if models_dir is None:
        models_dir = os.path.join(current_dir, 'models')
    
    logger.info(f"Loading models from {models_dir}")
    
    try:
        # Initialize ensemble
        ensemble = EnsembleModel()
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            raise ValueError(f"Models directory not found: {models_dir}")
        
        # List available model files
        model_files = [f for f in os.listdir(models_dir) 
                      if f.endswith(('.keras', '.h5')) and os.path.isfile(os.path.join(models_dir, f))]
        
        if not model_files:
            logger.error(f"No model files found in {models_dir}")
            raise ValueError(f"No model files found in {models_dir}")
            
        logger.info(f"Found model files: {', '.join(model_files)}")
        
        # Check if autoencoder exists to determine if we can use encoder-based models
        ae_dir = os.path.join(models_dir, 'ae')
        ae_path_256 = os.path.join(ae_dir, "autoencoder_L256.keras")
        has_autoencoder = os.path.exists(ae_path_256)
        
        if has_autoencoder:
            logger.info(f"Found autoencoder at {ae_path_256}, will use encoder-based DCT models")
        else:
            logger.warning("Autoencoder not found, will use standard DCT models")
        
        # Load SRM-CNN models with appropriate latent sizes
        # These are more reasonable latent space sizes for a CNN processing 256x256 images
        # The model input is always 256x256 regardless of latent size
        
        # Load model from 256_SRM_CNN_model.keras file - use latent size 256
        srm_l256 = SRMCNNModel(latent_size=256)
        model_path = os.path.join(models_dir, '256_SRM_CNN_model.keras')
        if os.path.exists(model_path):
            if srm_l256.load(model_path):
                ensemble.add_model(srm_l256)
                logger.info("Successfully loaded SRM-CNN model with latent size 256")
        
        # Load model from 128_SRM_CNN.h5 file - use latent size 128
        srm_l128 = SRMCNNModel(latent_size=128)
        model_path = os.path.join(models_dir, '128_SRM_CNN.h5')
        if os.path.exists(model_path):
            if srm_l128.load(model_path):
                ensemble.add_model(srm_l128)
                logger.info("Successfully loaded SRM-CNN model with latent size 128")
        
        # Load all DCT models and add them to ensemble
        dct_models = {
            'dct_model.h5': 'Standard DCT',
            'dct_kfold_model.h5': 'DCT K-fold',
            'dct_kfold_noEarlyStopping_model.h5': 'DCT K-fold (No Early Stopping)',
            'dct_kfold_noEarlyStopping_SGD_model.h5': 'DCT K-fold (SGD, No Early Stopping)'
        }
        
        for filename, model_desc in dct_models.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                # Create appropriate model type based on autoencoder availability
                if has_autoencoder:
                    # Use the encoder version
                    dct_model = DCTModel(block_size=8, num_coefficients=64, use_encoder=True, latent_size=256)
                    model_name_suffix = "-AE"
                    logger.info(f"Creating encoder-based {model_desc} model")
                else:
                    # Fallback to standard version
                    dct_model = DCTModel(block_size=8, num_coefficients=64)
                    model_name_suffix = ""
                    logger.info(f"Creating standard {model_desc} model")
                
                # Set custom model name
                dct_model.model_name = f"{model_desc}{model_name_suffix}"
                
                # Try to load model
                if dct_model.load(model_path):
                    ensemble.add_model(dct_model)
                    logger.info(f"Successfully loaded {dct_model.model_name} model")
        
        # Check if any models were loaded
        if not ensemble.models:
            import tensorflow as tf
            import keras
            logger.error("No models were loaded successfully!")
            logger.error(f"TensorFlow version: {tf.__version__}")
            logger.error(f"Keras version: {keras.__version__}")
            raise ValueError("No models were loaded successfully!")
        else:
            logger.info(f"Successfully loaded {len(ensemble.models)} models into ensemble")
        
        return ensemble
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        # Propagate the error instead of returning an empty ensemble
        raise 