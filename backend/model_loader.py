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
except ImportError as e:
    logger.error(f"Error importing model classes: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def load_models(models_dir=None, use_cache=True, force_evaluation=False):
    """Initialize and load all models from .h5 and .keras files.
    
    Args:
        models_dir (str): Directory containing model files
        use_cache (bool): Whether to use cached weights if available
        force_evaluation (bool): Force model evaluation even if cache exists
        
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
        
        # Find all model files recursively
        model_files = []
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(('.keras', '.h5')):
                    full_path = os.path.join(root, file)
                    model_files.append(full_path)
        
        if not model_files:
            logger.error(f"No model files found in {models_dir}")
            raise ValueError(f"No model files found in {models_dir}")
            
        logger.info(f"Found {len(model_files)} model files: {[os.path.basename(f) for f in model_files]}")
        
        # Check if autoencoder exists for encoder-based models
        ae_dir = os.path.join(models_dir, 'ae')
        # Check for both .h5 and .keras formats
        ae_path_h5 = os.path.join(ae_dir, "autoencoder_L256.h5")
        ae_path_keras = os.path.join(ae_dir, "autoencoder_L256.keras")
        
        has_autoencoder = os.path.exists(ae_path_h5) or os.path.exists(ae_path_keras)
        ae_path_256 = ae_path_h5 if os.path.exists(ae_path_h5) else ae_path_keras
        
        if has_autoencoder:
            logger.info(f"Found autoencoder at {ae_path_256}, will use encoder-based DCT models where applicable")
        else:
            logger.warning("Autoencoder not found, will use standard models")
        
        # Load all discovered model files
        loaded_count = 0
        for model_path in model_files:
            filename = os.path.basename(model_path)

            if filename.startswith("autoencoder_L"):
                logger.info(f"Skipping autoencoder file: {filename}")
                continue

            logger.info(f"Attempting to load: {filename}")
            
            # Try to load the model using different model types based on filename patterns
            model_loaded = False
            
            # Try SRM-CNN models first
            if 'srm' in filename.lower() or 'cnn' in filename.lower():
                model_loaded = _try_load_srm_model(model_path, filename, ensemble)

            # ✅ Always try DCT if SRM fails
            if not model_loaded:
                model_loaded = _try_load_dct_model(model_path, filename, ensemble, has_autoencoder)
            
            # If still not loaded, try as generic DCT model
            if not model_loaded:
                model_loaded = _try_load_generic_model(model_path, filename, ensemble)
            
            if model_loaded:
                loaded_count += 1
                logger.info(f"Successfully loaded: {filename}")
            else:
                logger.warning(f"Failed to load: {filename}")
        
        # Check if any models were loaded
        if loaded_count == 0:
            import tensorflow as tf
            import keras
            logger.error("No models were loaded successfully!")
            logger.error(f"TensorFlow version: {tf.__version__}")
            logger.error(f"Keras version: {keras.__version__}")
            raise ValueError("No models were loaded successfully!")
        else:
            logger.info(f"Successfully loaded {loaded_count}/{len(model_files)} models into ensemble")
            
            # Weight management with caching support
            weights_applied = False
            
            # Try to load cached weights first (unless forced evaluation)
            if use_cache and not force_evaluation:
                logger.info("Checking for cached model weights...")
                try:
                    from weight_cache import load_model_weights, get_cache_info
                    
                    # Get cache info first
                    cache_info = get_cache_info()
                    if cache_info:
                        logger.info(f"Found weight cache from {cache_info['timestamp_human']} ({cache_info['age_hours']:.1f}h old)")
                        if cache_info.get('ensemble_accuracy'):
                            logger.info(f"Cached ensemble accuracy: {cache_info['ensemble_accuracy']:.3f}")
                    
                    # Load cached weights
                    cached_weights = load_model_weights()
                    
                    if cached_weights:
                        # Check if cached model names match current models
                        current_model_names = set(model.model_name for model in ensemble.models)
                        cached_model_names = set(cached_weights.keys())
                        
                        if current_model_names == cached_model_names:
                            logger.info("Applying cached weights to ensemble...")
                            ensemble.set_model_weights(cached_weights)
                            weights_applied = True
                            logger.info("Successfully applied cached weights")
                            
                            # Log the applied weights
                            total_weight = sum(cached_weights.values())
                            for model_name, weight in cached_weights.items():
                                percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                                logger.info(f"  {model_name}: {weight:.4f} ({percentage:.1f}%)")
                        else:
                            logger.warning("Cached model names don't match current models, running evaluation")
                            logger.warning(f"Current: {sorted(current_model_names)}")
                            logger.warning(f"Cached:  {sorted(cached_model_names)}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load cached weights: {str(e)}")
            
            # Run evaluation if no cached weights were applied or if forced
            if not weights_applied or force_evaluation:
                logger.info("Running model evaluation to calculate weights...")
                
                # Import and use the evaluator to evaluate the ensemble
                from evaluate_models import ModelEvaluator
                from weight_cache import save_model_weights
                
                try:
                    evaluator = ModelEvaluator(test_images_dir=os.path.join(current_dir, "test_images_gw"))
                    results = evaluator.evaluate_ensemble(ensemble, max_images=None)
                    
                    if results and "individual_results" in results:
                        logger.info("Model evaluation completed successfully")
                        ensemble_metrics = results.get("ensemble_metrics", {})
                        ensemble_accuracy = ensemble_metrics.get("accuracy", "N/A")
                        if ensemble_accuracy != "N/A":
                            logger.info(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
                        
                        # Calculate and apply weights
                        weights = evaluator._calculate_model_weights(results["individual_results"])
                        logger.info(f"DEBUG: Calculated weights: {weights}")
                        
                        if weights and any(weight > 0 for weight in weights.values()):
                            logger.info("Applying performance-based weights to ensemble...")
                            ensemble.set_model_weights(weights)

                            # Verify weighted voting is enabled
                            if not getattr(ensemble, 'use_weighted_voting', False):
                                ensemble.enable_weighted_voting(True)
                                logger.info("Enabled weighted voting")
                            
                            # Save weights to cache
                            try:
                                cache_saved = save_model_weights(weights, results)
                                if cache_saved:
                                    logger.info("Model weights saved to cache for future use")
                                else:
                                    logger.warning("Failed to save weights to cache")
                            except Exception as cache_e:
                                logger.warning(f"Cache save error: {str(cache_e)}")
                            
                            # Log the applied weights
                            total_weight = sum(weights.values())
                            for model_name, weight in weights.items():
                                percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                                logger.info(f"  {model_name}: {weight:.4f} ({percentage:.1f}%)")
                            
                            weights_applied = True
                        else:
                            logger.warning("No valid weights calculated, using equal weights")
                    else:
                        logger.warning("Evaluation failed - using equal weights")
                        
                except Exception as e:
                    logger.error(f"Model evaluation failed: {str(e)}")
                    logger.info("Proceeding with equal model weights")
            
            if not weights_applied:
                logger.info("Using equal weights for all models")
        
        return ensemble
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        # Propagate the error instead of returning an empty ensemble
        raise


def _try_load_srm_model(model_path, filename, ensemble):
    """Try to load model as SRM-CNN with inferred latent size and label inversion if needed."""
    try:
        # Infer latent size from filename
        latent_size = 128  # default
        if '256' in filename:
            latent_size = 256
        elif '128' in filename:
            latent_size = 128
        elif '64' in filename:
            latent_size = 64
        elif '512' in filename:
            latent_size = 512
        
        # Determine if labels should be inverted based on model characteristics
        # This can be configured based on specific model training patterns
        invert_labels = True
        
        # Add specific rules for models that need label inversion
       
        srm_model = SRMCNNModel(latent_size=latent_size, invert_labels=invert_labels)
        srm_model.model_name = f"SRM-CNN-L{latent_size} ({os.path.basename(filename)})"
        
        if srm_model.load(model_path):
            ensemble.add_model(srm_model)
            logger.info(f"Loaded SRM-CNN model with latent_size={latent_size}, invert_labels={invert_labels}")
            return True
    except Exception as e:
        logger.debug(f"Failed to load {filename} as SRM-CNN model: {str(e)}")
    
    return False


def _try_load_dct_model(model_path, filename, ensemble, has_autoencoder):
    """Try to load model as DCT model with the new structure."""
    invert_labels = True
    try:
        # Always try encoder-based DCT model first if autoencoder is available
        if has_autoencoder:
            logger.info(f"Loading {filename} as encoder-based DCT model (training pipeline)")
            # Don't hardcode latent_size - let the model determine it from input shape
            dct_model = DCTModel(block_size=8, num_coefficients=64, use_encoder=True, invert_labels=invert_labels)
            # Make model name unique by including the filename
            dct_model.model_name = f"DCT-AE ({os.path.basename(filename)})"
        else:
            logger.info(f"Loading {filename} as standard DCT model (legacy structure)")
            dct_model = DCTModel(block_size=8, num_coefficients=64, invert_labels=invert_labels)
            dct_model.model_name = f"DCT ({os.path.basename(filename)})"
        
        if dct_model.load(model_path):
            ensemble.add_model(dct_model)
            return True
    except Exception as e:
        logger.debug(f"Failed to load {filename} as DCT model: {str(e)}")
    
    return False


def _try_load_generic_model(model_path, filename, ensemble):
    """Try to load model using different configurations as fallback."""
    # Try different SRM-CNN configurations
    for latent_size in [64, 128, 256, 512]:
        # Try both with and without label inversion
        for invert_labels in [False, True]:
            try:
                srm_model = SRMCNNModel(latent_size=latent_size, invert_labels=invert_labels)
                invert_suffix = "-inverted" if invert_labels else ""
                srm_model.model_name = f"Generic-SRM-L{latent_size}{invert_suffix} ({os.path.basename(filename)})"
                
                if srm_model.load(model_path):
                    ensemble.add_model(srm_model)
                    logger.info(f"Loaded generic SRM-CNN model with latent_size={latent_size}, invert_labels={invert_labels}")
                    return True
            except Exception:
                continue
    
    # Try new structure DCT Autoencoder model
    try:
        logger.info(f"Trying {filename} as new structure DCT autoencoder model")
        dct_ae_model = DCTModel(use_encoder=True)
        dct_ae_model.model_name = f"Generic-DCT-AE-New ({os.path.basename(filename)})"
        
        if dct_ae_model.load(model_path):
            ensemble.add_model(dct_ae_model)
            return True
    except Exception as e:
        logger.debug(f"Failed to load {filename} as new DCT autoencoder model: {str(e)}")
    
    # Try standard DCT model as last resort
    try:
        dct_model = DCTModel(block_size=8, num_coefficients=64)
        dct_model.model_name = f"Generic-DCT ({os.path.basename(filename)})"
        
        if dct_model.load(model_path):
            ensemble.add_model(dct_model)
            return True
    except Exception as e:
        logger.debug(f"Failed to load {filename} as generic DCT model: {str(e)}")
    
    return False 