import os
import sys
import json
import logging
import tempfile
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our model loader
from model_loader import load_models

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(current_dir, "uploads")
MODELS_FOLDER = os.path.join(current_dir, "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Initialize the model
model = None

def init_model(use_cache=True, force_evaluation=False):
    """Initialize the model for image analysis.
    
    Args:
        use_cache (bool): Whether to use cached weights if available
        force_evaluation (bool): Force model evaluation even if cache exists
    """
    global model
    
    try:
        logger.info("Initializing models...")
        model = load_models(MODELS_FOLDER, use_cache=use_cache, force_evaluation=force_evaluation)
        
        if model and len(model.models) > 0:
            logger.info(f"Models loaded successfully: {len(model.models)} model(s) available")
            for idx, m in enumerate(model.models):
                logger.info(f"  Model {idx+1}: {m.model_name}")
        else:
            logger.error("No models were loaded successfully! API will not work correctly.")
    except Exception as e:
        logger.error(f"Failed to load models! Error: {str(e)}")
        logger.exception("Stack trace:")
        logger.error("API will not function correctly without models.")
        model = None

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze an uploaded image for deepfake detection."""
    global model
    
    # Check if model is initialized
    if model is None:
        logger.error("Model not initialized when API was called")
        return jsonify({
            'error': 'Model not initialized',
            'message': 'The analysis model failed to initialize. Please try again later.'
        }), 500
    
    # Log how many models are loaded
    logger.info(f"Analyze API called with {len(model.models)} models available")
    
    # If we have models, list them
    if model.models:
        for idx, m in enumerate(model.models):
            logger.info(f"  Model {idx+1}: {m.model_name}")
    else:
        logger.warning("No models available for analysis!")
    
    # Check if file was uploaded
    if 'image' not in request.files:
        logger.warning("No file provided in request")
        return jsonify({
            'error': 'No file provided',
            'message': 'No image file was uploaded'
        }), 400
        
    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        logger.warning("Empty filename in request")
        return jsonify({
            'error': 'Empty file',
            'message': 'An empty file was uploaded'
        }), 400
    
    # Log the incoming file details
    logger.info(f"Processing uploaded file: {file.filename}")
    
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            file.save(temp.name)
            temp_path = temp.name
            logger.info(f"Saved uploaded file to temporary path: {temp_path}")
        
        # Analyze the image
        logger.info("Calling model.analyze...")
        result = model.analyze(temp_path)
        logger.info(f"Analysis completed with result: {result}")
        
        # Check for neutral prediction (close to 0.5)
        if result and isinstance(result, dict) and "probability" in result:
            prob = result["probability"]
            if 0.49 <= prob <= 0.51:
                logger.warning(f"NEUTRAL PREDICTION DETECTED: {prob}")
                logger.warning("This may indicate the model is falling back to default values")
        
        # Clean up the temporary file
        os.unlink(temp_path)
        logger.info("Temporary file removed")
        
        # Return the results
        return jsonify(result)
    
    except Exception as e:
        logger.exception("Error analyzing image")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check API status."""
    global model
    
    models_count = len(model.models) if model else 0
    model_names = [m.model_name for m in model.models] if model and model.models else []
    
    status_info = {
        'status': 'ok',
        'message': 'API is running',
        'model_loaded': model is not None and models_count > 0,
        'models_available': models_count,
        'model_names': model_names
    }
    
    logger.info(f"Status check: {status_info}")
    return jsonify(status_info)

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        'message': 'PixelProof Backend API',
        'status': 'running',
        'endpoints': {
            'analyze': '/api/analyze - POST - Upload image for analysis',
            'status': '/api/status - GET - Check API status'
        }
    })

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PixelProof Backend API')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Run model evaluation and update weights cache')
    parser.add_argument('--force-evaluation', action='store_true',
                       help='Force model evaluation even if cache exists')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable weight caching (always evaluate)')
    parser.add_argument('--cache-info', action='store_true',
                       help='Show cache information and exit')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear weight cache and exit')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the server on (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind the server to (default: 0.0.0.0)')
    return parser.parse_args()

def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Handle cache operations that don't require model loading
    if args.cache_info:
        from weight_cache import get_cache_info
        cache_info = get_cache_info()
        if cache_info:
            print("Weight Cache Information:")
            print(f"  File: {cache_info['cache_file']}")
            print(f"  Created: {cache_info['timestamp_human']}")
            print(f"  Age: {cache_info['age_hours']:.1f} hours")
            print(f"  Models: {cache_info['total_models']}")
            if cache_info.get('ensemble_accuracy'):
                print(f"  Ensemble Accuracy: {cache_info['ensemble_accuracy']:.3f}")
                print(f"  Ensemble F1: {cache_info.get('ensemble_f1', 'N/A'):.3f}")
                print(f"  Ensemble AUC: {cache_info.get('ensemble_auc', 'N/A'):.3f}")
        else:
            print("No weight cache found.")
        return
    
    if args.clear_cache:
        from weight_cache import clear_cache
        if clear_cache():
            print("Weight cache cleared successfully.")
        else:
            print("Failed to clear weight cache.")
        return
    
    # Configure model loading options
    use_cache = not args.no_cache
    force_evaluation = args.force_evaluation or args.evaluate
    
    if args.evaluate and not force_evaluation:
        # If just --evaluate is specified, force evaluation
        force_evaluation = True
    
    # Initialize model with specified options
    logger.info(f"Model loading options: use_cache={use_cache}, force_evaluation={force_evaluation}")
    init_model(use_cache=use_cache, force_evaluation=force_evaluation)
    
    # If only evaluation was requested, exit after model loading
    if args.evaluate and not args.force_evaluation:
        logger.info("Model evaluation completed. Exiting.")
        return
    
    # Start the Flask server
    logger.info(f"Starting Flask server on {args.host}:{args.port}...")
    app.run(debug=False, host=args.host, port=args.port)

# Initialize model when importing (for backward compatibility)
if __name__ != '__main__':
    init_model()

# Run the app
if __name__ == '__main__':
    main() 