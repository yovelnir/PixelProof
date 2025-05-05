import os
import numpy as np
from .base_model import BaseModel
import logging

# Configure logging
logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple deepfake detection models."""
    
    def __init__(self, debug_mode=True):
        """Initialize the ensemble model.
        
        Args:
            debug_mode (bool): Whether to output extra debugging information
        """
        super().__init__(model_name="Ensemble")
        self.models = []
        self.debug_mode = debug_mode
    
    def add_model(self, model):
        """Add a model to the ensemble.
        
        Args:
            model (BaseModel): Model to add to the ensemble
        """
        if not isinstance(model, BaseModel):
            raise TypeError("Model must be an instance of BaseModel")
        
        self.models.append(model)
        logger.info(f"Added {model.model_name} to ensemble")
    
    def load(self, model_path=None):
        """Implement the required abstract method.
        
        For EnsembleModel, models are added individually rather than loaded directly.
        
        Args:
            model_path: Ignored, as models are added individually
            
        Returns:
            bool: Success status (always True since actual loading happens separately)
        """
        # This method is implemented to satisfy the abstract method requirement,
        # but the actual model loading is handled by the add_model method
        return True
    
    def analyze(self, image_path):
        """Analyze an image with all models in the ensemble using majority voting.
        
        Args:
            image_path (str): Path to the image to analyze
            
        Returns:
            dict: Combined analysis results with majority voting
            
        Raises:
            ValueError: If no models are available or if all models fail
        """
        logger.info(f"Ensemble analyzing image: {image_path}")
        
        if not self.models:
            logger.error("No models in ensemble!")
            raise ValueError("No models available in the ensemble")
        
        # Track results from each model
        individual_results = []
        real_votes = 0
        fake_votes = 0
        all_fake_probabilities = []
        
        # Debug information for understanding each model's decision
        if self.debug_mode:
            logger.info("=" * 50)
            logger.info("ENSEMBLE DEBUG MODE ENABLED")
            logger.info("=" * 50)
        
        # Get predictions from each model
        for model in self.models:
            try:
                if self.debug_mode:
                    logger.info(f"\nProcessing model: {model.model_name}")
                    
                logger.info(f"Getting prediction from {model.model_name}")
                result = model.analyze(image_path)
                
                # Extract probability, handling different result formats
                if isinstance(result, dict):
                    # Each model returns fake probability in the 'probability' field
                    # This is consistent across all models (1-predicted_real_probability)
                    fake_probability = result.get("probability", None)
                    prediction = result.get("prediction", None)
                    
                    if fake_probability is None:
                        logger.warning(f"Missing probability in result from {model.model_name}")
                        continue
                    
                    # Log the entire result dictionary
                    logger.info(f"{model.model_name} result: {result}")
                    
                    if self.debug_mode:
                        logger.info(f"  - Raw output: {result}")
                        logger.info(f"  - Fake probability: {fake_probability}")
                        logger.info(f"  - Prediction: {prediction}")
                else:
                    # If result is not a dict (shouldn't happen), skip this model
                    logger.warning(f"Unexpected result type from {model.model_name}: {type(result)}")
                    continue
                
                # Validate probability is within [0,1]
                if not (0 <= fake_probability <= 1):
                    logger.warning(f"Invalid probability from {model.model_name}: {fake_probability}, clamping to [0,1]")
                    fake_probability = max(0, min(1, fake_probability))
                
                all_fake_probabilities.append(fake_probability)
                
                # Check the prediction string directly to count votes
                if prediction == "real":
                    real_votes += 1
                    if self.debug_mode:
                        logger.info(f"  - Adding REAL vote (total now: {real_votes})")
                else:
                    fake_votes += 1
                    if self.debug_mode:
                        logger.info(f"  - Adding FAKE vote (total now: {fake_votes})")
                
                # Store individual result
                individual_results.append({
                    "model": model.model_name,
                    "probability": float(fake_probability),  # This is fake probability
                    "prediction": prediction
                })
                logger.info(f"{model.model_name} prediction: {fake_probability:.4f} (fake probability)")
            except Exception as e:
                logger.error(f"Error from {model.model_name}: {str(e)}")
                # Skip this model rather than using a neutral prediction
                if self.debug_mode:
                    logger.info(f"  - Model error: {str(e)}")
        
        # If no models successfully analyzed the image
        if not individual_results:
            logger.error("All models failed to analyze the image")
            raise ValueError("All models failed to analyze the image")
        
        # Calculate average and median fake probability
        avg_fake_probability = float(np.mean(all_fake_probabilities))
        median_fake_probability = float(np.median(all_fake_probabilities))
        
        # Log all probability values to diagnose
        logger.info(f"All fake probabilities: {all_fake_probabilities}")
        logger.info(f"Average fake probability: {avg_fake_probability}")
        logger.info(f"Median fake probability: {median_fake_probability}")
        
        # Get vote count for logs
        vote_count = f"{fake_votes} fake, {real_votes} real"
        logger.info(f"Vote distribution: {vote_count}")
        
        # Determine final prediction based on majority vote primarily, then probability
        is_fake_votes = fake_votes > real_votes
        
        # Use majority voting as the primary decision factor
        # Only use probability when votes are tied
        if fake_votes == real_votes:
            # In case of tie, use the median probability
            is_fake = median_fake_probability > 0.5
            logger.info(f"Vote tie, using median probability: {median_fake_probability}")
            if self.debug_mode:
                logger.info(f"DECISION: Votes tied at {fake_votes}-{real_votes}, using median probability ({median_fake_probability})")
                logger.info(f"Final decision: {'FAKE' if is_fake else 'REAL'}")
        else:
            # Otherwise use majority vote
            is_fake = is_fake_votes
            logger.info(f"Using majority vote: {is_fake}")
            if self.debug_mode:
                logger.info(f"DECISION: Using majority vote {fake_votes}-{real_votes}")
                logger.info(f"Final decision: {'FAKE' if is_fake else 'REAL'}")
        
        # Log if model votes disagree with probability-based decision
        if (is_fake and median_fake_probability < 0.5) or (not is_fake and median_fake_probability > 0.5):
            logger.warning(f"Vote-based prediction ({is_fake}) differs from median probability ({median_fake_probability})")
            if self.debug_mode:
                logger.info(f"WARNING: Vote result ({is_fake}) contradicts median probability ({median_fake_probability})")
        
        # Confidence is based on vote unanimity and distance from 0.5
        if fake_votes == len(individual_results) or real_votes == len(individual_results):
            # Unanimous vote gets high confidence
            vote_confidence = 1.0
            logger.info("Unanimous vote, high confidence")
            if self.debug_mode:
                logger.info("Unanimous decision, setting high confidence")
        else:
            # Non-unanimous vote - confidence based on vote distribution
            vote_confidence = abs(fake_votes - real_votes) / len(individual_results)
            logger.info(f"Non-unanimous vote: confidence from distribution = {vote_confidence}")
            if self.debug_mode:
                logger.info(f"Non-unanimous decision, vote confidence: {vote_confidence}")
        
        # Combine probability and vote confidence
        probability_confidence = abs(median_fake_probability - 0.5) * 2  # Scale distance from 0.5 to [0,1]
        logger.info(f"Probability-based confidence: {probability_confidence}")
        
        final_confidence = float((vote_confidence + probability_confidence) / 2)
        if self.debug_mode:
            logger.info(f"Final confidence calculation: (vote_conf {vote_confidence} + prob_conf {probability_confidence}) / 2 = {final_confidence}")
        
        # Create final result
        result = {
            "probability": median_fake_probability,  # Use median instead of mean
            "prediction": "fake" if is_fake else "real",
            "confidence": final_confidence,  # Combined confidence
            "models_used": len(individual_results),
            "vote_distribution": vote_count,
            "individual_results": individual_results
        }
        
        if self.debug_mode:
            logger.info("=" * 50)
            logger.info(f"ENSEMBLE FINAL DECISION: {result['prediction'].upper()}")
            logger.info(f"Fake probability: {result['probability']}")
            logger.info(f"Confidence: {result['confidence']}")
            logger.info("=" * 50)
        
        logger.info(f"Final ensemble result: {result}")
        return result
    
    def preprocess(self, image_data):
        """Preprocess is handled by individual models.
        This returns the image data for processing by each model.
        
        Args:
            image_data: Image path or numpy array
            
        Returns:
            The image data unchanged
        """
        return image_data
    
    def predict(self, image_data):
        """Run predictions using analyze method.
        
        Args:
            image_data: Image data to be processed
            
        Returns:
            dict: Analysis results
            
        Raises:
            ValueError: If prediction fails
        """
        return self.analyze(image_data) 