import os
import numpy as np
from .base_model import BaseModel
import logging

# Configure logging
logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple deepfake detection models."""
    
    def __init__(self, debug_mode=False):
        """Initialize the ensemble model.
        
        Args:
            debug_mode (bool): Whether to output extra debugging information
        """
        super().__init__(model_name="Ensemble")
        self.models = []
        self.debug_mode = debug_mode
        self.model_weights = {}  # Store weights for each model
        self.use_weighted_voting = False  # Flag to enable weighted voting
    
    def add_model(self, model):
        """Add a model to the ensemble.
        
        Args:
            model (BaseModel): Model to add to the ensemble
        """
        if not isinstance(model, BaseModel):
            raise TypeError("Model must be an instance of BaseModel")
        
        self.models.append(model)
        
        # Update weights for all models to maintain equal distribution
        num_models = len(self.models)
        for existing_model in self.models:
            self.model_weights[existing_model.model_name] = 1.0 / num_models
        
        logger.info(f"Added {model.model_name} to ensemble")
    
    def set_model_weights(self, weights_dict):
        """Set weights for model predictions.
        
        Args:
            weights_dict (dict): Dictionary mapping model names to weights
        """
        logger.info(f"DEBUG: Setting model weights: {weights_dict}")
        logger.info(f"DEBUG: Current model names in ensemble: {[m.model_name for m in self.models]}")
        
        self.model_weights = weights_dict.copy()
        self.use_weighted_voting = True
        logger.info("Weighted voting enabled with weights:")
        for model_name, weight in self.model_weights.items():
            logger.info(f"  {model_name}: {weight:.3f}")
        
        logger.info(f"DEBUG: After setting - use_weighted_voting: {self.use_weighted_voting}")
        logger.info(f"DEBUG: After setting - model_weights: {self.model_weights}")
    
    @property
    def weights(self):
        """Get current model weights for external access."""
        return self.model_weights.copy()
    
    def enable_weighted_voting(self, enable=True):
        """Enable or disable weighted voting.
        
        Args:
            enable (bool): Whether to use weighted voting
        """
        self.use_weighted_voting = enable
        if enable:
            logger.info("Weighted voting enabled")
        else:
            logger.info("Weighted voting disabled - using majority voting")
    
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
        # logger.info(f"Ensemble analyzing image: {image_path}")
        
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
                    
                # logger.info(f"Getting prediction from {model.model_name}")
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
                    # logger.info(f"{model.model_name} result: {result}")
                    
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
                
                # Get the weight for this model
                model_weight = self.model_weights.get(model.model_name, 1.0 / len(self.models)) if self.use_weighted_voting else (1.0 / len(self.models))
                
                # Store individual result with weight information
                individual_results.append({
                    "model": model.model_name,
                    "probability": float(fake_probability),  # This is fake probability
                    "prediction": prediction,
                    "weight": float(model_weight)
                })
                # logger.info(f"{model.model_name} prediction: {fake_probability:.4f} (fake probability)")
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
        # logger.info(f"All fake probabilities: {all_fake_probabilities}")
        # logger.info(f"Average fake probability: {avg_fake_probability}")
        # logger.info(f"Median fake probability: {median_fake_probability}")
        
        # Get vote count for logs
        vote_count = f"{fake_votes} fake, {real_votes} real"
        # logger.info(f"Vote distribution: {vote_count}")
        
        # Determine final prediction based on voting method
        if self.use_weighted_voting:
            # Use weighted voting
            weighted_fake_score = 0.0
            total_weight = 0.0
            
            if self.debug_mode:
                logger.info("USING WEIGHTED VOTING:")
            
            for result in individual_results:
                model_name = result["model"]
                weight = self.model_weights.get(model_name, 1.0 / len(self.models))
                fake_prob = result["probability"]
                
                weighted_fake_score += fake_prob * weight
                total_weight += weight
                
                if self.debug_mode:
                    logger.info(f"  {model_name}: prob={fake_prob:.3f}, weight={weight:.3f}, contribution={fake_prob * weight:.3f}")
            
            # Normalize by total weight
            if total_weight > 0:
                weighted_fake_probability = weighted_fake_score / total_weight
            else:
                weighted_fake_probability = median_fake_probability
            
            is_fake = weighted_fake_probability > 0.765
            logger.info(f"🎯 WEIGHTED VOTING: weighted_fake_prob={weighted_fake_probability:.3f}, decision={'FAKE' if is_fake else 'REAL'}")
            if self.debug_mode:
                logger.info(f"DECISION: Using weighted voting (weighted_fake_prob={weighted_fake_probability:.3f})")
                logger.info(f"Final decision: {'FAKE' if is_fake else 'REAL'}")
        else:
            # Use majority voting as the primary decision factor
            # Only use probability when votes are tied
            is_fake_votes = fake_votes > real_votes
            
            if fake_votes == real_votes:
                # In case of tie, use the median probability
                is_fake = median_fake_probability > 0.5
                if self.debug_mode:
                    logger.info(f"DECISION: Votes tied at {fake_votes}-{real_votes}, using median probability ({median_fake_probability})")
                    logger.info(f"Final decision: {'FAKE' if is_fake else 'REAL'}")
            else:
                # Otherwise use majority vote
                is_fake = is_fake_votes
                if self.debug_mode:
                    logger.info(f"DECISION: Using majority vote {fake_votes}-{real_votes}")
                    logger.info(f"Final decision: {'FAKE' if is_fake else 'REAL'}")
        
        # Log if model votes disagree with probability-based decision
        if (is_fake and median_fake_probability < 0.5) or (not is_fake and median_fake_probability > 0.5):
            # logger.warning(f"Vote-based prediction ({is_fake}) differs from median probability ({median_fake_probability})")
            if self.debug_mode:
                logger.info(f"WARNING: Vote result ({is_fake}) contradicts median probability ({median_fake_probability})")
        
        # Improved confidence calculation based on multiple factors WITH model weights
        
        # Get the final probability that will be used (weighted or median)
        if self.use_weighted_voting:
            final_prob_for_confidence = weighted_fake_probability
        else:
            final_prob_for_confidence = median_fake_probability
        
        # 1. Decision strength: How far the final probability is from the decision boundary (0.5)
        decision_strength = abs(final_prob_for_confidence - 0.5) * 2  # Scale to [0,1]
        
        # 2. Weighted model agreement: How much models agree, considering their weights
        if self.use_weighted_voting and len(individual_results) > 1:
            # Calculate weighted variance to measure agreement
            weighted_mean = final_prob_for_confidence  # This is already the weighted mean
            weighted_variance = 0.0
            total_weight = 0.0
            
            for result in individual_results:
                model_name = result["model"]
                weight = self.model_weights.get(model_name, 1.0 / len(self.models))
                prob = result["probability"]
                
                weighted_variance += weight * (prob - weighted_mean) ** 2
                total_weight += weight
            
            if total_weight > 0:
                weighted_variance /= total_weight
                weighted_std = np.sqrt(weighted_variance)
                model_agreement = 1.0 - min(weighted_std * 2, 1.0)  # Lower weighted std = higher agreement
            else:
                model_agreement = 0.0
        else:
            # Fall back to regular standard deviation for non-weighted voting
            prob_std = float(np.std(all_fake_probabilities))
            model_agreement = 1.0 - min(prob_std * 2, 1.0)
        
        # 3. Weighted vote consensus: Consider model weights in voting
        if self.use_weighted_voting:
            # Calculate weighted votes
            weighted_fake_votes = 0.0
            weighted_real_votes = 0.0
            total_vote_weight = 0.0
            
            for result in individual_results:
                model_name = result["model"]
                weight = self.model_weights.get(model_name, 1.0 / len(self.models))
                prediction = result["prediction"]
                
                if prediction == "fake":
                    weighted_fake_votes += weight
                else:
                    weighted_real_votes += weight
                total_vote_weight += weight
            
            if total_vote_weight > 0:
                weighted_vote_ratio = max(weighted_fake_votes, weighted_real_votes) / total_vote_weight
                if weighted_vote_ratio >= 0.9:  # Very strong weighted consensus
                    vote_consensus = 1.0
                elif weighted_vote_ratio >= 0.75:  # Strong weighted majority
                    vote_consensus = 0.8
                elif weighted_vote_ratio >= 0.6:  # Clear weighted majority
                    vote_consensus = 0.6
                else:  # Weak or tied weighted votes
                    vote_consensus = 0.4
            else:
                weighted_vote_ratio = 0.0
                vote_consensus = 0.0
        else:
            # Use raw vote counts for non-weighted voting
            total_votes = fake_votes + real_votes
            if total_votes > 0:
                vote_ratio = max(fake_votes, real_votes) / total_votes
                weighted_vote_ratio = vote_ratio  # For logging consistency
                if vote_ratio == 1.0:  # Unanimous
                    vote_consensus = 1.0
                elif vote_ratio >= 0.75:  # Strong majority (3/4 or better)
                    vote_consensus = 0.8
                elif vote_ratio >= 0.6:  # Clear majority
                    vote_consensus = 0.6
                else:  # Tied or weak majority
                    vote_consensus = 0.3
            else:
                vote_ratio = 0.0
                weighted_vote_ratio = 0.0
                vote_consensus = 0.0
        
        # 4. Model weight distribution: Bonus if high-weight models are confident
        if self.use_weighted_voting and len(individual_results) > 1:
            # Calculate confidence of high-weight models
            high_weight_confidence = 0.0
            high_weight_total = 0.0
            
            # Consider models with above-average weight as "high-weight"
            avg_weight = 1.0 / len(self.models)
            
            for result in individual_results:
                model_name = result["model"]
                weight = self.model_weights.get(model_name, avg_weight)
                prob = result["probability"]
                
                if weight >= avg_weight:
                    # High-weight model: calculate its confidence (distance from 0.5)
                    model_confidence = abs(prob - 0.5) * 2
                    high_weight_confidence += weight * model_confidence
                    high_weight_total += weight
            
            if high_weight_total > 0:
                weight_confidence_bonus = high_weight_confidence / high_weight_total
            else:
                weight_confidence_bonus = 0.0
        else:
            weight_confidence_bonus = decision_strength  # Use decision strength when not using weighted voting
        
        # Weight the factors: decision strength (40%), model agreement (25%), vote consensus (15%), weight bonus (20%)
        final_confidence = float(0.4 * decision_strength + 0.25 * model_agreement + 0.15 * vote_consensus + 0.2 * weight_confidence_bonus)
        
        if self.debug_mode:
            logger.info(f"Weighted confidence breakdown:")
            logger.info(f"  - Decision strength: {decision_strength:.3f} (distance from 0.5)")
            logger.info(f"  - Model agreement: {model_agreement:.3f} ({'weighted' if self.use_weighted_voting else 'unweighted'} agreement)")
            logger.info(f"  - Vote consensus: {vote_consensus:.3f} (ratio={weighted_vote_ratio:.3f})")
            logger.info(f"  - Weight confidence bonus: {weight_confidence_bonus:.3f}")
            logger.info(f"  - Final confidence: {final_confidence:.3f}")
        
        # Determine final probability based on voting method used
        if self.use_weighted_voting:
            final_probability = weighted_fake_probability
            voting_method = "weighted"
        else:
            final_probability = median_fake_probability
            voting_method = "majority"
        
        # Create final result
        result = {
            "probability": final_probability,  # Use weighted or median probability based on voting method
            "prediction": "fake" if is_fake else "real",
            "confidence": final_confidence,  # Combined confidence
            "models_used": len(individual_results),
            "vote_distribution": vote_count,
            "voting_method": voting_method,
            "individual_results": individual_results
        }
        
        if self.debug_mode:
            logger.info("=" * 50)
            logger.info(f"ENSEMBLE FINAL DECISION: {result['prediction'].upper()}")
            logger.info(f"Fake probability: {result['probability']}")
            logger.info(f"Confidence: {result['confidence']}")
            logger.info("=" * 50)
        
        # logger.info(f"Final ensemble result: {result}")
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