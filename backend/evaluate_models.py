#!/usr/bin/env python3
"""
Model evaluation script for deepfake detection models.
Evaluates individual models and ensemble on test images.
"""

import os
import sys
import json
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def print_confusion_matrix(model_name: str, tn: int, fp: int, fn: int, tp: int, 
                          accuracy: float, precision: float, recall: float, 
                          f1: float, auc: float):
    """Print a nicely formatted confusion matrix with evaluation metrics.
    
    Args:
        model_name: Name of the model
        tn, fp, fn, tp: Confusion matrix values
        accuracy: Standard accuracy
        precision, recall, f1: Core binary classification metrics
        auc: Area under the ROC curve
    """
    print("\n" + "="*70)
    print(f"CONFUSION MATRIX - {model_name}")
    print("="*70)
    print("                    Predicted")
    print("                 Fake    Real")
    print(f"Actual   Fake  │ {tn:4d}  │ {fp:4d}  │")
    print(f"         Real  │ {fn:4d}  │ {tp:4d}  │")
    print("-"*70)
    
    # Core metrics
    print("CORE METRICS:")
    print(f"Accuracy:         {accuracy:.3f}")
    print(f"Precision:        {precision:.3f}")
    print(f"Recall:           {recall:.3f}")
    print(f"F1-Score:         {f1:.3f}")
    print(f"AUC:              {auc:.3f}")
    print(f"Specificity:      {tn/(tn+fp):.3f}" if (tn+fp) > 0 else "Specificity: N/A")
    print(f"Sensitivity:      {recall:.3f}")
    
    print("-"*70)
    
    # Class-specific interpretation
    print("CLASS-SPECIFIC PERFORMANCE:")
    fake_total = tn + fp
    real_total = fn + tp
    
    if fake_total > 0:
        fake_accuracy = tn / fake_total
        print(f"Fake Detection:   {fake_accuracy:.3f} ({tn}/{fake_total} correct)")
    else:
        print("Fake Detection:   N/A (no fake images in test set)")
        
    if real_total > 0:
        real_accuracy = tp / real_total
        print(f"Real Detection:   {real_accuracy:.3f} ({tp}/{real_total} correct)")
    else:
        print("Real Detection:   N/A (no real images in test set)")
    
    print("-"*70)
    
    # Confusion matrix explanation
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"• True Negatives (TN): {tn} - Correctly identified fake images")
    print(f"• False Positives (FP): {fp} - Real images incorrectly classified as fake")
    print(f"• False Negatives (FN): {fn} - Fake images incorrectly classified as real")
    print(f"• True Positives (TP): {tp} - Correctly identified real images")
    
    print("="*70 + "\n")

class ModelEvaluator:
    """Evaluates deepfake detection models on test dataset."""
    
    def __init__(self, test_images_dir: str = os.path.join(current_dir, "test_images")):
        """Initialize the evaluator.
        
        Args:
            test_images_dir: Path to test images directory
        """
        
        self.test_images_dir = test_images_dir
        self.fake_dir = os.path.join(test_images_dir, "fake")
        self.real_dir = os.path.join(test_images_dir, "real")
        
        # Verify directories exist
        if not os.path.exists(self.fake_dir):
            raise ValueError(f"Fake images directory not found: {self.fake_dir}")
        if not os.path.exists(self.real_dir):
            raise ValueError(f"Real images directory not found: {self.real_dir}")
        
        # Get test image paths and labels
        self.fake_images = self._load_fake_images()
        self.real_images = self._load_real_images()
        logger.info(f"Loaded {len(self.fake_images)} fake images")
        logger.info(f"Loaded {len(self.real_images)} real images")
    
    def _load_fake_images(self):
        fake_images = sorted([os.path.join(self.fake_dir, f) for f in os.listdir(self.fake_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        return fake_images

    def _load_real_images(self):
        real_images = sorted([os.path.join(self.real_dir, f) for f in os.listdir(self.real_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        return real_images

    
    def evaluate_model(self, model, max_images: int = None) -> Dict[str, Any]:
        """Evaluate a single model on test data.
        
        Args:
            model: Model to evaluate (must have analyze method)
            max_images: Maximum number of images to evaluate (for testing)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating model: {model.model_name}")
        
        predictions = []
        probabilities = []
        true_labels = []
        failed_images = []
        real_paths = []
        fake_paths = []

        if max_images:
            fake_paths = self.fake_images[:max_images]
            real_paths = self.real_images[:max_images]
        else:
            fake_paths = self.fake_images
            real_paths = self.real_images

        test_subset = list(zip(fake_paths, [0] * len(fake_paths))) + list(zip(real_paths, [1] * len(real_paths)))
        test_subset.sort(key=lambda x: os.path.basename(x[0]))
        
        logger.info(f"Evaluating on {len(test_subset)} images")
        
        start_time = time.time()
        
        for i, (image_path, true_label) in enumerate(test_subset):
            try:
                # Get model prediction
                result = model.analyze(image_path)
                
                if isinstance(result, dict) and "probability" in result and "prediction" in result:
                    fake_prob = result["probability"]  # This is fake probability
                    prediction = result["prediction"]
                    
                    # Convert prediction to binary (fake=0, real=1)
                    pred_binary = 0 if prediction == "fake" else 1
                    
                    predictions.append(pred_binary)
                    probabilities.append(fake_prob)
                    true_labels.append(true_label)
                else:
                    logger.warning(f"Invalid result format from {model.model_name} for {image_path}")
                    failed_images.append(image_path)
                    continue
                    
            except Exception as e:
                logger.error(f"Error evaluating {image_path} with {model.model_name}: {str(e)}")
                failed_images.append(image_path)
                continue
            
            # Progress logging - reduced frequency
            if (i + 1) % 100 == 0 or (i + 1) == len(test_subset):
                logger.info(f"  Processed {i + 1}/{len(test_subset)} images")
        
        evaluation_time = time.time() - start_time
        
        if not predictions:
            logger.error(f"No successful predictions from {model.model_name}")
            return {
                "model_name": model.model_name,
                "error": "No successful predictions",
                "failed_images_count": len(failed_images)
            }
        
        # Calculate core metrics
        try:
            accuracy = accuracy_score(true_labels, predictions)
            
            # Core binary classification metrics
            precision = precision_score(true_labels, predictions, average='binary', zero_division=0)
            recall = recall_score(true_labels, predictions, average='binary', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='binary', zero_division=0)
            
            # For AUC, we need to convert fake probabilities to real probabilities
            real_probabilities = [1 - p for p in probabilities]
            
            # Handle AUC calculation when only one class is present
            try:
                auc = roc_auc_score(true_labels, real_probabilities)
            except ValueError:
                # Only one class present, set AUC to 0.5 (random performance)
                auc = 0.5
                logger.warning(f"Only one class present in predictions for {model.model_name}, setting AUC to 0.5")
            
            # Confusion matrix - handle cases where not all classes are present
            cm = confusion_matrix(true_labels, predictions, labels=[0, 1])  # Ensure both classes
            
            # Handle different confusion matrix shapes
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                # Only one class predicted and one class in true labels
                if set(true_labels) == {0} and set(predictions) == {0}:
                    # All fake, predicted all fake
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                elif set(true_labels) == {1} and set(predictions) == {1}:
                    # All real, predicted all real
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                else:
                    # Mixed true labels but single prediction class
                    if 0 in predictions:  # Predicted all fake
                        tn = sum(1 for t in true_labels if t == 0)
                        fp = 0
                        fn = sum(1 for t in true_labels if t == 1)
                        tp = 0
                    else:  # Predicted all real
                        tn = 0
                        fp = sum(1 for t in true_labels if t == 0)
                        fn = 0
                        tp = sum(1 for t in true_labels if t == 1)
            else:
                # Fallback: manually calculate from predictions and true labels
                tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
                fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
                fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
                tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
            
            metrics = {
                "model_name": model.model_name,
                # Core metrics
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc": float(auc),
                # Confusion matrix components
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                # Evaluation metadata
                "total_predictions": len(predictions),
                "failed_images_count": len(failed_images),
                "evaluation_time_seconds": float(evaluation_time),
                "images_per_second": float(len(predictions) / evaluation_time) if evaluation_time > 0 else 0
            }
            
            logger.info(f"  Results: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
            print_confusion_matrix(model.model_name, tn, fp, fn, tp, accuracy, precision, recall, f1, auc)
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model.model_name}: {str(e)}")
            return {
                "model_name": model.model_name,
                "error": f"Metrics calculation failed: {str(e)}",
                "total_predictions": len(predictions),
                "failed_images_count": len(failed_images)
            }
    
    def evaluate_ensemble(self, ensemble, max_images: int = None) -> Dict[str, Any]:
        """Evaluate the ensemble model.
        
        Args:
            ensemble: Ensemble model to evaluate
            max_images: Maximum number of images to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("=" * 60)
        logger.info("EVALUATING ENSEMBLE MODEL")
        logger.info("=" * 60)
        
        # First evaluate individual models
        individual_results = []
        for model in ensemble.models:
            result = self.evaluate_model(model, max_images)
            individual_results.append(result)
        
        # Calculate model weights based on individual performance
        weights = self._calculate_model_weights(individual_results)
        
        # Print current ensemble weights (before applying new weights)
        print("\n" + "="*80)
        print("BEFORE WEIGHT APPLICATION:")
        print("="*80)
        self._print_ensemble_weights(ensemble)
        
        # Apply the calculated weights to the ensemble
        if hasattr(ensemble, 'set_model_weights') and weights:
            logger.info("Applying calculated weights to ensemble for evaluation...")
            ensemble.set_model_weights(weights)
            
            # Ensure weighted voting is enabled
            if not getattr(ensemble, 'use_weighted_voting', False):
                ensemble.enable_weighted_voting(True)
                logger.info("Enabled weighted voting for ensemble evaluation")
            
            print("\n" + "="*80)
            print("APPLIED WEIGHTS:")
            print("="*80)
            for name, weight in weights.items():
                print(f"  {name}: {weight:.4f}")
            print("="*80)
        
        # Now evaluate ensemble WITH the applied weights
        logger.info("Evaluating ensemble with performance-based weights...")
        ensemble_result = self.evaluate_model(ensemble, max_images)
        
        # Print summary confusion matrix table (now with weighted ensemble results)
        self._print_summary_confusion_matrices(individual_results, ensemble_result)
        
        return {
            "ensemble_metrics": ensemble_result,
            "individual_results": individual_results,  # Fixed key name to match model_loader expectation
            "model_weights": weights,
            "evaluation_summary": self._create_evaluation_summary(individual_results, ensemble_result)
        }
    
    def _print_summary_confusion_matrices(self, individual_results: List[Dict], ensemble_result: Dict):
        """Print a summary table of all confusion matrices with evaluation metrics."""
        print("\n" + "="*100)
        print("CONFUSION MATRIX SUMMARY - ALL MODELS")
        print("="*100)
        print(f"{'Model Name':<30} {'TN':<5} {'FP':<5} {'FN':<5} {'TP':<5} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'AUC':<6}")
        print("-"*100)
        
        # Print individual models
        for result in individual_results:
            if "error" not in result:
                name = result["model_name"][:29]  # Truncate long names
                tn = result.get("true_negatives", 0)
                fp = result.get("false_positives", 0)
                fn = result.get("false_negatives", 0)
                tp = result.get("true_positives", 0)
                acc = result.get("accuracy", 0)
                prec = result.get("precision", 0)
                rec = result.get("recall", 0)
                f1 = result.get("f1_score", 0)
                auc = result.get("auc", 0)
                
                print(f"{name:<30} {tn:<5} {fp:<5} {fn:<5} {tp:<5} {acc:<6.3f} {prec:<6.3f} {rec:<6.3f} {f1:<6.3f} {auc:<6.3f}")
        
        # Print ensemble result
        if "error" not in ensemble_result:
            print("-"*100)
            name = "ENSEMBLE"
            tn = ensemble_result.get("true_negatives", 0)
            fp = ensemble_result.get("false_positives", 0)
            fn = ensemble_result.get("false_negatives", 0)
            tp = ensemble_result.get("true_positives", 0)
            acc = ensemble_result.get("accuracy", 0)
            prec = ensemble_result.get("precision", 0)
            rec = ensemble_result.get("recall", 0)
            f1 = ensemble_result.get("f1_score", 0)
            auc = ensemble_result.get("auc", 0)
            
            print(f"{name:<30} {tn:<5} {fp:<5} {fn:<5} {tp:<5} {acc:<6.3f} {prec:<6.3f} {rec:<6.3f} {f1:<6.3f} {auc:<6.3f}")
        
        print("="*100)
        print("Legend:")
        print("  TN=True Negatives, FP=False Positives, FN=False Negatives, TP=True Positives")
        print("  Acc=Accuracy, Prec=Precision, Rec=Recall, F1=F1-Score, AUC=Area Under Curve")
        print("="*100 + "\n")
    
    def _print_ensemble_weights(self, ensemble):
        """Print the current weights used by the ensemble model."""
        print("\n" + "="*80)
        print("CURRENT ENSEMBLE MODEL WEIGHTS")
        print("="*80)
        
        try:
            # Check if ensemble has weights attribute and weighted voting is enabled
            if hasattr(ensemble, 'weights') and ensemble.weights and getattr(ensemble, 'use_weighted_voting', False):
                print("Weights currently used by the ensemble (weighted voting enabled):")
                total_weight = sum(ensemble.weights.values())
                
                for i, model in enumerate(ensemble.models):
                    model_name = model.model_name
                    weight = ensemble.weights.get(model_name, 0.0)
                    percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                    print(f"  {model_name}: {weight:.4f} ({percentage:.1f}%)")
                
                print(f"Total weight: {total_weight:.4f}")
                print(f"Weighted voting: {'ENABLED' if getattr(ensemble, 'use_weighted_voting', False) else 'DISABLED'}")
            elif hasattr(ensemble, 'weights') and ensemble.weights:
                print("Calculated weights (but equal weights in use - weighted voting disabled):")
                total_weight = sum(ensemble.weights.values())
                
                for i, model in enumerate(ensemble.models):
                    model_name = model.model_name
                    weight = ensemble.weights.get(model_name, 0.0)
                    percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                    print(f"  Calculated: {model_name}: {weight:.4f} ({percentage:.1f}%)")
                
                print("Actually using equal weights for all models:")
                equal_weight = 1.0 / len(ensemble.models) if ensemble.models else 0.0
                
                for model in ensemble.models:
                    print(f"  In use: {model.model_name}: {equal_weight:.4f} ({100/len(ensemble.models):.1f}%)")
            else:
                print("Ensemble uses equal weights for all models:")
                equal_weight = 1.0 / len(ensemble.models) if ensemble.models else 0.0
                
                for model in ensemble.models:
                    print(f"  {model.model_name}: {equal_weight:.4f} ({100/len(ensemble.models):.1f}%)")
                
        except Exception as e:
            print(f"Could not retrieve ensemble weights: {str(e)}")
            print("Assuming equal weights for all models")
        
        print("="*80 + "\n")
    
    def _calculate_model_weights(self, individual_results: List[Dict]) -> Dict[str, float]:
        """Calculate model weights based on performance metrics.
        
        Args:
            individual_results: List of individual model evaluation results
            
        Returns:
            Dictionary mapping model names to weights
        """
        weights = {}
        
        # Use accuracy and AUC as the primary metrics for weighting
        accuracy_scores = []
        auc_scores = []
        model_names = []
        
        for result in individual_results:
            if "error" not in result and "accuracy" in result and "auc" in result:
                accuracy_scores.append(result["accuracy"])
                auc_scores.append(result["auc"])
                model_names.append(result["model_name"])
        
        logger.info(f"DEBUG: Found {len(model_names)} models for weight calculation: {model_names}")
        if len(set(model_names)) != len(model_names):
            logger.warning(f"DEBUG: Duplicate model names detected! Unique names: {list(set(model_names))}")
            # Handle duplicates by averaging their scores
            unique_names = []
            unique_acc = []
            unique_auc = []
            
            for unique_name in set(model_names):
                indices = [i for i, name in enumerate(model_names) if name == unique_name]
                avg_acc = np.mean([accuracy_scores[i] for i in indices])
                avg_auc = np.mean([auc_scores[i] for i in indices])
                unique_names.append(unique_name)
                unique_acc.append(avg_acc)
                unique_auc.append(avg_auc)
                logger.info(f"  Averaged {unique_name}: Acc={avg_acc:.3f}, AUC={avg_auc:.3f}")
            
            model_names = unique_names
            accuracy_scores = unique_acc
            auc_scores = unique_auc
        
        if not accuracy_scores:
            logger.warning("No valid accuracy scores found, using equal weights")
            return {result["model_name"]: 1.0/len(individual_results) 
                   for result in individual_results}
        
        # Combine accuracy and AUC for more robust weighting
        # Convert to numpy arrays for easier manipulation
        acc_array = np.array(accuracy_scores)
        auc_array = np.array(auc_scores)
        
        # Use a gentler weighting approach that doesn't over-penalize small differences
        # Method 1: Power-based weighting (less aggressive than exponential)
        combined_scores = 0.5 * acc_array + 0.5 * auc_array
        
        # Use power function instead of exponential for gentler scaling
        # Power of 2-3 gives reasonable emphasis without being too extreme
        power_factor = 2.0
        powered_scores = np.power(combined_scores, power_factor)
        
        # Normalize to get weights
        weights_array = powered_scores / np.sum(powered_scores)
        
        # Apply minimum weight threshold to prevent any model from being completely ignored
        min_weight = 0.05  # Ensure each model gets at least 5% weight
        
        # Redistribute weights if any are below minimum
        below_min = weights_array < min_weight
        if np.any(below_min):
            # Set minimum weights
            adjusted_weights = np.maximum(weights_array, min_weight)
            
            # Calculate excess weight to redistribute
            excess_weight = np.sum(adjusted_weights) - 1.0
            
            # Redistribute excess from high-weight models
            above_min = ~below_min
            if np.any(above_min):
                # Proportionally reduce weights above minimum
                reduction_factor = max(0, 1.0 - excess_weight / np.sum(adjusted_weights[above_min]))
                adjusted_weights[above_min] *= reduction_factor
                
                # Renormalize
                adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
                weights_array = adjusted_weights
        
        for name, weight in zip(model_names, weights_array):
            weights[name] = float(weight)
        
        logger.info("Model weights calculated using accuracy + AUC:")
        for name, weight in weights.items():
            # Find the corresponding result for additional info
            result = next((r for r in individual_results if r["model_name"] == name), {})
            acc = result.get("accuracy", 0)
            auc = result.get("auc", 0)
            logger.info(f"  {name}: {weight:.3f} (Acc={acc:.3f}, AUC={auc:.3f})")
        
        return weights
    
    def _create_evaluation_summary(self, individual_results: List[Dict], 
                                 ensemble_result: Dict) -> Dict[str, Any]:
        """Create a summary of the evaluation results.
        
        Args:
            individual_results: Results from individual models
            ensemble_result: Results from ensemble model
            
        Returns:
            Summary dictionary
        """
        # Find best individual model
        best_model = None
        best_f1 = -1
        
        valid_results = [r for r in individual_results if "error" not in r]
        
        if valid_results:
            for result in valid_results:
                if result.get("f1_score", 0) > best_f1:
                    best_f1 = result["f1_score"]
                    best_model = result["model_name"]
        
        # Calculate improvements
        ensemble_f1 = ensemble_result.get("f1_score", 0)
        f1_improvement = ensemble_f1 - best_f1 if best_f1 > 0 else 0
        
        summary = {
            "total_models_evaluated": len(individual_results),
            "successful_models": len(valid_results),
            "best_individual_model": best_model,
            "best_individual_f1": float(best_f1) if best_f1 > 0 else None,
            "ensemble_f1": float(ensemble_f1),
            "f1_improvement": float(f1_improvement),
            "ensemble_better_than_best": f1_improvement > 0
        }
        
        return summary
    
    def save_results(self, results: Dict, filename: str = None):
        """Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary to save
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = os.path.join(current_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")


def run_evaluation(max_images_per_model: int = None, ensemble=None):
    """Run full model evaluation.
    
    Args:
        max_images_per_model: Limit number of images per model (for testing)
        ensemble: Pre-loaded ensemble model (optional, will load if not provided)
    """
    logger.info("Starting model evaluation...")
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Load models if not provided
        if ensemble is None:
            logger.info("Loading models for evaluation...")
            # Create a simple ensemble without evaluation to avoid circular dependency
            from models.ensemble_model import EnsembleModel
            from models.srm_cnn_model import SRMCNNModel
            from models.dct_model import DCTModel
            from models.dct_ae_model import DCTAutoencoderModel
            import os
            
            ensemble = EnsembleModel()
            models_dir = os.path.join(current_dir, "models")
            
            # Find model files
            model_files = []
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(('.keras', '.h5')) and not file.startswith("autoencoder_L"):
                        full_path = os.path.join(root, file)
                        model_files.append(full_path)
            
            logger.info(f"Found {len(model_files)} model files for evaluation")
            
            # Load models directly without calling load_models to avoid circular dependency
            for model_path in model_files:
                filename = os.path.basename(model_path)
                try:
                    # Try different model types
                    loaded = False
                    
                    # Try SRM-CNN first
                    if 'srm' in filename.lower() or 'cnn' in filename.lower():
                        latent_size = 256 if '256' in filename else 128
                        model = SRMCNNModel(latent_size=latent_size)
                        model.model_name = f"SRM-CNN-L{latent_size} ({filename})"
                        if model.load(model_path):
                            ensemble.add_model(model)
                            loaded = True
                    
                    # Try DCT if not loaded
                    if not loaded:
                        model = DCTModel(block_size=8, num_coefficients=64)
                        model.model_name = f"DCT ({filename})"
                        if model.load(model_path):
                            ensemble.add_model(model)
                            loaded = True
                    
                    if loaded:
                        logger.info(f"Loaded {filename} for evaluation")
                    else:
                        logger.warning(f"Failed to load {filename}")
                        
                except Exception as e:
                    logger.warning(f"Error loading {filename}: {str(e)}")
        
        if not ensemble or not ensemble.models:
            logger.error("No models available for evaluation")
            return {}
        
        # Run evaluation
        results = evaluator.evaluate_ensemble(ensemble, max_images_per_model)
        
        # Save results
        evaluator.save_results(results)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        
        summary = results["evaluation_summary"]
        logger.info(f"Models evaluated: {summary['successful_models']}/{summary['total_models_evaluated']}")
        if summary["best_individual_model"]:
            logger.info(f"Best individual model: {summary['best_individual_model']} (F1: {summary['best_individual_f1']:.3f})")
        logger.info(f"Ensemble F1 score: {summary['ensemble_f1']:.3f}")
        if summary["ensemble_better_than_best"]:
            logger.info(f"Ensemble improvement: +{summary['f1_improvement']:.3f} F1 score")
        else:
            logger.info("Ensemble did not improve over best individual model")
        
        # Return model weights for the ensemble to use
        return results.get("model_weights", {})
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.exception("Stack trace:")
        return {}


if __name__ == "__main__":
    run_evaluation() 