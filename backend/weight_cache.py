#!/usr/bin/env python3
"""
Weight cache utility for saving and loading model evaluation weights.
"""
import os
import json
import time
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_FILE = "model_weights_cache.json"

def save_model_weights(weights: Dict[str, float], 
                      evaluation_results: Dict[str, Any] = None,
                      cache_file: str = None) -> bool:
    """Save model weights and evaluation results to JSON cache file.
    
    Args:
        weights: Dictionary mapping model names to weights
        evaluation_results: Full evaluation results (optional)
        cache_file: Path to cache file (optional, uses default if None)
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    if cache_file is None:
        cache_file = DEFAULT_CACHE_FILE
    
    try:
        # Prepare cache data
        cache_data = {
            "timestamp": time.time(),
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_weights": weights,
            "total_models": len(weights),
            "weight_sum": sum(weights.values()),
            "evaluation_summary": {}
        }
        
        # Add evaluation summary if provided
        if evaluation_results:
            if "ensemble_metrics" in evaluation_results:
                ensemble_metrics = evaluation_results["ensemble_metrics"]
                cache_data["evaluation_summary"] = {
                    "ensemble_accuracy": ensemble_metrics.get("accuracy"),
                    "ensemble_f1": ensemble_metrics.get("f1_score"),
                    "ensemble_auc": ensemble_metrics.get("auc"),
                    "models_evaluated": ensemble_metrics.get("total_predictions", 0)
                }
            
            # Add individual model performance
            if "individual_results" in evaluation_results:
                individual_performance = {}
                for result in evaluation_results["individual_results"]:
                    if "error" not in result:
                        model_name = result["model_name"]
                        individual_performance[model_name] = {
                            "accuracy": result.get("accuracy"),
                            "f1_score": result.get("f1_score"),
                            "auc": result.get("auc"),
                            "weight": weights.get(model_name, 0.0)
                        }
                cache_data["individual_performance"] = individual_performance
        
        # Write to cache file
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Model weights saved to cache: {cache_path}")
        logger.info(f"Cached weights for {len(weights)} models")
        for model_name, weight in weights.items():
            logger.info(f"  {model_name}: {weight:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model weights cache: {str(e)}")
        return False

def load_model_weights(cache_file: str = None, 
                      max_age_hours: float = 24.0) -> Optional[Dict[str, float]]:
    """Load model weights from JSON cache file.
    
    Args:
        cache_file: Path to cache file (optional, uses default if None)
        max_age_hours: Maximum age of cache in hours (default 24 hours)
        
    Returns:
        Dictionary mapping model names to weights, or None if cache invalid/missing
    """
    if cache_file is None:
        cache_file = DEFAULT_CACHE_FILE
    
    cache_path = Path(cache_file)
    
    # Check if cache file exists
    if not cache_path.exists():
        logger.info(f"No weight cache found at {cache_path}")
        return None
    
    try:
        # Load cache data
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Check cache age
        cache_timestamp = cache_data.get("timestamp", 0)
        cache_age_hours = (time.time() - cache_timestamp) / 3600
        
        if cache_age_hours > max_age_hours:
            logger.warning(f"Weight cache is too old ({cache_age_hours:.1f} hours), ignoring")
            return None
        
        # Extract weights
        weights = cache_data.get("model_weights", {})
        
        if not weights:
            logger.warning("No weights found in cache file")
            return None
        
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            logger.warning(f"Cache weights don't sum to 1.0 (sum={total_weight:.3f}), normalizing")
            # Normalize weights
            if total_weight > 0:
                weights = {name: weight / total_weight for name, weight in weights.items()}
        
        logger.info(f"Loaded weights from cache (age: {cache_age_hours:.1f}h)")
        logger.info(f"Cached weights for {len(weights)} models:")
        for model_name, weight in weights.items():
            logger.info(f"  {model_name}: {weight:.4f}")
        
        return weights
        
    except Exception as e:
        logger.error(f"Failed to load model weights cache: {str(e)}")
        return None

def get_cache_info(cache_file: str = None) -> Optional[Dict[str, Any]]:
    """Get information about the cache file without loading weights.
    
    Args:
        cache_file: Path to cache file (optional, uses default if None)
        
    Returns:
        Dictionary with cache information, or None if cache doesn't exist
    """
    if cache_file is None:
        cache_file = DEFAULT_CACHE_FILE
    
    cache_path = Path(cache_file)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        cache_timestamp = cache_data.get("timestamp", 0)
        cache_age_hours = (time.time() - cache_timestamp) / 3600
        
        info = {
            "cache_file": str(cache_path),
            "timestamp": cache_timestamp,
            "timestamp_human": cache_data.get("timestamp_human", "Unknown"),
            "age_hours": cache_age_hours,
            "total_models": cache_data.get("total_models", 0),
            "weight_sum": cache_data.get("weight_sum", 0),
            "has_evaluation_summary": "evaluation_summary" in cache_data,
            "has_individual_performance": "individual_performance" in cache_data
        }
        
        if "evaluation_summary" in cache_data:
            info["ensemble_accuracy"] = cache_data["evaluation_summary"].get("ensemble_accuracy")
            info["ensemble_f1"] = cache_data["evaluation_summary"].get("ensemble_f1")
            info["ensemble_auc"] = cache_data["evaluation_summary"].get("ensemble_auc")
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get cache info: {str(e)}")
        return None

def clear_cache(cache_file: str = None) -> bool:
    """Remove the cache file.
    
    Args:
        cache_file: Path to cache file (optional, uses default if None)
        
    Returns:
        bool: True if cleared successfully, False otherwise
    """
    if cache_file is None:
        cache_file = DEFAULT_CACHE_FILE
    
    cache_path = Path(cache_file)
    
    try:
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleared weight cache: {cache_path}")
            return True
        else:
            logger.info(f"No cache file to clear: {cache_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        return False 