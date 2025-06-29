# Model Evaluation and Weighted Ensemble System

This system automatically evaluates deepfake detection models and implements performance-based weighted ensemble predictions.

## Overview

The system includes:

1. **Automatic Model Evaluation**: When the backend starts, models are evaluated on test images
2. **Performance-Based Weighting**: Better-performing models get higher weights in ensemble decisions
3. **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, AUC, and more
4. **Flexible Evaluation**: Support for quick (limited images) and full evaluation

## Files

### Core Files

- `evaluate_models.py` - Main evaluation logic and ModelEvaluator class
- `run_full_evaluation.py` - Standalone script for comprehensive evaluation
- `models/ensemble_model.py` - Modified to support weighted voting
- `model_loader.py` - Modified to run evaluation on startup

### Test Data Structure

```
backend/test_images/
├── fake/          # 640 fake images (label = 0)
└── real/          # 452 real images (label = 1)
```

## How It Works

### 1. Automatic Startup Evaluation

When the backend starts:
1. Models are loaded into the ensemble
2. Quick evaluation runs (100 images per model by default)
3. Model weights are calculated based on F1-scores
4. Ensemble switches to weighted voting mode

### 2. Weighted Voting Algorithm

Instead of majority voting, the ensemble now uses:
- **Weight Calculation**: Based on F1-score using softmax normalization
- **Weighted Prediction**: Each model's prediction is weighted by its performance
- **Final Decision**: Weighted average probability determines the final prediction

### 3. Evaluation Metrics

For each model, the system calculates:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

## Usage

### Automatic Evaluation (Default)
The system runs automatically when the backend starts. Check the logs for evaluation results.

### Manual Evaluation

#### Quick Evaluation (50 images per model)
```bash
cd backend
python run_full_evaluation.py --quick
```

#### Full Evaluation (all images)
```bash
cd backend
python run_full_evaluation.py --full
```

#### Custom Evaluation
```python
from evaluate_models import run_evaluation

# Evaluate with 200 images per model
weights = run_evaluation(max_images_per_model=200)
```

## Configuration

### Disabling Weighted Voting
If you want to use majority voting instead:

```python
# In your code after loading models
ensemble.enable_weighted_voting(False)
```

### Customizing Weights
You can manually set model weights:

```python
custom_weights = {
    "Model1": 0.4,
    "Model2": 0.6
}
ensemble.set_model_weights(custom_weights)
```

## Performance Benefits

The weighted ensemble typically provides:
- **Better Accuracy**: Leverages strengths of best-performing models
- **Improved Robustness**: Reduces impact of poorly-performing models
- **Adaptive Behavior**: Automatically adjusts to model performance changes

## Monitoring

### Logs
The system provides detailed logging for:
- Individual model performance
- Weight calculations
- Ensemble decision process
- Evaluation progress

### Saved Results
Evaluation results are automatically saved as JSON files with timestamps:
- `evaluation_results_YYYYMMDD_HHMMSS.json`

### Example Log Output
```
2024-01-15 10:30:15 - Model evaluation starting...
2024-01-15 10:32:20 - SRM-CNN-L256: Accuracy=0.892, F1=0.885, AUC=0.945
2024-01-15 10:34:15 - DCT-AE-New: Accuracy=0.856, F1=0.848, AUC=0.921
2024-01-15 10:36:10 - Model weights calculated:
2024-01-15 10:36:10 -   SRM-CNN-L256: 0.654
2024-01-15 10:36:10 -   DCT-AE-New: 0.346
2024-01-15 10:36:10 - Weighted voting enabled
```

## Troubleshooting

### Common Issues

1. **Missing test_images directory**: Ensure `backend/test_images/fake/` and `backend/test_images/real/` exist
2. **Evaluation fails**: Check that test images are in supported formats (jpg, jpeg, png, bmp)
3. **No improvement**: If ensemble doesn't improve over individual models, check model diversity

### Performance Tips

1. **Quick Startup**: Use limited images (100) for faster server startup
2. **Full Analysis**: Run full evaluation offline for comprehensive analysis
3. **Custom Limits**: Adjust `max_images_per_model` based on your needs

## Integration

The system integrates seamlessly with the existing API:
- All existing endpoints continue to work
- Predictions are now performance-weighted
- No changes needed to client code

The weighted ensemble provides better accuracy while maintaining the same API interface. 