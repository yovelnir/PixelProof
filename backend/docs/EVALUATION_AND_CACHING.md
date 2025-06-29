# Model Evaluation and Weight Caching System

This document describes the enhanced model loading system with automatic evaluation and weight caching support.

## Overview

The PixelProof backend now supports intelligent model weight caching to dramatically improve startup times while maintaining optimal performance. The system automatically evaluates model performance and caches the results for future use.

## Key Features

- **Automatic Weight Calculation**: Models are evaluated and weighted based on accuracy + AUC performance
- **Intelligent Caching**: Weights are saved to JSON cache for fast startup
- **Cache Validation**: Automatic cache validation with age limits and model matching
- **Flexible Modes**: Multiple operation modes for different use cases

## Quick Start

### 1. First Run (with Evaluation)
```bash
# Run evaluation and start server
python backend/app.py

# Or using the runner script
python backend/run_backend.py
```

### 2. Fast Startup (using cached weights)
```bash
# Start server with cached weights (if available)
python backend/run_backend.py --serve-only
```

### 3. Force Re-evaluation
```bash
# Force new evaluation and update cache
python backend/run_backend.py --force-evaluation
```

## Command Line Options

### Backend App (`app.py`)
```bash
python backend/app.py [OPTIONS]

Options:
  --evaluate              Run model evaluation and update weights cache
  --force-evaluation      Force model evaluation even if cache exists
  --no-cache             Disable weight caching (always evaluate)
  --cache-info           Show cache information and exit
  --clear-cache          Clear weight cache and exit
  --port PORT            Port to run the server on (default: 5000)
  --host HOST            Host to bind the server to (default: 0.0.0.0)
```

### Backend Runner (`run_backend.py`)
```bash
python backend/run_backend.py [OPTIONS]

Mode Selection:
  --evaluate-only        Run model evaluation and save weights, then exit
  --serve-only          Start server using cached weights (no evaluation)
  --docker              Run using Docker container

Evaluation Options:
  --force-evaluation    Force model evaluation even if cache exists
  --no-cache           Disable weight caching

Cache Management:
  --cache-info         Show cache information and exit
  --clear-cache        Clear weight cache and exit

Server Options:
  --port PORT          Port to run the server on (default: 5000)
  --host HOST          Host to bind the server to (default: 0.0.0.0)
```

## Usage Examples

### Development Workflow

1. **Initial Setup**:
   ```bash
   # Run evaluation once to create cache
   python backend/run_backend.py --evaluate-only
   ```

2. **Fast Development Cycles**:
   ```bash
   # Start server quickly using cached weights
   python backend/run_backend.py --serve-only
   ```

3. **Model Updates**:
   ```bash
   # Re-evaluate after model changes
   python backend/run_backend.py --force-evaluation
   ```

### Production Deployment

1. **One-time Evaluation**:
   ```bash
   # Pre-compute optimal weights
   python backend/app.py --evaluate
   ```

2. **Fast Production Startup**:
   ```bash
   # Start with cached weights
   python backend/app.py
   ```

### Cache Management

```bash
# Check cache status
python backend/app.py --cache-info

# Clear cache
python backend/app.py --clear-cache

# Force fresh evaluation
python backend/app.py --force-evaluation
```

## Cache File Format

The cache is stored in `backend/model_weights_cache.json`:

```json
{
  "timestamp": 1703123456.789,
  "timestamp_human": "2024-06-21 15:30:56",
  "model_weights": {
    "SRM-CNN-L128 (128_SRM_CNN.h5)": 0.4234,
    "DCT-AE-L256 (256new_best_dct_model.keras)": 0.2891,
    "DCT-AE-L128 (128new_best_dct_model.keras)": 0.1567,
    "SRM-CNN-L256 (256_SRM_CNN_model.keras)": 0.1308
  },
  "total_models": 4,
  "weight_sum": 1.0,
  "evaluation_summary": {
    "ensemble_accuracy": 0.910,
    "ensemble_f1": 0.908,
    "ensemble_auc": 0.969,
    "models_evaluated": 200
  },
  "individual_performance": {
    "SRM-CNN-L128 (128_SRM_CNN.h5)": {
      "accuracy": 0.865,
      "f1_score": 0.849,
      "auc": 0.966,
      "weight": 0.4234
    }
  }
}
```

## Performance Benefits

### Startup Time Comparison

| Mode | Startup Time | Use Case |
|------|-------------|----------|
| Full Evaluation | ~3-5 minutes | First run, model updates |
| Cached Weights | ~10-30 seconds | Development, production |
| No Cache | ~3-5 minutes | Always fresh evaluation |

### Weight Calculation Algorithm

The system uses a balanced approach for calculating model weights:

1. **Performance Metrics**: 50% accuracy + 50% AUC
2. **Gentle Scaling**: Power function (x²) instead of aggressive exponential
3. **Minimum Threshold**: Each model gets at least 5% weight
4. **Fair Distribution**: Better models get more weight without eliminating others

## Cache Validation

The cache system includes several validation mechanisms:

- **Age Limit**: Cache expires after 24 hours by default
- **Model Matching**: Cached models must match currently loaded models
- **Weight Validation**: Weights must sum to 1.0 (with auto-normalization)
- **Integrity Check**: JSON format and required fields validation

## Troubleshooting

### Cache Issues

```bash
# Problem: Cache seems outdated
# Solution: Clear and regenerate
python backend/app.py --clear-cache
python backend/app.py --force-evaluation

# Problem: Models don't match cache
# Solution: Cache will be automatically ignored, new evaluation will run

# Problem: Cache file corrupted
# Solution: Clear cache and regenerate
python backend/app.py --clear-cache
```

### Performance Issues

```bash
# Problem: Evaluation takes too long
# Solution: Check test images directory and model files

# Problem: Weights seem incorrect
# Solution: Check individual model performance
python backend/app.py --cache-info
```

## Integration with Docker

The Docker setup automatically handles caching:

```bash
# Docker with evaluation
python backend/run_backend.py --docker

# The cache file is preserved through volume mounts
# Location: backend/model_weights_cache.json
```

## Configuration

### Cache Settings

You can modify cache behavior in `backend/weight_cache.py`:

- `DEFAULT_CACHE_FILE`: Cache file location
- `max_age_hours`: Cache expiration time (default: 24 hours)
- `min_weight`: Minimum weight per model (default: 5%)

### Evaluation Settings

Model evaluation can be configured in `backend/evaluate_models.py`:

- Test images directory
- Evaluation metrics combination
- Weight calculation algorithm
- Performance thresholds

## Best Practices

1. **Development**: Use `--serve-only` for fast iterations
2. **CI/CD**: Run `--evaluate-only` in build pipeline
3. **Production**: Always use cached weights for fast startup
4. **Model Updates**: Force re-evaluation after model changes
5. **Monitoring**: Check cache info regularly for performance insights

## API Integration

The weight caching system is transparent to the API:

- All API endpoints work identically
- Weight calculation happens at startup
- Model predictions use optimal weights automatically
- No changes required in client code

## Backwards Compatibility

The system maintains full backwards compatibility:

- Default behavior includes automatic evaluation/caching
- Existing deployment scripts continue to work
- Docker containers work without changes
- Legacy model loading is preserved 