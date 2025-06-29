 # PixelProof - Advanced Deepfake Detection

PixelProof is a sophisticated web application that uses multiple AI models in an ensemble approach to detect deepfake images with high accuracy. The system features advanced model evaluation, weighted voting, and performance-optimized caching.

## Features

### 🔍 **Detection Capabilities**
- **Multi-model ensemble**: SRM-CNN and DCT models working together
- **Weighted voting system**: Performance-based model weighting
- **High accuracy**: Optimized ensemble achieving 83.8% accuracy
- **Confidence scoring**: Advanced confidence calculation with model agreement analysis
- **Real-time analysis**: Fast image processing with progress tracking

### 🎨 **User Interface**
- **Modern React frontend**: Built with Next.js and Tailwind CSS
- **Dark/Light theme**: Automatic theme detection with manual toggle
- **Responsive design**: Works on desktop and mobile devices
- **Interactive animations**: Smooth transitions and visual feedback
- **Detailed results**: Metadata extraction and analysis breakdown

### 🤖 **AI Models**
- **SRM-CNN Models**: Spatial Rich Model with CNN (128/256 latent dimensions)
- **DCT Models**: Discrete Cosine Transform with Autoencoder (128/256 dimensions)
- **Ensemble Intelligence**: Weighted combination with performance-based optimization
- **Model Evaluation**: Automated evaluation with metric-based weight calculation

### 🐳 **Deployment**
- **Docker containerization**: Production-ready deployment
- **Docker Compose**: Single-command deployment
- **Model caching**: Performance weights cached for fast startup
- **GPU support**: NVIDIA GPU acceleration support

## Project Structure

```
PixelProof/
├── frontend/pixelproof/              # Next.js frontend application
│   ├── app/
│   │   ├── components/               # React components
│   │   │   ├── ImageUpload.js       # Drag & drop upload
│   │   │   ├── ResultShowcase.js    # Results display
│   │   │   ├── ProgressBar.js       # Upload progress
│   │   │   └── Toast.js             # Notifications
│   │   ├── assets/                  # Logo and images
│   │   ├── globals.css              # Global styles
│   │   ├── layout.js                # App layout
│   │   └── page.js                  # Main page
│   ├── public/                      # Static assets
│   ├── package.json                 # Dependencies
│   └── tailwind.config.js           # Tailwind configuration
├── backend/
│   ├── models/                      # AI model files
│   │   ├── *.keras                  # Trained model files
│   │   ├── ae/                      # Autoencoder models
│   │   ├── base_model.py            # Abstract base class
│   │   ├── srm_cnn_model.py         # SRM-CNN implementation
│   │   ├── dct_model.py             # DCT model implementation
│   │   └── ensemble_model.py        # Ensemble logic
│   ├── utils/                       # Utility functions
│   │   ├── image_processing.py      # Image preprocessing
│   │   └── model_utils.py           # Model utilities
│   ├── docs/                        # Documentation
│   ├── tests/                       # Unit tests
│   ├── app.py                       # Flask API server
│   ├── model_loader.py              # Model loading logic
│   ├── evaluate_models.py           # Model evaluation system
│   ├── weight_cache.py              # Performance caching
│   ├── run_backend.py               # Enhanced backend runner
│   ├── requirements.txt             # Python dependencies
│   └── Dockerfile                   # Container configuration
├── docker-compose.yml               # Multi-container setup
├── run_backend.sh                   # Backend startup script
└── README.md                        # This file
```

## Quick Start

### 🛠️ **Development Setup**

#### Backend Setup

**Option 1: Docker (Recommended)**
```bash
cd backend
docker build -t pixelproof-backend .
docker run -p 5000:5000 -v $(pwd):/app pixelproof-backend
```

**Option 2: Native Python**
```bash
cd backend
# Install dependencies (requires TensorFlow, etc.)
pip install -r requirements.txt
# Note: May require system dependencies for OpenCV
python app.py
```

#### Frontend Setup
```bash
cd frontend/pixelproof
npm install
npm run dev
```

## Advanced Backend Features

### 🎯 **Model Evaluation System**

The backend includes a sophisticated evaluation system that:
- Tests models on validation datasets
- Calculates accuracy, precision, recall, F1-score, and AUC
- Automatically determines optimal model weights
- Caches results for fast startup

```bash
# Run evaluation and save optimal weights
./run_backend.sh --evaluate-only

# Check cached weights
./run_backend.sh --cache-info

# Force fresh evaluation
./run_backend.sh --force-evaluation
```

### ⚡ **Performance Optimization**

- **Weight Caching**: Model weights cached as JSON for instant loading
- **Lazy Loading**: Models loaded on-demand
- **GPU Acceleration**: NVIDIA GPU support via Docker
- **Memory Management**: Efficient model weight distribution

### 🔧 **Backend Runner Options**

```bash
./run_backend.sh                    # Standard mode with auto-evaluation
./run_backend.sh --serve-only       # Use cached weights only (fastest)
./run_backend.sh --docker           # Docker deployment
./run_backend.sh --evaluate-only    # Evaluation mode only
./run_backend.sh --force-evaluation # Force fresh evaluation
./run_backend.sh --cache-info       # Show cache information
./run_backend.sh --clear-cache      # Clear weight cache
./run_backend.sh --help             # Show all options
```

## API Documentation

### **POST /api/analyze**
Analyzes an uploaded image for deepfake detection.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` file (JPEG, PNG, etc.)

**Response:**
```json
{
  "prediction": "real|fake",
  "probability": 0.123,
  "confidence": 0.856,
  "models_used": 4,
  "vote_distribution": "3 real, 1 fake",
  "processing_time": 1.234,
  "model_details": {
    "ensemble_weights": {
      "SRM-CNN-L256": 0.2094,
      "DCT-AE-L256": 0.2960,
      "SRM-CNN-L128": 0.2054,
      "DCT-AE-L128": 0.2892
    }
  }
}
```

### **GET /api/status**
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "API is running",
  "model_loaded": true,
  "models_available": 4,
  "model_names": ["SRM-CNN-L256", "DCT-AE-L256", "SRM-CNN-L128", "DCT-AE-L128"]
}
```

## Model Performance

The current ensemble achieves the following performance metrics:

| Metric | Score |
|--------|-------|
| **Accuracy** | 83.8% |
| **Precision** | 86.0% |
| **Recall** | 84.3% |
| **F1-Score** | 85.0% |
| **AUC** | 89.3% |

### Model Weights Distribution
- **DCT-AE-L256**: 29.6% (highest weight)
- **DCT-AE-L128**: 28.9%
- **SRM-CNN-L256**: 20.9%
- **SRM-CNN-L128**: 20.5%

## Technology Stack

### **Frontend**
- **Framework**: Next.js 14 (React 18)
- **Styling**: Tailwind CSS
- **Language**: JavaScript (ES2022)
- **Features**: SSR, Image optimization, Dark mode

### **Backend**
- **Framework**: Flask
- **AI/ML**: TensorFlow 2.18, Keras 3.8
- **Image Processing**: OpenCV, PIL
- **Data Science**: NumPy, SciPy, scikit-learn
- **Containerization**: Docker

### **Models**
- **SRM-CNN**: Spatial Rich Model with Convolutional Neural Network
- **DCT**: Discrete Cosine Transform with Autoencoder
- **Ensemble**: Weighted voting with performance-based optimization

## Troubleshooting

### Common Issues

1. **"No module named 'flask'" Error**
   - Solution: Use Docker deployment (recommended)
   - Alternative: Install Python dependencies locally

2. **Docker Compose Version Issues**
   - Try: `docker compose up` (newer syntax)
   - Or use: `./run_backend.sh --docker`

3. **Model Loading Failures**
   - Check: TensorFlow version compatibility
   - Ensure: Model files are present in `backend/models/`

### Performance Tips

- Use `--serve-only` flag for fastest startup with cached weights
- Enable GPU acceleration in Docker for better performance
- Monitor memory usage with multiple large models
