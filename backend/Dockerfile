FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - use specific TensorFlow version compatible with the models
RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    numpy \
    pillow \
    tensorflow==2.18.0 \
    keras==3.8.0 \
    h5py \
    werkzeug \
    opencv-python \
    scipy \
    scikit-learn \
    matplotlib

# Create necessary directories
RUN mkdir -p /app/uploads /app/models

# Copy the application code
COPY . /app/

# Set TensorFlow logging level to reduce noise
ENV TF_CPP_MIN_LOG_LEVEL=1

# Expose API port
EXPOSE 5000

# Start the application
CMD ["python", "app.py"] 