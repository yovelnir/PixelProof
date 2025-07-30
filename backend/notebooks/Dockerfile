# ----- Base image with CUDA + cuDNN + TensorFlow 2.18 -----
    FROM tensorflow/tensorflow:2.18.0-gpu

    # ----- OS packages you usually miss -----
    RUN apt-get update && \
        apt-get install -y --no-install-recommends git nano && \
        rm -rf /var/lib/apt/lists/*
    
    # ----- Python deps (add yours in requirements.txt) -----
    COPY requirements.txt /tmp/req.txt
    RUN pip install --no-cache-dir -r /tmp/req.txt \
        && rm /tmp/req.txt
    
    # ----- Working directory -----
    WORKDIR /workspace
    
    # ----- Jupyter config -----
    ENV JUPYTER_TOKEN=$JUPYTER_TOKEN \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        PYTHONUNBUFFERED=1
    
    EXPOSE 8888
    
    CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]