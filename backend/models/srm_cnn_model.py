import os
import sys
import numpy as np
import tensorflow as tf
from .base_model import BaseModel
import logging

# Set recursion limit higher (as you had)
sys.setrecursionlimit(3000)

# Configure logging
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SRM: pure‑TF kernel + apply (mirrors training numerics exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _build_srm_kernel():
    # 5 filters, each 5x5, identical to your training kernels
    SRM_FILTERS = np.array([
        # Filter 1: Laplacian-High Boost
        [[[0, 0, -1, 0, 0],
          [0, -1,  2, -1, 0],
          [-1, 2,  4,  2,-1],
          [0, -1,  2, -1, 0],
          [0,  0, -1,  0, 0]]],

        # Filter 2: Edge & Noise Enhancer
        [[[-1, 2, -2,  2, -1],
          [ 2,-6,  8, -6,  2],
          [-2, 8,-12,  8, -2],
          [ 2,-6,  8, -6,  2],
          [-1, 2, -2,  2, -1]]],

        # Filter 3: Diagonal Residual Capture
        [[[ 2, -1, 0, -1,  2],
          [-1, -2, 3, -2, -1],
          [ 0,  3, 0,  3,  0],
          [-1, -2, 3, -2, -1],
          [ 2, -1, 0, -1,  2]]],

        # Filter 4: Vertical Edge Residuals
        [[[0, 0, 0, 0, 0],
          [1,-2, 1,-2, 1],
          [0, 0, 0, 0, 0],
          [-1, 2,-1, 2,-1],
          [0, 0, 0, 0, 0]]],

        # Filter 5: High Frequency Noise Extractor
        [[[ 1, -4,  6, -4,  1],
          [ -4, 16,-24, 16, -4],
          [  6,-24, 36,-24,  6],
          [ -4, 16,-24, 16, -4],
          [  1, -4,  6, -4,  1]]],
    ], dtype=np.float32)
    # Convert to TF conv2d filter: (kh, kw, in_channels=1, out_channels=5)
    return tf.constant(np.transpose(SRM_FILTERS, (2, 3, 1, 0)), dtype=tf.float32)

_SRM_K = _build_srm_kernel()

@tf.function
def _srm_apply_rgb(img_bhwc: tf.Tensor) -> tf.Tensor:
    """Apply SRM to RGB: (B,H,W,3) -> (B,H,W,15)."""
    feats = []
    for c in tf.split(img_bhwc, 3, axis=-1):
        feats.append(tf.nn.conv2d(c, _SRM_K, strides=1, padding="SAME"))
    return tf.concat(feats, axis=-1)


class SRMCNNModel(BaseModel):
    """SRM‑CNN model for deepfake detection.

    Inference preprocessing is made identical to training:
      - tf.io.read_file → tf.image.decode_jpeg(3)
      - tf.image.resize(256,256) bilinear, antialias=False
      - float32 / 255.0
      - SRM (TF conv2d) → 15 channels
      - Model expects (None, 256, 256, 15)
    """

    def __init__(self, latent_size=64, invert_labels=False):
        """
        Args:
            latent_size (int): Size of the latent representation (model hyperparam; not an input size).
            invert_labels (bool): If model trained with fake=1, real=0, set True to invert semantics.
        """
        super().__init__(model_name=f"SRM-CNN-L{latent_size}")
        self.latent_size = latent_size
        self.image_size = 256
        self.model = None
        self.invert_labels = invert_labels

    # ─────────────────────────────────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────────────────────────────────
    def load(self, model_path):
        """Load a trained SRM‑CNN model."""
        logger.info(f"Loading {self.model_name} from {model_path}...")
        if not os.path.exists(model_path):
            logger.error(f"Error: Model file {model_path} does not exist")
            return False
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocess: EXACT training pipeline (TF only)
    # ─────────────────────────────────────────────────────────────────────────
    def _tf_load_resize_norm(self, image_path: str) -> tf.Tensor:
        """(256,256,3) float32 in [0,1], matches training (bilinear, no antialias)."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)  # match training decoder
        img = tf.image.resize(img, (self.image_size, self.image_size),
                              method=tf.image.ResizeMethod.BILINEAR, antialias=False)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def preprocess(self, image_path):
        """Return (1, 256, 256, 15) float32 numpy—exactly what the CNN saw at training."""
        # Load/resize/normalize in TF to avoid PIL/NumPy drift
        img = self._tf_load_resize_norm(image_path)   # (256,256,3)
        # Apply SRM (pure TF); output (1,256,256,15)
        srm = _srm_apply_rgb(img[None, ...])
        # Return numpy batch
        return srm.numpy().astype("float32")

    # ─────────────────────────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, processed_image):
        """Return P(real) in [0,1] from (1,256,256,15)."""
        if self.model is None:
            logger.error("Error: Model not loaded!")
            raise ValueError("Model not loaded")

        arr = processed_image
        if isinstance(arr, tf.Tensor):
            arr = arr.numpy()
        arr = np.asarray(arr)

        if arr.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B,H,W,C), got shape {arr.shape}")

        # Validate against model input shape (None, 256, 256, 15)
        expected = self.model.input_shape
        if expected and len(expected) == 4:
            _, eh, ew, ec = expected
            _, ah, aw, ac = arr.shape
            # Allow None for batch; check spatial and channel dims
            if (eh is not None and eh != ah) or (ew is not None and ew != aw) or (ec is not None and ec != ac):
                raise ValueError(f"Input shape mismatch: got {(ah, aw, ac)}, expected {(eh, ew, ec)}")

        try:
            pred = self.model(arr, training=False).numpy()
            if not isinstance(pred, np.ndarray) or pred.size == 0:
                raise ValueError(f"Unexpected prediction format: {pred}")
            p_real = float(pred.flatten()[0])
            # Clamp if needed
            if p_real < 0.0 or p_real > 1.0:
                logger.warning(f"Prediction out of [0,1]: {p_real}, clamping")
                p_real = max(0.0, min(1.0, p_real))
            return p_real
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Analyze
    # ─────────────────────────────────────────────────────────────────────────
    def analyze(self, image_path):
        """Returns {'probability': P(fake), 'prediction': 'real'|'fake', 'confidence':[0,1]}."""
        if self.model is None:
            logger.error(f"Model {self.model_name} is not loaded!")
            raise ValueError("Model not loaded")

        try:
            x = self.preprocess(image_path)          # (1,256,256,15)
            p_real = self.predict(x)                 # scalar in [0,1]

            if self.invert_labels:
                p_fake = p_real
                p_real = 1.0 - p_real
            else:
                p_fake = 1.0 - p_real

            pred = "real" if p_real > 0.5 else "fake"
            confidence = abs(p_real - 0.5) * 2.0

            return {
                "probability": float(p_fake),   # publish P(fake) for API consistency
                "prediction": pred,
                "confidence": float(confidence),
            }

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise ValueError(f"Analysis failed: {str(e)}")
