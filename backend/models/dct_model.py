import os
import numpy as np
import tensorflow as tf
from .base_model import BaseModel
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SRM: pure-TF kernel + apply, to mirror training numerics exactly
# ─────────────────────────────────────────────────────────────────────────────
def _build_srm_kernel():
    # 5 filters, each 5x5, same as your training kernels
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
    """Apply 5 SRM filters to each RGB channel (B,H,W,3) -> (B,H,W,15)"""
    feats = []
    for c in tf.split(img_bhwc, 3, axis=-1):
        feats.append(tf.nn.conv2d(c, _SRM_K, strides=1, padding="SAME"))
    return tf.concat(feats, axis=-1)


class DCTModel(BaseModel):
    """DCT-based model for deepfake detection with AE encoder branch."""

    def __init__(self, block_size=8, num_coefficients=64, use_encoder=False, latent_size=256, invert_labels=False):
        model_name = f"DCT-B{block_size}-C{num_coefficients}"
        if use_encoder:
            model_name = f"DCT-AE-L{latent_size}"
        super().__init__(model_name=model_name)

        self.block_size = block_size
        self.num_coefficients = num_coefficients
        self.image_size = 256
        self.use_encoder = use_encoder
        self.latent_size = latent_size
        self.model = None
        self.encoder = None
        self.invert_labels = invert_labels

    # ─────────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────────
    def load(self, model_path):
        """Load classifier and (optionally) its matching autoencoder encoder."""
        logger.info(f"Loading DCT model from {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"Error: Model file {model_path} does not exist")
            return False

        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Model loaded. Input shape: %s, Output shape: %s",
                        self.model.input_shape, self.model.output_shape)

            if self.use_encoder:
                expected_input_size = self.model.input_shape[1]
                # 8*8*128=8192, 8*8*256=16384
                if expected_input_size == 8192:
                    ae_fname = "autoencoder_L128"
                    self.latent_size = 128
                elif expected_input_size == 16384:
                    ae_fname = "autoencoder_L256"
                    self.latent_size = 256
                else:
                    logger.error(f"Unexpected input size {expected_input_size}, cannot select autoencoder")
                    return False

                base = os.path.join(os.path.dirname(model_path), "ae")
                ae_path = os.path.join(base, f"{ae_fname}.h5")
                if not os.path.exists(ae_path):
                    ae_path = os.path.join(base, f"{ae_fname}.keras")

                logger.info(f"Using autoencoder from {ae_path} for latent size {self.latent_size}")

                if not os.path.exists(ae_path):
                    logger.error(f"Autoencoder file not found: {ae_path}")
                    return False

                autoencoder = tf.keras.models.load_model(ae_path, compile=False)
                try:
                    self.encoder = tf.keras.models.Model(
                        inputs=autoencoder.input,
                        outputs=autoencoder.get_layer('encoder_output').output
                    )
                    self.encoder.trainable = False
                except Exception as e:
                    logger.error(f"Failed to extract encoder using 'encoder_output' layer: {e}")
                    return False

                logger.info("Encoder ready. In: %s Out: %s", self.encoder.input_shape, self.encoder.output_shape)

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocess: EXACT training pipeline (TF only)
    # ─────────────────────────────────────────────────────────────────────────
    def _tf_load_resize_norm(self, image_path: str) -> tf.Tensor:
        """(256,256,3) float32 in [0,1], matches training (bilinear, no antialias)."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (self.image_size, self.image_size),
                              method=tf.image.ResizeMethod.BILINEAR, antialias=False)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def preprocess(self, image_path):
        """
        Training-accurate pipeline:
          - read/decode/resize/normalize in TF
          - SRM → 15ch (TF conv2d)
          - encoder (frozen)
          - flatten
          - DCT(type=2, norm='ortho')
          - L2 normalize over features (axis=-1)
        Returns (1, FLAT_LEN) float32 numpy.
        """
        if self.use_encoder:
            if self.encoder is None:
                logger.error("Encoder model not loaded!")
                raise ValueError("Encoder model not loaded")

            img = self._tf_load_resize_norm(image_path)         # (256,256,3)
            srm = _srm_apply_rgb(img[None, ...])                # (1,256,256,15)
            latent = self.encoder(srm, training=False)          # (1,8,8,L)
            z = tf.reshape(latent, (1, -1))                     # (1, FLAT_LEN)
            dct = tf.signal.dct(z, type=2, norm='ortho')        # (1, FLAT_LEN)
            dct = tf.math.l2_normalize(dct, axis=-1)            # <<< correct axis
            return dct.numpy().astype("float32")

        # If you still need a non-encoder branch, you can implement it here.
        # Note: it will **not** match the AE-trained classifier pipeline by design.
        raise NotImplementedError("Non-encoder DCT path intentionally disabled to avoid mismatch with training.")

    # ─────────────────────────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, processed_features):
        """Run inference on preprocessed features."""
        if self.model is None:
            raise ValueError("Model not loaded")

        arr = processed_features
        if isinstance(arr, tf.Tensor):
            arr = arr.numpy()
        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr[None, :]

        expected = self.model.input_shape  # (None, FLAT_LEN)
        if expected is None or len(expected) < 2:
            raise ValueError(f"Unexpected model input shape: {expected}")

        exp_feat = expected[1]
        if arr.shape[1] != exp_feat:
            raise ValueError(f"Feature dim mismatch: got {arr.shape[1]}, expected {exp_feat}")

        return self.model.predict(arr, verbose=0)

    # ─────────────────────────────────────────────────────────────────────────
    # Analyze wrapper
    # ─────────────────────────────────────────────────────────────────────────
    def analyze(self, image_path):
        """Return {'probability': P(fake), 'prediction': 'real'|'fake', 'confidence': [0,1]}."""
        if self.model is None:
            raise ValueError("Model not loaded")
        if self.use_encoder and self.encoder is None:
            raise ValueError("Encoder not loaded")

        feats = self.preprocess(image_path)                 # (1, FLAT_LEN)
        raw = self.predict(feats)                           # (1, 1) sigmoid

        if not isinstance(raw, np.ndarray):
            raise ValueError(f"Unexpected prediction format: {type(raw)}")

        p_real = float(raw.flatten()[0])                    # trained with 0=fake, 1=real
        if self.invert_labels:
            p_fake = p_real
            p_real = 1.0 - p_real
        else:
            p_fake = 1.0 - p_real

        pred = "real" if p_real > 0.5 else "fake"
        confidence = abs(p_real - 0.5) * 2.0

        return {
            "probability": float(p_fake),  # publish P(fake) for your API
            "prediction": pred,
            "confidence": float(confidence),
        }
