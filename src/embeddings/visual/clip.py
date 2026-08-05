"""
CLIP Visual Encoder for AIC Video Retrieval System.

Wraps open_clip to encode:
  - Text queries → 512-dim L2-normalised vectors (for FAISS search)
  - Keyframe images → 512-dim L2-normalised vectors

Model used: ViT-B/32 (matching the pre-extracted clip-features-32 .npy files).
The model name "clip-features-32" in the dataset refers to CLIP ViT-B/32,
where 32 is the patch size.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip not installed. Run: pip install open-clip-torch")

from src.common.constants import CLIP32_FEATURE_DIM
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Model name mapping
_CLIP32_MODEL = "ViT-B-32"
_CLIP32_PRETRAINED = "openai"   # Use OpenAI's original weights for compatibility


class CLIPEncoder:
    """
    CLIP ViT-B/32 encoder — matches the pre-extracted .npy features.

    Usage:
        encoder = CLIPEncoder()
        encoder.load()

        # Encode a text query → FAISS-ready vector
        vec = encoder.encode_text("người dẫn mặc áo đỏ phát biểu ngoài trời")

        # Encode a keyframe image
        vec = encoder.encode_image("datasets/keyframes/L21/V001/5.jpg")
    """

    def __init__(
        self,
        model_name: str = _CLIP32_MODEL,
        pretrained: str = _CLIP32_PRETRAINED,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = CLIP32_FEATURE_DIM

        self._model = None
        self._preprocess = None
        self._tokenizer = None

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------

    def load(self) -> "CLIPEncoder":
        """Load model weights. Call once before encoding."""
        logger.info(f"Loading CLIP {self.model_name} ({self.pretrained}) on {self.device}")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        logger.info(f"CLIP loaded — output dim: {self.dim}")
        return self

    # ----------------------------------------------------------
    # Encode Text
    # ----------------------------------------------------------

    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode one or more text strings into CLIP vectors.

        Args:
            text:      Single string or list of strings
            normalize: L2-normalise output (required for cosine sim with FAISS)

        Returns:
            np.ndarray shape (dim,) for single string, (N, dim) for list
        """
        self._check_loaded()
        single = isinstance(text, str)
        texts = [text] if single else text

        tokens = self._tokenizer(texts).to(self.device)

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)

        result = features.cpu().numpy().astype(np.float32)
        return result[0] if single else result

    # ----------------------------------------------------------
    # Encode Image
    # ----------------------------------------------------------

    def encode_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a single keyframe image into a CLIP vector.

        Args:
            image:     File path (str/Path), PIL Image, or np.ndarray (H,W,C)
            normalize: L2-normalise output

        Returns:
            np.ndarray shape (dim,)
        """
        self._check_loaded()

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        tensor = self._preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(tensor)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()[0].astype(np.float32)

    def encode_images_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of images in batches.

        Returns:
            np.ndarray shape (N, dim)
        """
        self._check_loaded()
        all_vecs = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    pil = Image.open(img).convert("RGB")
                elif isinstance(img, np.ndarray):
                    pil = Image.fromarray(img)
                else:
                    pil = img
                tensors.append(self._preprocess(pil))

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self._model.encode_image(batch_tensor)
                if normalize:
                    features = features / features.norm(dim=-1, keepdim=True)
            all_vecs.append(features.cpu().numpy().astype(np.float32))

        return np.vstack(all_vecs)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _check_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("CLIPEncoder not loaded. Call encoder.load() first.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
