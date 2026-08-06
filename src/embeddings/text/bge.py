"""
BGE-M3 Text Encoder for AIC Video Retrieval System.

BGE-M3 (BAAI/bge-m3) is a multilingual, multi-granularity text embedding model
supporting 100+ languages including Vietnamese and English.

Used to encode:
- OCR text → dense vector for Qdrant search
- Captions (EN/VI) → dense vector for Qdrant search
- ASR transcripts → dense vector for Qdrant search
- Query text → dense vector for text-side retrieval

Output dimension: 1024 (dense embedding)
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from FlagEmbedding import BGEM3FlagModel
    _BGE_AVAILABLE = True
except ImportError:
    _BGE_AVAILABLE = False
    try:
        from sentence_transformers import SentenceTransformer
        _ST_AVAILABLE = True
    except ImportError:
        _ST_AVAILABLE = False


class BGEEncoder:
    """
    BGE-M3 text encoder — multilingual dense embeddings for Qdrant indexing.

    Supports two backends:
    1. FlagEmbedding (preferred) — full BGE-M3 with sparse + dense
    2. sentence-transformers (fallback) — dense only

    Usage:
        encoder = BGEEncoder()
        encoder.load()
        vec = encoder.encode("người dẫn mặc áo đỏ phát biểu")
        # vec.shape → (1024,)

        vecs = encoder.encode_batch(["text1", "text2", "text3"])
        # vecs.shape → (3, 1024)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize
        self._device    = device
        self._model     = None
        self._backend   = None  # "flag" or "sentence_transformers"

    def load(self) -> "BGEEncoder":
        """Load BGE-M3 model using the best available backend."""
        if _BGE_AVAILABLE:
            logger.info(f"Loading BGE-M3 via FlagEmbedding: {self.model_name}")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True,
                device=self._device,
            )
            self._backend = "flag"
        elif _ST_AVAILABLE:
            logger.warning("FlagEmbedding not available, using sentence-transformers fallback")
            self._model = SentenceTransformer(self.model_name, device=self._device)
            self._backend = "sentence_transformers"
        else:
            raise ImportError(
                "Neither FlagEmbedding nor sentence-transformers is installed. "
                "Run: pip install FlagEmbedding  or  pip install sentence-transformers"
            )
        logger.info(f"BGE-M3 loaded (backend={self._backend})")
        return self

    def encode(
        self,
        text: str,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Encode a single text string.

        Returns:
            np.ndarray shape (1024,)
        """
        result = self.encode_batch([text], normalize=normalize)
        return result[0]

    def encode_batch(
        self,
        texts: List[str],
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Encode a list of text strings in batches.

        Returns:
            np.ndarray shape (N, 1024)
        """
        self._check_loaded()
        use_norm = normalize if normalize is not None else self.normalize

        if self._backend == "flag":
            output = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = output["dense_vecs"].astype(np.float32)
        else:
            vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=use_norm,
                show_progress_bar=False,
            ).astype(np.float32)

        if use_norm and self._backend == "flag":
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-8)

        return vecs

    def _check_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("BGEEncoder not loaded. Call encoder.load() first.")

    @property
    def dim(self) -> int:
        return 1024

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
