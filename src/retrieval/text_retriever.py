"""
Text Retriever — Qdrant-based search over captions, OCR, and ASR text.

This retriever is used when text features (captions, OCR, subtitles) have
been extracted and indexed into Qdrant. At Sprint 2, this is a functional
stub that gracefully returns an empty list if Qdrant is not available,
allowing the KIS pipeline to run on visual retrieval alone.

Sprint 3 will fully populate this retriever when OCR/Caption extraction
notebooks (kaggle_02 and kaggle_03) have been run.
"""

from __future__ import annotations

from typing import List, Optional

from src.retrieval.base import BaseRetriever
from src.common.types import SearchResult
from src.common.constants import (
    QDRANT_COLLECTION_CAPTIONS,
    QDRANT_COLLECTION_OCR,
    QDRANT_COLLECTION_ASR,
    QDRANT_VECTOR_DIM_BGE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import qdrant_client — optional at this sprint
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import ScoredPoint
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    logger.warning("qdrant_client not installed — TextRetriever will return empty results.")


class TextRetriever(BaseRetriever):
    """
    Retrieves keyframes via dense/sparse text search over Qdrant collections.

    Supports 3 text modalities:
      - "caption"  — auto-generated image captions (Qwen2.5-VL)
      - "ocr"      — extracted on-screen text (PaddleOCR)
      - "asr"      — speech transcripts (Whisper)

    Each modality maps to a separate Qdrant collection, queried via
    BGE-M3 dense vectors and/or BM25 sparse vectors.

    Args:
        qdrant_url:   Qdrant server URL (default: localhost:6333)
        modality:     Which collection to search: "caption" | "ocr" | "asr"
        bge_encoder:  Optional BGE-M3 encoder; if None, retriever is a stub
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        modality: str = "caption",
        bge_encoder=None,
    ):
        self.qdrant_url = qdrant_url
        self.modality = modality
        self._bge_encoder = bge_encoder
        self._client: Optional[QdrantClient] = None

        # Map modality → Qdrant collection name
        self._collection_map = {
            "caption": QDRANT_COLLECTION_CAPTIONS,
            "ocr":     QDRANT_COLLECTION_OCR,
            "asr":     QDRANT_COLLECTION_ASR,
        }

        if modality not in self._collection_map:
            raise ValueError(f"Unknown modality '{modality}'. "
                             f"Choose from: {list(self._collection_map)}")

    @property
    def name(self) -> str:
        return f"text_{self.modality}"

    @property
    def collection(self) -> str:
        return self._collection_map[self.modality]

    # ----------------------------------------------------------
    # Connect
    # ----------------------------------------------------------

    def connect(self) -> "TextRetriever":
        """Establish Qdrant connection. Call before retrieve()."""
        if not _QDRANT_AVAILABLE:
            logger.warning("Qdrant not available — TextRetriever disabled.")
            return self
        try:
            self._client = QdrantClient(url=self.qdrant_url, timeout=10)
            # Quick health check
            self._client.get_collections()
            logger.info(f"Qdrant connected: {self.qdrant_url} | collection={self.collection}")
        except Exception as e:
            logger.warning(f"Qdrant connection failed ({e}) — TextRetriever disabled.")
            self._client = None
        return self

    # ----------------------------------------------------------
    # Retrieve
    # ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
    ) -> List[SearchResult]:
        """
        Search Qdrant for keyframes whose text (caption/ocr/asr) matches query.

        Returns empty list if Qdrant is not connected (graceful fallback).
        """
        if self._client is None or self._bge_encoder is None:
            logger.debug(f"[{self.name}] Skipped (not configured)")
            return []

        # Check collection exists
        try:
            collections = [c.name for c in self._client.get_collections().collections]
            if self.collection not in collections:
                logger.debug(f"[{self.name}] Collection '{self.collection}' not found — skipped")
                return []
        except Exception as e:
            logger.warning(f"[{self.name}] Qdrant health check failed: {e}")
            return []

        # Encode query with BGE-M3
        query_vec = self._bge_encoder.encode(query, normalize=True)

        # Dense vector search
        try:
            hits = self._client.search(
                collection_name=self.collection,
                query_vector=query_vec.tolist(),
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Qdrant search failed: {e}")
            return []

        results: List[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(SearchResult(
                keyframe_id=payload.get("keyframe_id", ""),
                video_id=payload.get("video_id", ""),
                n=int(payload.get("n", 0)),
                frame_idx=int(payload.get("frame_idx", 0)),
                pts_time=float(payload.get("pts_time", 0.0)),
                score=float(hit.score),
                retriever_source=self.name,
                metadata={"text_snippet": payload.get("text", "")[:100]},
            ))

        logger.debug(f"[{self.name}] '{query[:50]}' → {len(results)} results")
        return results
