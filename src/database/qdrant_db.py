"""
Qdrant Vector Database Adapter for AIC Video Retrieval System.

Manages 3 collections:
  - "captions"  → BGE-M3 dense vectors of keyframe captions (EN+VI)
  - "ocr"       → BGE-M3 dense vectors of OCR text
  - "asr"       → BGE-M3 dense vectors of ASR transcripts

Each Qdrant point payload:
    {
      "keyframe_id": "L21_V001_n5",
      "video_id":    "L21_V001",
      "n":           5,
      "frame_idx":   450,
      "pts_time":    18.24,
      "text":        "the raw text that was embedded"
    }
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.common.constants import (
    QDRANT_COLLECTION_CAPTIONS,
    QDRANT_COLLECTION_OCR,
    QDRANT_COLLECTION_ASR,
    QDRANT_VECTOR_DIM_BGE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams,
        PointStruct, PayloadSchemaType,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


class QdrantDB:
    """
    Qdrant adapter for inserting and searching BGE-M3 text vectors.

    Usage:
        db = QdrantDB(url="http://localhost:6333")
        db.connect()
        db.create_collections()

        # Insert from extracted JSON files
        db.index_from_json(
            json_dir="datasets/captions",
            collection="captions",
            text_field="caption_en",   # field in each keyframe dict
            bge_encoder=encoder,
        )

        # Search
        hits = db.search("người dẫn mặc áo đỏ", collection="captions",
                         query_vec=bge_vec, top_k=100)
    """

    COLLECTIONS = {
        "captions": QDRANT_COLLECTION_CAPTIONS,
        "ocr":      QDRANT_COLLECTION_OCR,
        "asr":      QDRANT_COLLECTION_ASR,
    }

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.url     = url
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[QdrantClient] = None

    def connect(self) -> "QdrantDB":
        """Establish Qdrant connection."""
        if not _QDRANT_AVAILABLE:
            raise ImportError("qdrant_client not installed. Run: pip install qdrant-client")
        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        try:
            self._client.get_collections()
            logger.info(f"Qdrant connected: {self.url}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Qdrant at {self.url}: {e}")
        return self

    # ----------------------------------------------------------
    # Collection Management
    # ----------------------------------------------------------

    def create_collections(self, overwrite: bool = False) -> None:
        """Create all 3 AIC collections if they don't exist."""
        self._check_connected()
        existing = {c.name for c in self._client.get_collections().collections}

        for alias, col_name in self.COLLECTIONS.items():
            if col_name in existing:
                if overwrite:
                    self._client.delete_collection(col_name)
                    logger.info(f"Deleted existing collection: {col_name}")
                else:
                    logger.info(f"Collection already exists: {col_name}")
                    continue

            self._client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(
                    size=QDRANT_VECTOR_DIM_BGE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {col_name} (dim={QDRANT_VECTOR_DIM_BGE})")

    def collection_count(self, collection_name: str) -> int:
        """Return number of points in a collection."""
        self._check_connected()
        info = self._client.get_collection(collection_name)
        return info.points_count

    # ----------------------------------------------------------
    # Indexing from extracted JSON files
    # ----------------------------------------------------------

    def index_from_json(
        self,
        json_dir: str,
        collection: str,     # "captions" | "ocr" | "asr"
        text_field: str,     # field name inside each keyframe dict
        bge_encoder,         # BGEEncoder instance (must be loaded)
        batch_size: int = 256,
        skip_empty: bool = True,
    ) -> int:
        """
        Read all JSON files in json_dir, encode the text_field with BGE-M3,
        and upload vectors to the specified Qdrant collection.

        Args:
            json_dir:    Directory containing L{XX}_{V}.json files
            collection:  Target Qdrant collection alias
            text_field:  JSON key in each keyframe dict containing text
                         e.g. "caption_en", "texts" (OCR list), "asr_text"
            bge_encoder: Loaded BGEEncoder
            batch_size:  Points uploaded per Qdrant batch
            skip_empty:  Skip keyframes with no text

        Returns:
            Total number of vectors indexed
        """
        self._check_connected()
        col_name  = self.COLLECTIONS.get(collection, collection)
        json_files = sorted(Path(json_dir).glob("*.json"))
        logger.info(f"Indexing {len(json_files)} JSON files → Qdrant '{col_name}'")

        total_indexed = 0
        points_buffer: List[PointStruct] = []

        for json_path in json_files:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            video_id = data.get("video_id", json_path.stem)
            keyframes = data.get("keyframes", [])

            for kf in keyframes:
                raw_text = kf.get(text_field, "")
                # OCR "texts" field is a list — join into string
                if isinstance(raw_text, list):
                    raw_text = " ".join(raw_text)
                raw_text = str(raw_text).strip()

                if skip_empty and not raw_text:
                    continue

                n          = int(kf.get("n", 0))
                frame_idx  = int(kf.get("frame_idx", 0))
                pts_time   = float(kf.get("pts_time", 0.0))
                keyframe_id = f"{video_id}_n{n}"

                # Encode text
                vec = bge_encoder.encode(raw_text, normalize=True)

                # Build Qdrant point
                points_buffer.append(PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, keyframe_id + col_name)),
                    vector=vec.tolist(),
                    payload={
                        "keyframe_id": keyframe_id,
                        "video_id":    video_id,
                        "n":           n,
                        "frame_idx":   frame_idx,
                        "pts_time":    pts_time,
                        "text":        raw_text[:500],  # Truncate for payload size
                    },
                ))

                # Upload batch
                if len(points_buffer) >= batch_size:
                    self._client.upsert(
                        collection_name=col_name,
                        points=points_buffer,
                    )
                    total_indexed += len(points_buffer)
                    points_buffer = []

        # Upload remaining
        if points_buffer:
            self._client.upsert(
                collection_name=col_name,
                points=points_buffer,
            )
            total_indexed += len(points_buffer)

        logger.info(f"Indexed {total_indexed:,} vectors → {col_name}")
        return total_indexed

    # ----------------------------------------------------------
    # Search
    # ----------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        collection: str,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search a Qdrant collection with a BGE-M3 query vector.

        Returns:
            List of dicts with keys: keyframe_id, video_id, n, frame_idx, pts_time, score, text
        """
        self._check_connected()
        col_name = self.COLLECTIONS.get(collection, collection)

        hits = self._client.search(
            collection_name=col_name,
            query_vector=query_vec.tolist(),
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "keyframe_id": payload.get("keyframe_id", ""),
                "video_id":    payload.get("video_id", ""),
                "n":           int(payload.get("n", 0)),
                "frame_idx":   int(payload.get("frame_idx", 0)),
                "pts_time":    float(payload.get("pts_time", 0.0)),
                "score":       float(hit.score),
                "text":        payload.get("text", ""),
            })
        return results

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _check_connected(self) -> None:
        if self._client is None:
            raise RuntimeError("QdrantDB not connected. Call connect() first.")

    @property
    def is_connected(self) -> bool:
        return self._client is not None
