"""
Retrieval Pipeline — end-to-end query processing for all 3 AIC task types.

Orchestrates the full retrieval flow for Sprint 2 (KIS focus):
  Query text
    → QueryClassifier  (which type?)
    → QueryParser      (structured fields)
    → VisualRetriever  (FAISS CLIP-32 search)
    → TextRetriever    (Qdrant caption/ocr — graceful stub)
    → RRF Fusion
    → FrameSelector    (pick best frame)
    → EvidenceResult   (video_id, frame_idx, pts_time)

Sprint 3/4 will extend this with:
  - CLIPReranker (cross-modal reranking)
  - QA pipeline (VLM answer extraction)
  - TRAKE pipeline (2-phase temporal alignment)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.enums import QueryType
from src.common.types import (
    EvidenceResult, SearchResult, TextualKISQuery,
)
from src.reasoning.query_classifier import QueryClassifier
from src.reasoning.query_parser import QueryParser
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.text_retriever import TextRetriever
from src.fusion.reciprocal_rank import ReciprocalRankFusion
from src.evidence.frame_selector import FrameSelector
from src.database.faiss_db import FaissDB
from src.storage.metadata_store import MetadataStore
from src.embeddings.visual.clip import CLIPEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline routing queries to the appropriate sub-pipeline.

    Usage:
        pipeline = RetrievalPipeline.from_index_dir(
            index_dir="indexes",
            clip_model="ViT-B-32",
        )
        result = pipeline.run({"text": "người dẫn mặc áo đỏ..."})
        # → EvidenceResult(video_id="L21_V001", frame_idx=1500, ...)
    """

    def __init__(
        self,
        faiss_db: FaissDB,
        meta_store: MetadataStore,
        encoder: CLIPEncoder,
        text_retrievers: Optional[List[TextRetriever]] = None,
        rrf_k: int = 60,
        visual_weight: float = 1.0,
        text_weight: float = 0.8,
        top_k_retrieval: int = 100,
        top_k_fusion: int = 50,
    ):
        self._faiss_db   = faiss_db
        self._meta_store = meta_store
        self._encoder    = encoder

        # Core modules
        self._classifier  = QueryClassifier()
        self._parser      = QueryParser()
        self._vis_ret     = VisualRetriever(faiss_db, meta_store, encoder)
        self._text_rets   = text_retrievers or []
        self._rrf         = ReciprocalRankFusion(k=rrf_k)
        self._selector    = FrameSelector()

        # Weights: visual vs each text retriever
        self._visual_weight = visual_weight
        self._text_weight   = text_weight
        self._top_k_ret     = top_k_retrieval
        self._top_k_fus     = top_k_fusion

    # ----------------------------------------------------------
    # Factory
    # ----------------------------------------------------------

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str = "indexes",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        device: Optional[str] = None,
        **kwargs,
    ) -> "RetrievalPipeline":
        """
        Load all components from a pre-built index directory.

        Expected layout:
            index_dir/
            ├── faiss_visual.index
            ├── keyframe_master.parquet
            └── faiss_ids_map.json   (optional, for validation)
        """
        idx = Path(index_dir)

        # Load FAISS index
        faiss_db = FaissDB()
        faiss_db.load(str(idx / "faiss_visual.index"))

        # Load metadata store
        meta_store = MetadataStore(
            map_keyframes_root="",   # Not needed when loading parquet
            keyframes_image_root="",
        ).load(str(idx / "keyframe_master.parquet"))

        # Load CLIP encoder
        encoder = CLIPEncoder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=device,
        ).load()

        logger.info(
            f"RetrievalPipeline ready — "
            f"index: {faiss_db.total_vectors:,} vectors, "
            f"metadata: {meta_store.total_keyframes:,} keyframes"
        )
        return cls(faiss_db, meta_store, encoder, **kwargs)

    # ----------------------------------------------------------
    # Main Entry Point
    # ----------------------------------------------------------

    def run(
        self,
        query_dict: Dict[str, Any],
        query_id: str = "",
    ) -> Optional[EvidenceResult]:
        """
        Process one query dict and return the best evidence result.

        Args:
            query_dict: Parsed from JSON input file. Examples:
                KIS:   {"type": "textual_kis", "text": "..."}
                Q&A:   {"description": "...", "question": "..."}
                TRAKE: {"activity": "...", "events": [...]}
            query_id:   Optional ID string for logging

        Returns:
            EvidenceResult or None if no results found
        """
        qtype = self._classifier.classify(query_dict)
        logger.info(f"[Pipeline] query_id='{query_id}' type={qtype.value}")

        if qtype == QueryType.TEXTUAL_KIS:
            return self._run_kis(query_dict, query_id)
        elif qtype == QueryType.QA:
            return self._run_kis(query_dict, query_id)  # Phase 1 same as KIS; QA answer in Sprint 4
        elif qtype == QueryType.TRAKE:
            logger.warning("TRAKE pipeline not yet implemented — running as KIS.")
            return self._run_kis(query_dict, query_id)
        return None

    def run_batch(
        self,
        query_dicts: List[Dict[str, Any]],
    ) -> List[Optional[EvidenceResult]]:
        """Process a list of queries and return results in order."""
        results = []
        for i, qdict in enumerate(query_dicts):
            qid = qdict.get("query_id", str(i))
            result = self.run(qdict, query_id=qid)
            results.append(result)
        return results

    # ----------------------------------------------------------
    # KIS Sub-Pipeline
    # ----------------------------------------------------------

    def _run_kis(
        self,
        query_dict: Dict[str, Any],
        query_id: str,
    ) -> Optional[EvidenceResult]:
        """
        Full KIS retrieval flow:
        Text → CLIP encode → FAISS search → (+ text search) → RRF → best frame
        """
        # 1. Parse query
        raw_text = query_dict.get("text") or query_dict.get("description", "")
        kis_query = self._parser.parse_kis(raw_text, top_k=self._top_k_ret)
        retrieval_text = self._parser.build_retrieval_text(kis_query)

        # 2. Visual retrieval (FAISS)
        vis_results = self._vis_ret.retrieve(retrieval_text, top_k=self._top_k_ret)

        # 3. Text retrieval (Qdrant — may return empty if not configured)
        all_result_lists = [vis_results]
        all_weights      = [self._visual_weight]

        for text_ret in self._text_rets:
            txt_results = text_ret.retrieve(raw_text, top_k=self._top_k_ret)
            if txt_results:
                all_result_lists.append(txt_results)
                all_weights.append(self._text_weight)

        # 4. RRF fusion
        fused = self._rrf.fuse(
            result_lists=all_result_lists,
            weights=all_weights,
            top_k=self._top_k_fus,
        )

        if not fused:
            logger.warning(f"[Pipeline] No results for query_id='{query_id}'")
            return None

        # 5. Select best frame
        evidence = self._selector.select_best(fused, query_id=query_id)
        return evidence
