"""
Q&A Pipeline — End-to-end handler for Query Dạng 2 (Visual Question Answering).

Flow:
  1. Retrieve top-K candidate keyframes (same retrieval as KIS)
  2. For each candidate frame (in score order):
     a. Ask VLM: "Is this frame the right moment to answer the question?"
        → RelevanceScore (short, fast call)
     b. If relevant, run full answer extraction:
        → QAAnswer
     c. Stop when a high-confidence answer is found (score ≥ threshold)
        OR after processing max_frames candidates
  3. Return the best (frame, answer) pair

This "retrieve-then-verify" approach avoids running expensive VLM inference
on all 100 candidates — typically the correct frame appears in the top 5-10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.types import EvidenceResult, SearchResult, QAQuery, QASubmission
from src.reasoning.query_parser import QueryParser
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.text_retriever import TextRetriever
from src.fusion.reciprocal_rank import ReciprocalRankFusion
from src.evidence.frame_selector import FrameSelector
from src.llm.qwen_client import QwenVLClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default thresholds
_RELEVANCE_THRESHOLD   = 0.5   # RelevanceScore.confidence to proceed with full answer
_ANSWER_CONF_THRESHOLD = 0.6   # QAAnswer.confidence to stop early
_MAX_FRAMES_TO_CHECK   = 10    # Max candidates to run VLM on


class QAPipeline:
    """
    Visual Question Answering pipeline.

    Takes a QAQuery, retrieves candidate keyframes using CLIP + Qdrant,
    then runs Qwen2.5-VL to find the best frame and extract the answer.

    Args:
        visual_retriever: Loaded VisualRetriever (FAISS)
        text_retrievers:  List of TextRetriever (Qdrant caption/ocr/asr)
        rrf:              ReciprocalRankFusion instance
        vlm_client:       Loaded QwenVLClient
        keyframe_image_root: Root dir of keyframe images for VLM inference
        max_frames_to_check: Max candidates to pass through VLM
        relevance_threshold: Minimum VLM relevance confidence to proceed
        answer_conf_threshold: Early-stop if answer confidence >= this

    Usage:
        pipeline = QAPipeline(
            visual_retriever=vis_ret,
            vlm_client=qwen_client,
            keyframe_image_root="datasets/keyframes/keyframes",
        )
        result = pipeline.run(qa_query)
        # result.answer → "5"
        # result.evidence.frame_idx → 1500
    """

    def __init__(
        self,
        visual_retriever: VisualRetriever,
        vlm_client: QwenVLClient,
        keyframe_image_root: str,
        text_retrievers: Optional[List[TextRetriever]] = None,
        rrf: Optional[ReciprocalRankFusion] = None,
        max_frames_to_check: int = _MAX_FRAMES_TO_CHECK,
        relevance_threshold: float = _RELEVANCE_THRESHOLD,
        answer_conf_threshold: float = _ANSWER_CONF_THRESHOLD,
        top_k_retrieval: int = 100,
        top_k_fusion: int = 30,
    ):
        self._vis_ret     = visual_retriever
        self._vlm         = vlm_client
        self._kf_root     = Path(keyframe_image_root)
        self._text_rets   = text_retrievers or []
        self._rrf         = rrf or ReciprocalRankFusion(k=60)
        self._selector    = FrameSelector()
        self._parser      = QueryParser()

        self.max_frames        = max_frames_to_check
        self.rel_threshold     = relevance_threshold
        self.ans_conf_threshold = answer_conf_threshold
        self._top_k_ret        = top_k_retrieval
        self._top_k_fus        = top_k_fusion

    # ----------------------------------------------------------
    # Main Entry
    # ----------------------------------------------------------

    def run(self, qa_query: QAQuery, query_id: str = "") -> Optional[QASubmission]:
        """
        Execute the full Q&A pipeline.

        Args:
            qa_query:  Parsed QAQuery (from QueryParser.parse_qa)
            query_id:  ID string for logging

        Returns:
            QASubmission with video_id, frame_idx, and answer string
            OR None if no candidates found
        """
        logger.info(f"[QA] query_id='{query_id}' | Q: {qa_query.question[:80]}")

        # Step 1: Retrieve candidate keyframes (same as KIS)
        candidates = self._retrieve_candidates(qa_query)
        if not candidates:
            logger.warning(f"[QA] No candidates for query_id='{query_id}'")
            return None

        logger.debug(f"[QA] {len(candidates)} candidates after fusion")

        # Step 2: Iterate through top candidates, run VLM
        best_frame:  Optional[SearchResult] = None
        best_answer: str  = ""
        best_conf:   float = 0.0

        for rank, candidate in enumerate(candidates[:self.max_frames]):
            img_path = self._get_image_path(candidate)
            if not img_path or not img_path.exists():
                logger.debug(f"[QA] Image not found for {candidate.keyframe_id}, skipping")
                continue

            img_str = str(img_path)

            # 2a. Quick relevance check
            rel = self._vlm.score_relevance(
                img_str,
                qa_query.event_description,
                qa_query.question,
            )
            logger.debug(
                f"[QA] [{rank+1}/{self.max_frames}] {candidate.keyframe_id} "
                f"relevance={rel.confidence:.2f} (relevant={rel.relevant})"
            )

            if not rel.relevant or rel.confidence < self.rel_threshold:
                continue

            # 2b. Full answer extraction
            qa_ans = self._vlm.answer_question(
                img_str,
                qa_query.event_description,
                qa_query.question,
                answer_language=qa_query.answer_language,
            )
            logger.debug(
                f"[QA] [{rank+1}] answer='{qa_ans.answer[:60]}' conf={qa_ans.confidence:.2f}"
            )

            if qa_ans.confidence > best_conf:
                best_conf   = qa_ans.confidence
                best_answer = qa_ans.answer
                best_frame  = candidate

            # Early stop if confident enough
            if best_conf >= self.ans_conf_threshold:
                logger.debug(f"[QA] Early stop at rank {rank+1} (conf={best_conf:.2f})")
                break

        # Step 3: Fall back to top visual result if VLM found nothing
        if best_frame is None:
            logger.warning(f"[QA] VLM found no relevant frame — using top retrieval result")
            best_frame  = candidates[0]
            best_answer = "(VLM could not determine answer)"
            best_conf   = candidates[0].score

        # Step 4: Build evidence + submission
        evidence = self._selector.select_best(
            [best_frame],
            query_id=query_id,
            explanation=f"Q&A answer: '{best_answer}' (conf={best_conf:.2f})",
        )

        logger.info(
            f"[QA] Result: {best_frame.video_id} frame_idx={best_frame.frame_idx} "
            f"answer='{best_answer[:60]}' conf={best_conf:.2f}"
        )

        return QASubmission(
            query_id=query_id,
            video_id=best_frame.video_id,
            frame_idx=best_frame.frame_idx,
            answer=best_answer,
        )

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _retrieve_candidates(self, qa_query: QAQuery) -> List[SearchResult]:
        """Retrieve and fuse candidates using event_description as query text."""
        search_text = qa_query.event_description
        vis_results = self._vis_ret.retrieve(search_text, top_k=self._top_k_ret)

        all_lists   = [vis_results]
        all_weights = [1.0]
        for text_ret in self._text_rets:
            txt = text_ret.retrieve(search_text, top_k=self._top_k_ret)
            if txt:
                all_lists.append(txt)
                all_weights.append(0.8)

        fused = self._rrf.fuse(all_lists, all_weights, top_k=self._top_k_fus)
        return fused

    def _get_image_path(self, result: SearchResult) -> Optional[Path]:
        """
        Reconstruct keyframe image path from SearchResult metadata.

        Expected structure:
            {kf_root}/Keyframes_{batch_id}/keyframes/{video_id}/{n}.jpg
        """
        try:
            batch_id = result.video_id.split("_")[0]   # e.g. "L21"
            return (
                self._kf_root
                / f"Keyframes_{batch_id}"
                / "keyframes"
                / result.video_id
                / f"{result.n}.jpg"
            )
        except Exception:
            return None
