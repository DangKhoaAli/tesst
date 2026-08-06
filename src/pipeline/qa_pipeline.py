"""
Q&A Pipeline — End-to-end handler for Query Dạng 2 (Visual Question Answering) v2.

Changes from v1:
- Single VLM call per frame (combined relevance + answer), was 2 calls.
- max_frames increased 10 → 20 for better coverage.
- Multi-frame answer voting: if ≥ 2 frames return the same answer → boost confidence.
- Retrieval uses build_qa_retrieval_text() which combines event_description + question
  keywords → better candidate recall.
- answer_type is passed to VLM for type-specific answer formatting.
- Early-stop only on high-confidence + found frames.
- QAAnswer.found=False frames are skipped instead of used as fallback.

Flow:
  1. Build rich retrieval text (event_description + question keywords)
  2. Retrieve top-K candidate keyframes (CLIP + optional Qdrant)
  3. For each candidate (up to max_frames):
     a. Run combined VLM call → QAAnswer (found, answer, confidence, observation)
     b. If found=True and confidence > threshold → add to answer pool
     c. Early-stop if best confidence >= high_conf_threshold
  4. Multi-frame voting: find the most common answer among found frames
  5. Return best (frame, answer) pair; fallback to top retrieval if nothing found
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.types import EvidenceResult, SearchResult, QAQuery, QASubmission
from src.reasoning.query_parser import QueryParser
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.text_retriever import TextRetriever
from src.fusion.reciprocal_rank import ReciprocalRankFusion
from src.evidence.frame_selector import FrameSelector
from src.llm.qwen_client import QwenVLClient
from src.llm.response_parser import QAAnswer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds
_MIN_CONFIDENCE      = 0.30   # Minimum to consider a found answer
_HIGH_CONF_THRESHOLD = 0.80   # Early-stop if this confident
_MAX_FRAMES          = 20     # Max candidates to run VLM on
_VOTE_THRESHOLD      = 2      # Min frames agreeing on answer → boost confidence


class QAPipeline:
    """
    Visual Question Answering pipeline (v2).

    Args:
        visual_retriever:     Loaded VisualRetriever (FAISS)
        text_retrievers:      List of TextRetriever (Qdrant caption/ocr/asr)
        rrf:                  ReciprocalRankFusion instance
        vlm_client:           Loaded QwenVLClient
        keyframe_image_root:  Root dir of keyframe images for VLM inference
        max_frames:           Max candidates to pass through VLM (default: 20)
        min_confidence:       Minimum confidence to count an answer (default: 0.30)
        high_conf_threshold:  Early-stop if answer confidence >= this (default: 0.80)
        top_k_retrieval:      Candidates from FAISS/Qdrant
        top_k_fusion:         Candidates kept after RRF fusion

    Usage:
        pipeline = QAPipeline(
            visual_retriever=vis_ret,
            vlm_client=qwen_client,
            keyframe_image_root="datasets/keyframes/keyframes",
        )
        result = pipeline.run(qa_query)
        # result.answer → "5"
        # result.frame_idx → 1500
    """

    def __init__(
        self,
        visual_retriever: VisualRetriever,
        vlm_client: QwenVLClient,
        keyframe_image_root: str,
        text_retrievers: Optional[List[TextRetriever]] = None,
        rrf: Optional[ReciprocalRankFusion] = None,
        max_frames: int = _MAX_FRAMES,
        min_confidence: float = _MIN_CONFIDENCE,
        high_conf_threshold: float = _HIGH_CONF_THRESHOLD,
        top_k_retrieval: int = 100,
        top_k_fusion: int = 30,
    ):
        self._vis_ret    = visual_retriever
        self._vlm        = vlm_client
        self._kf_root    = Path(keyframe_image_root)
        self._text_rets  = text_retrievers or []
        self._rrf        = rrf or ReciprocalRankFusion(k=60)
        self._selector   = FrameSelector()
        self._parser     = QueryParser()

        self.max_frames         = max_frames
        self.min_confidence     = min_confidence
        self.high_conf_threshold = high_conf_threshold
        self._top_k_ret         = top_k_retrieval
        self._top_k_fus         = top_k_fusion

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
        logger.info(
            f"[QA] query_id='{query_id}' | type={qa_query.answer_type} "
            f"| Q: {qa_query.question[:80]}"
        )

        # Step 1: Retrieve candidate keyframes using enriched QA text
        candidates = self._retrieve_candidates(qa_query)
        if not candidates:
            logger.warning(f"[QA] No candidates for query_id='{query_id}'")
            return None

        logger.info(f"[QA] {len(candidates)} candidates after fusion")

        # Step 2: Run VLM on top candidates (single combined call per frame)
        found_answers: List[Tuple[SearchResult, QAAnswer]] = []
        best_conf = 0.0
        best_frame: Optional[SearchResult] = None
        best_qa: Optional[QAAnswer] = None

        for rank, candidate in enumerate(candidates[:self.max_frames]):
            img_path = self._get_image_path(candidate)
            if not img_path or not img_path.exists():
                logger.debug(f"[QA] Image not found for {candidate.keyframe_id}, skipping")
                continue

            # Combined 1-call: relevance + answer together
            qa_ans = self._vlm.answer_question(
                image_path=str(img_path),
                event_description=qa_query.event_description,
                question=qa_query.question,
                answer_language=qa_query.answer_language if qa_query.answer_language != "auto" else "vi",
                answer_type=qa_query.answer_type,
            )

            logger.debug(
                f"[QA] [{rank+1}/{self.max_frames}] {candidate.keyframe_id} "
                f"found={qa_ans.found} conf={qa_ans.confidence:.2f} "
                f"answer='{qa_ans.answer[:50]}'"
            )

            if not qa_ans.found or qa_ans.confidence < self.min_confidence:
                continue

            found_answers.append((candidate, qa_ans))

            if qa_ans.confidence > best_conf:
                best_conf  = qa_ans.confidence
                best_frame = candidate
                best_qa    = qa_ans

            # Early stop if very confident
            if best_conf >= self.high_conf_threshold:
                logger.info(f"[QA] Early stop at rank {rank+1} (conf={best_conf:.2f})")
                break

        # Step 3: Multi-frame answer voting — boost confidence if multiple frames agree
        if len(found_answers) >= _VOTE_THRESHOLD:
            best_frame, best_qa = self._vote_best_answer(found_answers)
            logger.info(
                f"[QA] Vote result: answer='{best_qa.answer}' "
                f"conf={best_qa.confidence:.2f} from {len(found_answers)} frames"
            )

        # Step 4: Fallback to top retrieval result if VLM found nothing
        if best_frame is None or best_qa is None:
            logger.warning(f"[QA] VLM found no answer — using top retrieval result with no answer")
            best_frame  = candidates[0]
            answer_text = ""
        else:
            answer_text = best_qa.answer

        logger.info(
            f"[QA] Result: {best_frame.video_id} frame_idx={best_frame.frame_idx} "
            f"answer='{answer_text[:60]}' conf={best_conf:.2f}"
        )

        return QASubmission(
            query_id=query_id,
            video_id=best_frame.video_id,
            frame_idx=best_frame.frame_idx,
            answer=answer_text,
        )

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _retrieve_candidates(self, qa_query: QAQuery) -> List[SearchResult]:
        """
        Retrieve and fuse candidates using enriched QA retrieval text
        (event_description + question visual keywords).
        """
        # Use enriched QA retrieval text instead of just event_description
        search_text = self._parser.build_qa_retrieval_text(qa_query)
        logger.debug(f"[QA] Retrieval text: {search_text[:120]}")

        vis_results = self._vis_ret.retrieve(search_text, top_k=self._top_k_ret)

        all_lists   = [vis_results]
        all_weights = [1.0]
        for text_ret in self._text_rets:
            # Also search with just event_description for text retrievers
            txt = text_ret.retrieve(qa_query.event_description, top_k=self._top_k_ret)
            if txt:
                all_lists.append(txt)
                all_weights.append(0.8)

        fused = self._rrf.fuse(all_lists, all_weights, top_k=self._top_k_fus)
        return fused

    def _get_image_path(self, result: SearchResult) -> Optional[Path]:
        """
        Reconstruct keyframe image path from SearchResult.

        Expected structure:
            {kf_root}/Keyframes_{batch_id}/keyframes/{video_id}/{n}.jpg

        Also tries flat structure:
            {kf_root}/{video_id}/{n}.jpg
        """
        try:
            batch_id = result.video_id.split("_")[0]   # e.g. "L21"

            # Primary path (standard AIC dataset structure)
            primary = (
                self._kf_root
                / f"Keyframes_{batch_id}"
                / "keyframes"
                / result.video_id
                / f"{result.n}.jpg"
            )
            if primary.exists():
                return primary

            # Fallback: flat structure
            flat = self._kf_root / result.video_id / f"{result.n}.jpg"
            if flat.exists():
                return flat

            # Fallback: directly under kf_root
            direct = self._kf_root / f"{result.keyframe_id}.jpg"
            if direct.exists():
                return direct

            return primary  # Return primary anyway so caller sees the missing path
        except Exception as e:
            logger.debug(f"[QA] _get_image_path error for {result.keyframe_id}: {e}")
            return None

    def _vote_best_answer(
        self,
        found_answers: List[Tuple[SearchResult, QAAnswer]],
    ) -> Tuple[SearchResult, QAAnswer]:
        """
        Multi-frame voting: find the most common answer and boost confidence.

        Algorithm:
          - Normalize answers (lowercase, strip)
          - Count occurrences of each unique answer
          - Pick the answer with the highest (count × avg_confidence) score
          - Use the frame with the highest individual confidence for that answer
        """
        # Normalize answers for comparison
        def normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        # Group by normalized answer
        answer_groups: Dict[str, List[Tuple[SearchResult, QAAnswer]]] = {}
        for frame, qa in found_answers:
            key = normalize(qa.answer)
            if key not in answer_groups:
                answer_groups[key] = []
            answer_groups[key].append((frame, qa))

        # Score each group: count × mean_confidence
        best_key = ""
        best_score = -1.0
        for norm_answer, group in answer_groups.items():
            count = len(group)
            avg_conf = sum(qa.confidence for _, qa in group) / count
            group_score = count * avg_conf
            if group_score > best_score:
                best_score = group_score
                best_key = norm_answer

        best_group = answer_groups[best_key]
        # Pick the frame with highest individual confidence
        best_frame, best_qa = max(best_group, key=lambda x: x[1].confidence)

        # Boost confidence if multiple frames agree
        vote_count = len(best_group)
        if vote_count >= 3:
            boosted_conf = min(1.0, best_qa.confidence + 0.10)
        elif vote_count == 2:
            boosted_conf = min(1.0, best_qa.confidence + 0.05)
        else:
            boosted_conf = best_qa.confidence

        # Create boosted QAAnswer
        boosted_qa = QAAnswer(
            answer=best_qa.answer,
            confidence=boosted_conf,
            found=True,
            observation=best_qa.observation,
            raw_output=best_qa.raw_output,
        )
        return best_frame, boosted_qa


