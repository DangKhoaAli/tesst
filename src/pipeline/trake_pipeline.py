"""
TRAKE Pipeline — Temporal Retrieval & Alignment of Key Events (Query Dạng 3).

AIC competition requirement:
  Given an activity (e.g., "Nhảy cao") and a sequence of N event steps,
  find the video and submit ONE frame_idx per event step.

Two-Phase Strategy:
  ┌─────────────────────────────────────────────────────────┐
  │  Phase 1 — Video Retrieval (Which video?)               │
  │  • Combine all event descriptions as one query          │
  │  • FAISS visual + Qdrant text → RRF at video level      │
  │  • Return top-K candidate video_ids                     │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  Phase 2 — Event Alignment (Which frame per event?)     │
  │  For each candidate video:                              │
  │    For each event step:                                 │
  │      • Encode event description + hint → CLIP vector    │
  │      • retrieve_within_video() → ranked local frames    │
  │      • (Optional) VLM score_alignment() → verify top-N  │
  │      • Pick best frame with temporal ordering           │
  │  Score video by sum of per-event alignment confidences  │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  Phase 3 — Video Selection                             │
  │  • Pick the video with the highest total alignment score │
  │  • Return TRAKESubmission: {video_id, event → frame_idx} │
  └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.types import (
    TRAKEQuery, EventStep, TRAKESubmission, TRAKEEventResult,
    SearchResult, EvidenceResult,
)
from src.retrieval.visual_retriever import VisualRetriever
from src.retrieval.text_retriever import TextRetriever
from src.fusion.reciprocal_rank import ReciprocalRankFusion
from src.evidence.frame_selector import FrameSelector
from src.embeddings.visual.clip import CLIPEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Min VLM alignment confidence required to trust a frame
_VLM_CONF_THRESHOLD = 0.45
# Number of top frames per event to pass through VLM (max)
_VLM_MAX_FRAMES_PER_EVENT = 5


@dataclass
class _VideoAlignment:
    """Internal: alignment result for one candidate video."""
    video_id: str
    total_score: float = 0.0
    event_frames: Dict[int, SearchResult] = field(default_factory=dict)
    event_confidences: Dict[int, float] = field(default_factory=dict)


class TRAKEPipeline:
    """
    Temporal Retrieval & Alignment of Key Events pipeline.

    Args:
        visual_retriever:    Loaded VisualRetriever
        clip_encoder:        Loaded CLIPEncoder (for event-specific encoding)
        text_retrievers:     Optional list of Qdrant TextRetriever
        rrf:                 ReciprocalRankFusion instance
        vlm_client:          Optional QwenVLClient (for alignment verification)
        enable_vlm_verify:   Whether to run VLM on candidate frames
        top_k_videos:        Number of candidate videos from Phase 1

    Usage:
        pipeline = TRAKEPipeline(
            visual_retriever=vis_ret,
            clip_encoder=encoder,
        )
        submission = pipeline.run(trake_query, query_id="q001")
        # submission.video_id → "L21_V001"
        # submission.events[0].frame_idx → 900
    """

    def __init__(
        self,
        visual_retriever: VisualRetriever,
        clip_encoder: CLIPEncoder,
        text_retrievers: Optional[List[TextRetriever]] = None,
        rrf: Optional[ReciprocalRankFusion] = None,
        vlm_client=None,
        enable_vlm_verify: bool = True,
        top_k_videos: int = 5,
        top_k_frames_per_event: int = 20,
    ):
        self._vis_ret    = visual_retriever
        self._encoder    = clip_encoder
        self._text_rets  = text_retrievers or []
        self._rrf        = rrf or ReciprocalRankFusion(k=60)
        self._vlm        = vlm_client
        self._selector   = FrameSelector()

        self.enable_vlm_verify       = enable_vlm_verify and (vlm_client is not None)
        self.top_k_videos            = top_k_videos
        self.top_k_frames_per_event  = top_k_frames_per_event

    # ----------------------------------------------------------
    # Main Entry
    # ----------------------------------------------------------

    def run(
        self,
        trake_query: TRAKEQuery,
        query_id: str = "",
    ) -> Optional[TRAKESubmission]:
        """
        Execute the full TRAKE pipeline.

        Returns TRAKESubmission or None if no candidates found.
        """
        logger.info(
            f"[TRAKE] query_id='{query_id}' | activity='{trake_query.activity_name}' "
            f"| {len(trake_query.event_sequence)} events"
        )

        # --- Phase 1: Find candidate videos ---
        if trake_query.video_id:
            # Skip video retrieval if already specified
            candidate_video_ids = [trake_query.video_id]
            logger.info(f"[TRAKE] Skipping Phase 1 (video_id pre-specified: {trake_query.video_id})")
        else:
            candidate_video_ids = self._phase1_video_retrieval(trake_query)

        if not candidate_video_ids:
            logger.warning(f"[TRAKE] Phase 1 found no candidate videos")
            return None

        logger.info(f"[TRAKE] Phase 1 candidates: {candidate_video_ids}")

        # --- Phase 2: Align events in each candidate video ---
        video_alignments: List[_VideoAlignment] = []
        for video_id in candidate_video_ids:
            alignment = self._phase2_event_alignment(trake_query, video_id, query_id)
            video_alignments.append(alignment)

        if not video_alignments:
            logger.warning(f"[TRAKE] Phase 2 found no alignments")
            return None

        # --- Phase 3: Select best video ---
        best = max(video_alignments, key=lambda a: a.total_score)
        logger.info(
            f"[TRAKE] Best video: {best.video_id} "
            f"(score={best.total_score:.3f})"
        )

        # Build submission
        events = []
        for event in trake_query.event_sequence:
            ev_id = event.event_id
            frame = best.event_frames.get(ev_id)
            events.append(TRAKEEventResult(
                event_id=ev_id,
                frame_idx=frame.frame_idx if frame else 0,
                pts_time=frame.pts_time if frame else 0.0,
            ))

        return TRAKESubmission(
            query_id=query_id,
            video_id=best.video_id,
            events=events,
        )

    # ----------------------------------------------------------
    # Phase 1: Video-Level Retrieval
    # ----------------------------------------------------------

    def _phase1_video_retrieval(self, trake_query: TRAKEQuery) -> List[str]:
        """
        Build a composite query from activity + all event descriptions,
        retrieve at video level, return top-K video_ids.
        """
        # Composite query: activity + all event descriptions
        parts = [trake_query.activity_name]
        for ev in trake_query.event_sequence:
            parts.append(ev.description)
            if ev.semantic_keyframe_hint:
                parts.append(ev.semantic_keyframe_hint)
        composite_query = ". ".join(parts)

        # Visual retrieval (global, large top_k to cover all keyframes)
        global_top_k = min(self.top_k_videos * 100, 500)
        vis_results = self._vis_ret.retrieve(composite_query, top_k=global_top_k)

        # Text retrieval
        all_lists   = [vis_results]
        all_weights = [1.0]
        for text_ret in self._text_rets:
            txt = text_ret.retrieve(trake_query.activity_name, top_k=global_top_k)
            if txt:
                all_lists.append(txt)
                all_weights.append(0.7)

        # Video-level RRF (aggregate scores per video)
        video_ranked = self._rrf.fuse_video_level(
            result_lists=all_lists,
            weights=all_weights,
            top_k=self.top_k_videos,
        )
        return [r.video_id for r in video_ranked]

    # ----------------------------------------------------------
    # Phase 2: Per-Event Alignment Within a Video
    # ----------------------------------------------------------

    def _phase2_event_alignment(
        self,
        trake_query: TRAKEQuery,
        video_id: str,
        query_id: str,
    ) -> _VideoAlignment:
        """
        For each event step, find the best matching keyframe within video_id.
        Enforces temporal ordering: frame for event N+1 must come after event N.
        """
        alignment = _VideoAlignment(video_id=video_id)
        event_results: Dict[int, List[SearchResult]] = {}

        for ev in trake_query.event_sequence:
            # Build event-specific CLIP query
            event_query = f"{ev.description}. {ev.semantic_keyframe_hint}"
            query_vec = self._encoder.encode_text(event_query, normalize=True)

            # Search within this video only
            candidates = self._vis_ret.retrieve_within_video(
                query_vec=query_vec,
                video_id=video_id,
                top_k=self.top_k_frames_per_event,
            )
            event_results[ev.event_id] = candidates
            logger.debug(
                f"[TRAKE] {video_id} event {ev.event_id} '{ev.event_name}': "
                f"{len(candidates)} candidates"
            )

        # Select frames with temporal ordering enforced
        selections = self._selector.select_per_event(
            event_results=event_results,
            enforce_temporal_order=True,
        )

        # Optional VLM verification
        total_score = 0.0
        for ev in trake_query.event_sequence:
            ev_id      = ev.event_id
            best_frame = selections.get(ev_id)
            if best_frame is None:
                continue

            conf = best_frame.score  # Default: CLIP cosine score

            if self.enable_vlm_verify:
                conf = self._vlm_verify_event(ev, best_frame, conf)

            alignment.event_frames[ev_id]       = best_frame
            alignment.event_confidences[ev_id]  = conf
            total_score += conf

        alignment.total_score = total_score / max(len(trake_query.event_sequence), 1)
        logger.debug(
            f"[TRAKE] {video_id}: avg_score={alignment.total_score:.3f} "
            f"({len(alignment.event_frames)}/{len(trake_query.event_sequence)} events aligned)"
        )
        return alignment

    def _vlm_verify_event(
        self,
        event: EventStep,
        candidate: SearchResult,
        clip_score: float,
    ) -> float:
        """
        Run VLM alignment verification on the top candidate frame.
        Returns adjusted confidence (blend of CLIP score + VLM score).
        """
        if self._vlm is None:
            return clip_score

        # Reconstruct image path from visual retriever meta store
        meta = self._vis_ret._meta_store.get_by_faiss_id(
            list(self._vis_ret._meta_store._faiss_id_to_meta.keys())[0]  # fallback
        )
        kf_meta = self._vis_ret._meta_store.get_by_keyframe_id(candidate.keyframe_id)
        if kf_meta is None or not kf_meta.image_path:
            return clip_score

        try:
            alignment = self._vlm.score_alignment(
                image_path=kf_meta.image_path,
                event_name=event.event_name,
                semantic_keyframe_hint=event.semantic_keyframe_hint or event.description,
            )
            # Blend: 60% VLM + 40% CLIP
            blended = 0.6 * alignment.confidence + 0.4 * clip_score
            logger.debug(
                f"[TRAKE VLM] event={event.event_name} "
                f"match={alignment.match} vlm_conf={alignment.confidence:.2f} "
                f"→ blended={blended:.2f}"
            )
            return blended
        except Exception as e:
            logger.warning(f"[TRAKE VLM] Verification failed: {e}")
            return clip_score
