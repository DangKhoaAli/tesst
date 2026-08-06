"""
Structured Output Parser for VLM responses (v2).

Changes from v1:
- parse_qa_answer: Now handles the new combined JSON format
  {"found": bool, "answer": str, "confidence": float, "observation": str}
  Falls back gracefully to v1 heuristic if JSON not found.
- parse_alignment: Handles new chain-of-thought format with "observation" field.
- All parsers: robust extraction from markdown fences, raw JSON, and plain text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Marker returned by VLM when it cannot answer from the frame
_NOT_FOUND_MARKERS = [
    "không có thông tin",
    "không thể xác định",
    "không nhìn thấy",
    "không rõ",
    "not found",
    "cannot determine",
    "n/a",
    "không có",
]


@dataclass
class QAAnswer:
    """Parsed result from a Q&A VLM call."""
    answer: str           # The actual answer text
    confidence: float     # 0.0 – 1.0
    found: bool           # True if frame contains the answer
    observation: str      # What VLM observed in the frame
    raw_output: str       # Original VLM text (for debugging)


@dataclass
class RelevanceScore:
    """Parsed result from a frame-relevance VLM call."""
    relevant: bool
    confidence: float     # 0.0 – 1.0
    reason: str
    raw_output: str


@dataclass
class AlignmentScore:
    """Parsed result from a TRAKE alignment VLM call."""
    match: bool
    confidence: float
    reason: str
    observation: str      # What VLM observed (new in v2)
    raw_output: str


class ResponseParser:
    """
    Converts raw Qwen2.5-VL text output into typed Python objects.

    Priority: JSON parse → regex extraction → heuristic fallback
    """

    # ----------------------------------------------------------
    # Q&A Answer Parsing (v2: handles combined JSON format)
    # ----------------------------------------------------------

    def parse_qa_answer(self, raw_output: str) -> QAAnswer:
        """
        Parse raw VLM output as a Q&A answer.

        Handles two formats:
        1. (New) JSON: {"found": bool, "answer": str, "confidence": float, "observation": str}
        2. (Legacy) Plain text answer → estimate confidence heuristically
        """
        text = raw_output.strip()

        # --- Try JSON parse first (new format) ---
        parsed = self._try_parse_json(text)
        if parsed and "answer" in parsed:
            answer = str(parsed.get("answer", "")).strip()
            found = bool(parsed.get("found", True))
            confidence = float(parsed.get("confidence", 0.5))
            observation = str(parsed.get("observation", ""))

            # If VLM says not found or answer is a not-found marker
            if not found or self._is_not_found(answer):
                return QAAnswer(
                    answer="",
                    confidence=0.0,
                    found=False,
                    observation=observation,
                    raw_output=raw_output,
                )

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))
            return QAAnswer(
                answer=answer,
                confidence=confidence,
                found=True,
                observation=observation,
                raw_output=raw_output,
            )

        # --- Fallback: plain text (legacy format) ---
        if self._is_not_found(text):
            return QAAnswer(
                answer="",
                confidence=0.0,
                found=False,
                observation="",
                raw_output=raw_output,
            )

        # Estimate confidence from length and hedging
        hedge_patterns = [
            r"\bcó thể\b", r"\bcó lẽ\b", r"\bkhông chắc\b",
            r"\bI think\b", r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b",
            r"\bkhoảng\b", r"\btầm\b",
        ]
        hedging = any(re.search(p, text, re.IGNORECASE) for p in hedge_patterns)

        if len(text) < 30 and not hedging:
            confidence = 0.80
        elif hedging:
            confidence = 0.40
        elif len(text) < 80:
            confidence = 0.65
        else:
            confidence = 0.55  # Very long answer → likely rambling

        return QAAnswer(
            answer=text,
            confidence=confidence,
            found=True,
            observation="",
            raw_output=raw_output,
        )

    # ----------------------------------------------------------
    # Relevance Score Parsing
    # ----------------------------------------------------------

    def parse_relevance(self, raw_output: str) -> RelevanceScore:
        """
        Parse VLM response as a relevance JSON.
        Expected: {"relevant": true, "confidence": 0.9, "reason": "..."}
        """
        parsed = self._try_parse_json(raw_output)
        if parsed:
            return RelevanceScore(
                relevant=bool(parsed.get("relevant", False)),
                confidence=float(parsed.get("confidence", 0.5)),
                reason=str(parsed.get("reason", "")),
                raw_output=raw_output,
            )

        # Fallback: keyword search
        text_lower = raw_output.lower()
        is_relevant = any(kw in text_lower for kw in ["true", "có", "yes", "phù hợp", "đúng", "relevant"])
        is_not = any(kw in text_lower for kw in ["false", "không", "no", "không phù hợp"])

        if is_not:
            is_relevant = False
        return RelevanceScore(
            relevant=is_relevant,
            confidence=0.45,
            reason="(parsed via keyword fallback)",
            raw_output=raw_output,
        )

    # ----------------------------------------------------------
    # TRAKE Alignment Score Parsing (v2: includes observation)
    # ----------------------------------------------------------

    def parse_alignment(self, raw_output: str) -> AlignmentScore:
        """
        Parse VLM response as a TRAKE alignment JSON.
        New format: {"observation": str, "match": bool, "confidence": float, "reason": str}
        Legacy format: {"match": bool, "confidence": float, "reason": str}
        """
        parsed = self._try_parse_json(raw_output)
        if parsed:
            return AlignmentScore(
                match=bool(parsed.get("match", False)),
                confidence=float(parsed.get("confidence", 0.5)),
                reason=str(parsed.get("reason", "")),
                observation=str(parsed.get("observation", "")),
                raw_output=raw_output,
            )

        # Fallback: keyword search
        text_lower = raw_output.lower()
        # Check negative first (stronger signal)
        is_no = any(kw in text_lower for kw in ["false", "không khớp", "không đúng", "no match"])
        is_yes = any(kw in text_lower for kw in ["true", "yes", "khớp", "đúng", "match"])
        is_match = is_yes and not is_no

        return AlignmentScore(
            match=is_match,
            confidence=0.40,
            reason="(parsed via keyword fallback)",
            observation="",
            raw_output=raw_output,
        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _is_not_found(self, text: str) -> bool:
        """Check if a text string indicates the VLM couldn't find an answer."""
        t = text.lower().strip()
        return any(marker in t for marker in _NOT_FOUND_MARKERS)

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract and parse a JSON object from raw text."""
        # Direct parse
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        # Extract JSON block from markdown code fence
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            try:
                obj = json.loads(code_block.group(1))
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

        # Extract first complete {...} in the text (handles trailing text)
        # Use a more permissive regex that allows nested objects
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to find JSON across multiple lines
        multiline_match = re.search(r"\{[\s\S]*?\}", text)
        if multiline_match:
            try:
                obj = json.loads(multiline_match.group(0))
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

        logger.debug(f"JSON parse failed for: {text[:120]!r}")
        return None
