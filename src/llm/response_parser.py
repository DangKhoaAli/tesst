"""
Structured Output Parser for VLM responses.

Parses raw text output from Qwen2.5-VL into structured Python objects,
handling malformed JSON gracefully with regex fallbacks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QAAnswer:
    """Parsed result from a Q&A VLM call."""
    answer: str          # The actual answer text
    confidence: float    # 0.0 – 1.0 (estimated from VLM response quality)
    raw_output: str      # Original VLM text (for debugging)


@dataclass
class RelevanceScore:
    """Parsed result from a frame-relevance VLM call."""
    relevant: bool
    confidence: float    # 0.0 – 1.0
    reason: str
    raw_output: str


@dataclass
class AlignmentScore:
    """Parsed result from a TRAKE alignment VLM call."""
    match: bool
    confidence: float
    reason: str
    raw_output: str


class ResponseParser:
    """
    Converts raw Qwen2.5-VL text output into typed Python objects.

    Tries JSON parsing first, falls back to regex extraction,
    then falls back to a conservative default if everything fails.
    """

    # ----------------------------------------------------------
    # Q&A Answer Parsing
    # ----------------------------------------------------------

    def parse_qa_answer(self, raw_output: str) -> QAAnswer:
        """
        Parse raw VLM output as a Q&A answer.

        The VLM is prompted to answer directly (not JSON), so we just
        clean and normalise the text. Confidence is estimated heuristically:
        - Short, direct answers → high confidence
        - Long, hedging answers (có thể, có lẽ, I think...) → lower confidence
        """
        text = raw_output.strip()

        # Estimate confidence from output length and hedging language
        hedge_patterns = [
            r"\bcó thể\b", r"\bcó lẽ\b", r"\bkhông chắc\b",
            r"\bI think\b", r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b",
        ]
        hedging = any(re.search(p, text, re.IGNORECASE) for p in hedge_patterns)

        if len(text) < 20 and not hedging:
            confidence = 0.85
        elif hedging:
            confidence = 0.4
        else:
            confidence = 0.65

        return QAAnswer(answer=text, confidence=confidence, raw_output=raw_output)

    # ----------------------------------------------------------
    # Relevance Score Parsing
    # ----------------------------------------------------------

    def parse_relevance(self, raw_output: str) -> RelevanceScore:
        """
        Parse VLM response as a relevance JSON.
        Expected format: {"relevant": true, "confidence": 0.9, "reason": "..."}
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
        is_relevant = any(kw in text_lower for kw in ["true", "có", "yes", "phù hợp", "đúng"])
        return RelevanceScore(
            relevant=is_relevant,
            confidence=0.4,
            reason="(parsed via fallback)",
            raw_output=raw_output,
        )

    # ----------------------------------------------------------
    # TRAKE Alignment Score Parsing
    # ----------------------------------------------------------

    def parse_alignment(self, raw_output: str) -> AlignmentScore:
        """
        Parse VLM response as a TRAKE alignment JSON.
        Expected format: {"match": true, "confidence": 0.85, "reason": "..."}
        """
        parsed = self._try_parse_json(raw_output)
        if parsed:
            return AlignmentScore(
                match=bool(parsed.get("match", False)),
                confidence=float(parsed.get("confidence", 0.5)),
                reason=str(parsed.get("reason", "")),
                raw_output=raw_output,
            )

        # Fallback: keyword search
        text_lower = raw_output.lower()
        is_match = any(kw in text_lower for kw in ["true", "yes", "khớp", "đúng", "match"])
        return AlignmentScore(
            match=is_match,
            confidence=0.4,
            reason="(parsed via fallback)",
            raw_output=raw_output,
        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract and parse a JSON object from raw text."""
        # Direct parse
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Extract JSON block from markdown code fence
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Extract first {...} in the text
        json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        logger.debug(f"JSON parse failed, raw: {text[:100]}")
        return None
