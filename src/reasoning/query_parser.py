"""
Query Parser for AIC Video Retrieval System.

Converts raw natural-language queries into structured objects
(TextualKISQuery, QAQuery, TRAKEQuery) using a lightweight
rule-based extractor backed by an optional LLM for refinement.

Optimization v2:
- build_retrieval_text() appends spatial, color, object, and OCR hints
  to the CLIP encoding prompt for maximum KIS / Q&A recall.
- Vi→En keyword mapping boosts CLIP cross-lingual alignment (CLIP was
  trained predominantly on English captions).
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.common.types import TextualKISQuery, QAQuery, TRAKEQuery, EventStep
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Colour keywords (vi + en)
# ============================================================
_COLORS_VI = ["đỏ", "xanh", "vàng", "trắng", "đen", "tím", "hồng", "cam", "nâu", "xám"]
_COLORS_EN = ["red", "blue", "green", "yellow", "white", "black", "purple",
              "pink", "orange", "brown", "grey", "gray"]
_ALL_COLORS = _COLORS_VI + _COLORS_EN

# ============================================================
# Scene / environment keywords
# ============================================================
_SCENE_KEYWORDS = {
    "outdoor": ["ngoài trời", "ngoài sân", "sân vận động", "đường phố",
                "outdoor", "outside", "street", "stadium", "field"],
    "indoor":  ["trong nhà", "phòng họp", "hội trường", "studio",
                "indoor", "inside", "hall", "room"],
    "press_conference": ["họp báo", "press conference", "briefing"],
    "ceremony": ["lễ", "ceremony", "award", "trao giải"],
    "sport":    ["thể thao", "thi đấu", "sport", "athletic", "race", "jump"],
}

# ============================================================
# Object keywords — common AIC topic areas
# ============================================================
_OBJECT_PATTERNS = {
    "person": r"\b(người|diễn giả|phát ngôn viên|vận động viên|cầu thủ|người phát biểu"
              r"|speaker|athlete|player|person|man|woman)\b",
    "vehicle": r"\b(xe|ô tô|xe buýt|xe tải|car|bus|truck|vehicle)\b",
    "flag":    r"\b(cờ|flag|banner)\b",
    "screen":  r"\b(màn hình|bảng|screen|board|sign)\b",
}


class QueryParser:
    """
    Converts raw query text into a structured query object.

    Usage:
        parser = QueryParser()
        kis_query = parser.parse_kis("Tìm video người dẫn mặc áo đỏ...")
        qa_query  = parser.parse_qa("Trong video lễ trao giải...", "Có bao nhiêu người?")
    """

    def parse_kis(
        self,
        raw_text: str,
        top_k: int = 100,
    ) -> TextualKISQuery:
        """
        Parse a KIS (Textual Known-Item Search) query.

        Extracts:
        - Detected object types (person, vehicle, ...)
        - Colors mentioned
        - Scene context (outdoor/indoor/...)
        - Any OCR text hints (quoted strings, uppercase words)
        - Spatial hints (left/right/behind/above...)

        Args:
            raw_text: Full natural-language query string
            top_k:    Number of retrieval candidates to request

        Returns:
            TextualKISQuery with structured fields populated
        """
        text_lower = raw_text.lower()

        # --- Extract colors ---
        colors = [c for c in _ALL_COLORS if c in text_lower]

        # --- Extract objects ---
        objects: List[str] = []
        for obj_label, pattern in _OBJECT_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                objects.append(obj_label)

        # --- Extract scene ---
        scene = ""
        for scene_label, keywords in _SCENE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                scene = scene_label
                break

        # --- Extract OCR hints (quoted text or ALL CAPS words) ---
        ocr_hints: List[str] = []
        # Quoted text: "..." or '...'
        quoted = re.findall(r'["\']([^"\']{2,})["\']', raw_text)
        ocr_hints.extend(quoted)
        # ALL CAPS words (likely on-screen text like VTV1, VNPT...)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', raw_text)
        ocr_hints.extend([w for w in caps_words if w not in ("TV", "HD", "OK")])

        # --- Extract spatial hints ---
        spatial_patterns = [
            r"(phía\s+\w+|bên\s+\w+)",                # Vietnamese: phía sau, bên trái
            r"(behind|in front|to the (left|right)|above|below|next to)",
        ]
        spatial_hints: List[str] = []
        for pat in spatial_patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                hint = m if isinstance(m, str) else " ".join(m).strip()
                if hint:
                    spatial_hints.append(hint)

        query = TextualKISQuery(
            raw_text=raw_text,
            parsed_objects=list(set(objects)),
            parsed_scene=scene,
            parsed_colors=list(set(colors)),
            ocr_keywords=ocr_hints,
            spatial_hints=spatial_hints,
            top_k=top_k,
        )

        logger.debug(
            f"[QueryParser] KIS parsed: objects={query.parsed_objects}, "
            f"colors={query.parsed_colors}, scene='{query.parsed_scene}'"
        )
        return query

    def parse_qa(
        self,
        event_description: str,
        question: str,
        answer_language: str = "auto",
        top_k: int = 20,
    ) -> QAQuery:
        """
        Parse a Q&A query.

        Infers answer_type from question keywords:
          - "bao nhiêu / how many" → "count"
          - "ai / who"             → "name"
          - "có / không / yes"     → "yes_no"
          - otherwise              → "description"
        """
        q_lower = question.lower()

        if any(kw in q_lower for kw in ["bao nhiêu", "how many", "mấy", "số lượng"]):
            answer_type = "count"
        elif any(kw in q_lower for kw in ["ai ", "who ", "tên "]):
            answer_type = "name"
        elif any(kw in q_lower for kw in ["có không", "yes or no", "có phải", "is it"]):
            answer_type = "yes_no"
        else:
            answer_type = "description"

        return QAQuery(
            event_description=event_description,
            question=question,
            answer_type=answer_type,
            answer_language=answer_language,
            top_k=top_k,
        )

    # ============================================================
    # Vietnamese → English keyword maps for CLIP cross-lingual boost
    # (CLIP was trained on English captions; EN terms ↑ recall)
    # ============================================================
    _VI_TO_EN_COLORS = {
        "đỏ": "red", "xanh": "blue", "vàng": "yellow", "trắng": "white",
        "đen": "black", "tím": "purple", "hồng": "pink", "cam": "orange",
        "nâu": "brown", "xám": "gray",
    }
    _VI_TO_EN_SCENES = {
        "ngoài trời": "outdoor", "trong nhà": "indoor", "sân khấu": "stage",
        "sân vận động": "stadium", "phòng họp": "conference room",
        "đường phố": "street", "họp báo": "press conference",
        "lễ trao giải": "award ceremony",
    }
    _VI_TO_EN_SPATIAL = {
        "phía sau": "behind", "phía trước": "in front",
        "bên trái": "on the left", "bên phải": "on the right",
        "phía trên": "above", "phía dưới": "below",
        "bên cạnh": "next to",
    }

    def build_retrieval_text(self, kis_query: "TextualKISQuery") -> str:
        """
        Build the final retrieval text string to encode with CLIP.

        Strategy (v2 — optimized):
          1. Start with full raw_text (Vietnamese context preserved).
          2. Append English translations of colors, scenes, and objects.
             → Boosts CLIP recall because CLIP is English-dominant.
          3. Append spatial relationship hints for precise localization.
          4. Append quoted OCR keywords (text visible on screen).
        """
        parts = [kis_query.raw_text]

        # 1. Translate & append colors (VI + EN for dual signal)
        color_en_terms = []
        for vi_color in kis_query.parsed_colors:
            en = self._VI_TO_EN_COLORS.get(vi_color, vi_color)
            color_en_terms.append(en)
        if color_en_terms:
            parts.append("wearing " + " and ".join(color_en_terms))

        # 2. Translate & append scene context
        if kis_query.parsed_scene:
            scene_text = kis_query.parsed_scene
            # Try to translate each Vietnamese phrase in the scene
            for vi, en in self._VI_TO_EN_SCENES.items():
                if vi in kis_query.raw_text.lower():
                    scene_text = en
                    break
            parts.append(scene_text)

        # 3. Append object labels (CLIP responds well to English nouns)
        if kis_query.parsed_objects:
            parts.append(", ".join(kis_query.parsed_objects))

        # 4. Append spatial hints translated to English
        spatial_en = []
        for hint in kis_query.spatial_hints:
            translated = hint
            for vi, en in self._VI_TO_EN_SPATIAL.items():
                if vi in hint:
                    translated = en
                    break
            spatial_en.append(translated)
        if spatial_en:
            parts.append(", ".join(spatial_en))

        # 5. Append OCR keywords (for frames with on-screen text)
        if kis_query.ocr_keywords:
            parts.append("text on screen: " + " ".join(kis_query.ocr_keywords))

        return ". ".join(parts)
