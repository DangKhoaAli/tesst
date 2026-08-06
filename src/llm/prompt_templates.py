"""
Prompt Templates for AIC Video Retrieval System.

Centralised prompt management for all LLM/VLM tasks:
  1. QA_ANSWER    — Extract answer from a keyframe image given a question
  2. QA_RELEVANCE — Score how relevant a frame is to an event description
  3. TRAKE_ALIGN  — Verify if a keyframe matches a specific event moment
  4. CAPTION      — Generate descriptive caption (delegated to captioner.py)

Each template is a callable that accepts context variables and returns
a formatted messages list compatible with Qwen2.5-VL chat format.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ============================================================
# 1. Q&A Answer Extraction
# ============================================================

QA_ANSWER_SYSTEM = (
    "Bạn là một hệ thống phân tích video chính xác. "
    "Dựa trên hình ảnh keyframe được cung cấp, hãy trả lời câu hỏi một cách ngắn gọn và chính xác. "
    "Chỉ trả lời thông tin được hỏi, không giải thích thêm."
)

QA_ANSWER_SYSTEM_EN = (
    "You are a precise video analysis system. "
    "Based on the keyframe image provided, answer the question concisely and accurately. "
    "Only provide the answer, no extra explanation."
)


def build_qa_answer_prompt(
    image_path: str,
    event_description: str,
    question: str,
    answer_language: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Build Qwen2.5-VL messages for extracting an answer from a keyframe.

    Args:
        image_path:        Absolute path to the keyframe .jpg
        event_description: Context about the event in the video
        question:          Specific question to answer
        answer_language:   "vi" | "en" | "auto"

    Returns:
        messages list for Qwen2.5-VL processor
    """
    if answer_language == "vi":
        lang_instruction = "Trả lời bằng tiếng Việt."
    elif answer_language == "en":
        lang_instruction = "Answer in English."
    else:
        lang_instruction = "Trả lời bằng tiếng Việt hoặc tiếng Anh tùy ngữ cảnh."

    user_text = (
        f"Bối cảnh sự kiện: {event_description}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"{lang_instruction} Trả lời ngắn gọn nhất có thể."
    )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": user_text},
            ],
        }
    ]


# ============================================================
# 2. Q&A Frame Relevance Scoring
# ============================================================

def build_qa_relevance_prompt(
    image_path: str,
    event_description: str,
    question: str,
) -> List[Dict[str, Any]]:
    """
    Score whether this keyframe is the right moment to answer the question.

    Returns a prompt asking the VLM to respond with JSON:
        {"relevant": true/false, "confidence": 0-1, "reason": "..."}
    """
    user_text = (
        f"Bối cảnh: {event_description}\n"
        f"Câu hỏi cần trả lời: {question}\n\n"
        "Khung hình này có phải là khoảnh khắc phù hợp nhất để trả lời câu hỏi trên không?\n"
        "Trả lời theo định dạng JSON:\n"
        '{"relevant": true hoặc false, "confidence": số từ 0 đến 1, "reason": "lý do ngắn"}'
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": user_text},
            ],
        }
    ]


# ============================================================
# 3. TRAKE Event Alignment (Sprint 5)
# ============================================================

def build_trake_align_prompt(
    image_path: str,
    event_name: str,
    semantic_keyframe_hint: str,
) -> List[Dict[str, Any]]:
    """
    Verify if a keyframe matches the semantic moment described by a TRAKE event step.

    Returns a prompt asking for JSON:
        {"match": true/false, "confidence": 0-1, "reason": "..."}
    """
    user_text = (
        f"Sự kiện cần xác định: {event_name}\n"
        f"Khoảnh khắc cụ thể: {semantic_keyframe_hint}\n\n"
        "Khung hình này có khớp với khoảnh khắc được mô tả ở trên không?\n"
        "Trả lời theo định dạng JSON:\n"
        '{"match": true hoặc false, "confidence": số từ 0 đến 1, "reason": "lý do ngắn"}'
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": user_text},
            ],
        }
    ]


# ============================================================
# 4. System messages (reusable)
# ============================================================

SYSTEM_QA   = QA_ANSWER_SYSTEM
SYSTEM_VERIFY = (
    "Bạn là hệ thống xác minh khung hình video. "
    "Hãy phân tích kỹ hình ảnh và trả lời theo đúng định dạng JSON được yêu cầu."
)
