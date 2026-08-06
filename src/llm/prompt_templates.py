"""
Prompt Templates for AIC Video Retrieval System (v2 — Accuracy-Optimized).

Changes from v1:
- QA: Combined 1-call approach (relevance + answer in one prompt)
  → Saves VRAM / latency, more coherent reasoning
- QA: answer_type-specific instructions (count / color / name / yes_no / description)
  → Forces structured, short answers that are easy to parse
- TRAKE: Chain-of-thought "describe first, then judge"
  → Forces VLM to look at the image carefully before deciding
- System prompts: Explicit output format examples
"""

from __future__ import annotations

from typing import Any, Dict, List


# ============================================================
# System Prompts
# ============================================================

SYSTEM_QA = (
    "Bạn là một hệ thống phân tích keyframe video cho cuộc thi AI Challenge. "
    "Nhiệm vụ của bạn là quan sát kỹ hình ảnh và trả lời câu hỏi dựa trên những gì "
    "THỰC SỰ NHÌN THẤY trong ảnh. "
    "Luôn trả lời theo đúng định dạng JSON được yêu cầu. "
    "Nếu không thể xác định câu trả lời từ ảnh, đặt found=false."
)

SYSTEM_VERIFY = (
    "Bạn là hệ thống xác minh keyframe video. "
    "Hãy mô tả ngắn gọn những gì bạn nhìn thấy trong ảnh, "
    "sau đó đánh giá xem ảnh có khớp với yêu cầu không. "
    "Trả lời đúng định dạng JSON được yêu cầu."
)

SYSTEM_TRAKE = (
    "Bạn là hệ thống nhận dạng khoảnh khắc thể thao trong video. "
    "Quan sát kỹ toàn bộ hình ảnh: tư thế cơ thể, vị trí chân tay, "
    "các vật thể xung quanh. Sau đó đánh giá xem khoảnh khắc này "
    "có khớp với mô tả không. Trả lời theo định dạng JSON."
)


# ============================================================
# 1. Q&A — Combined Answer Extraction (1 call, replaces 2-step)
# ============================================================

_QA_TYPE_INSTRUCTIONS = {
    "count": (
        "Câu hỏi yêu cầu ĐẾM SỐ LƯỢNG. "
        "Hãy đếm cẩn thận rồi trả lời bằng một con số. "
        "Ví dụ answer: \"5\" hoặc \"3 người\"."
    ),
    "color": (
        "Câu hỏi yêu cầu xác định MÀU SẮC. "
        "Hãy mô tả màu sắc cụ thể (đỏ, xanh lam, xanh lá, vàng, trắng, đen, v.v.). "
        "Ví dụ answer: \"màu đỏ\" hoặc \"áo trắng\"."
    ),
    "name": (
        "Câu hỏi yêu cầu xác định TÊN NGƯỜI hoặc TỔ CHỨC. "
        "Nếu có chữ trong ảnh, đọc chính xác. "
        "Ví dụ answer: \"Nguyễn Văn A\" hoặc \"VTV1\"."
    ),
    "yes_no": (
        "Câu hỏi yêu cầu trả lời CÓ hoặc KHÔNG. "
        "Ví dụ answer: \"Có\" hoặc \"Không\"."
    ),
    "description": (
        "Hãy trả lời ngắn gọn, chỉ những gì được hỏi. "
        "Không giải thích thêm. Tối đa 10 từ."
    ),
}


def build_qa_combined_prompt(
    image_path: str,
    event_description: str,
    question: str,
    answer_type: str = "description",
    answer_language: str = "vi",
) -> List[Dict[str, Any]]:
    """
    Single-call QA prompt: combines relevance check + answer extraction.

    Returns JSON:
        {
            "found": true/false,        // Can the question be answered from this frame?
            "answer": "...",            // Short answer (empty string if found=false)
            "confidence": 0.0-1.0,     // How confident are you?
            "observation": "..."        // Brief description of what you see (1 sentence)
        }

    Args:
        image_path:        Absolute path to keyframe .jpg
        event_description: Context about the event
        question:          Specific question to answer
        answer_type:       "count" | "color" | "name" | "yes_no" | "description"
        answer_language:   "vi" | "en" | "auto"
    """
    type_instruction = _QA_TYPE_INSTRUCTIONS.get(answer_type, _QA_TYPE_INSTRUCTIONS["description"])

    if answer_language == "en":
        lang_note = "Answer in English."
    else:
        lang_note = "Trả lời bằng tiếng Việt."

    user_text = (
        f"Bối cảnh sự kiện: {event_description}\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Hướng dẫn: {type_instruction} {lang_note}\n\n"
        "Trả lời theo định dạng JSON sau (không thêm gì ngoài JSON):\n"
        '{"found": true/false, "answer": "câu trả lời ngắn", '
        '"confidence": 0.0-1.0, "observation": "mô tả 1 câu về ảnh"}'
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


def build_qa_answer_prompt(
    image_path: str,
    event_description: str,
    question: str,
    answer_language: str = "auto",
    answer_type: str = "description",
) -> List[Dict[str, Any]]:
    """
    Backward-compatible wrapper — delegates to build_qa_combined_prompt.
    Kept for compatibility with existing QwenVLClient.answer_question() calls.
    """
    return build_qa_combined_prompt(
        image_path=image_path,
        event_description=event_description,
        question=question,
        answer_type=answer_type,
        answer_language=answer_language if answer_language != "auto" else "vi",
    )


# ============================================================
# 2. Q&A Frame Relevance (kept for backward-compat, rarely used now)
# ============================================================

def build_qa_relevance_prompt(
    image_path: str,
    event_description: str,
    question: str,
) -> List[Dict[str, Any]]:
    """
    Legacy 2-step relevance check. Now superseded by build_qa_combined_prompt.
    Still available for explicit 2-step mode if needed.
    """
    user_text = (
        f"Bối cảnh: {event_description}\n"
        f"Câu hỏi cần trả lời: {question}\n\n"
        "Khung hình này có chứa đủ thông tin để trả lời câu hỏi trên không?\n"
        'Trả lời JSON: {"relevant": true/false, "confidence": 0.0-1.0, "reason": "lý do 1 câu"}'
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
# 3. TRAKE Event Alignment — Chain-of-Thought
# ============================================================

def build_trake_align_prompt(
    image_path: str,
    event_name: str,
    semantic_keyframe_hint: str,
    activity_name: str = "",
) -> List[Dict[str, Any]]:
    """
    TRAKE alignment verification with chain-of-thought.

    Ask VLM to:
      1. First describe what it sees (forces it to look)
      2. Then compare with the target event description
      3. Return structured JSON

    Returns JSON:
        {
            "observation": "what I see in the image",
            "match": true/false,
            "confidence": 0.0-1.0,
            "reason": "why it matches or not"
        }
    """
    activity_ctx = f"Hoạt động thể thao: {activity_name}\n" if activity_name else ""

    user_text = (
        f"{activity_ctx}"
        f"Sự kiện cần xác định: [{event_name}]\n"
        f"Mô tả khoảnh khắc cụ thể: {semantic_keyframe_hint}\n\n"
        "Bước 1: Mô tả ngắn gọn (1-2 câu) những gì bạn nhìn thấy trong ảnh này "
        "(tư thế cơ thể, hành động, vị trí so với vật thể xung quanh).\n"
        "Bước 2: Đánh giá xem ảnh có khớp với khoảnh khắc mô tả ở trên không.\n\n"
        "Trả lời JSON (không thêm gì ngoài JSON):\n"
        '{"observation": "mô tả ảnh", "match": true/false, '
        '"confidence": 0.0-1.0, "reason": "lý do khớp hoặc không khớp"}'
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
# 4. Caption Generation (used by feature extractor)
# ============================================================

CAPTION_SYSTEM = (
    "You are a concise video caption generator. "
    "Describe the keyframe in 1-2 sentences, focusing on: "
    "people (count, clothing, actions), objects, location/setting, "
    "and any visible text. Be specific and factual."
)


def build_caption_prompt(image_path: str) -> List[Dict[str, Any]]:
    """Generate a descriptive caption for a keyframe image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {
                    "type": "text",
                    "text": (
                        "Describe this video keyframe concisely. "
                        "Include: number of people, what they're wearing/doing, "
                        "the setting/location, and any visible text or logos. "
                        "Max 2 sentences."
                    ),
                },
            ],
        }
    ]
