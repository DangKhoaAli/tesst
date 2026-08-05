"""
Common Enums for AIC Video Retrieval System.
"""

from enum import Enum


class QueryType(str, Enum):
    """The 3 official AIC-HCMC query task types."""
    TEXTUAL_KIS = "textual_kis"   # Dạng 1: Tìm kiếm chính xác theo văn bản
    QA          = "qa"            # Dạng 2: Hỏi–Đáp (Visual Question Answering)
    TRAKE       = "trake"         # Dạng 3: Temporal Retrieval & Alignment of Key Events

class ModelType(str, Enum):
    CLIP = "clip"
    CLIP32 = "clip32"             # Pre-extracted CLIP-32 (from Kaggle .npy)
    SIGLIP = "siglip"
    SIGLIP2 = "siglip2"
    DINOV2 = "dinov2"
    PADDLE_OCR = "paddle_ocr"
    WHISPER = "whisper"
    FASTER_WHISPER = "faster_whisper"
    YOLO = "yolo"
    FLORENCE2 = "florence2"
    LLAVA = "llava"
    QWEN25_VL = "qwen25_vl"      # Qwen2.5-VL for Q&A and TRAKE alignment

class SearchStrategy(str, Enum):
    VISUAL_HEAVY = "visual_heavy"
    OCR_HEAVY = "ocr_heavy"
    CAPTION_HEAVY = "caption_heavy"
    HYBRID_FUSION = "hybrid_fusion"
    TEMPORAL_SEQUENCE = "temporal_sequence"

class IndexBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"
    BM25 = "bm25"

class TaskType(str, Enum):
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    EVALUATION = "evaluation"
    PREPROCESSING = "preprocessing"
