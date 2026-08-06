"""
Abstract Base Extractor for AIC Video Retrieval System.

All feature extractors (OCR, ASR, Caption, Object) implement this interface,
ensuring a consistent batch-processing API for Kaggle Notebook pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseExtractor(ABC):
    """
    Interface that every feature extractor must implement.

    Subclasses must implement:
        extract_one()  — process a single item (image path or video path)
        extract_batch()— process multiple items (default: loops over extract_one)

    Output JSON schema per keyframe (stored in datasets/<modality>/<video_id>.json):
        {
          "video_id": "L21_V001",
          "keyframes": [
            {"n": 1, "frame_idx": 0, "pts_time": 0.0, "<modality>": <output>},
            ...
          ]
        }
    """

    @abstractmethod
    def extract_one(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Extract features from a single input (image or video file).

        Args:
            input_path: Absolute path to the input file
            **kwargs:   Extra arguments (e.g., n, frame_idx, pts_time for keyframes)

        Returns:
            Dict with extracted feature data
        """
        ...

    def extract_batch(
        self,
        input_paths: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Extract features from multiple inputs.
        Default: serial loop over extract_one().
        Override for GPU-batched implementations.
        """
        return [self.extract_one(p, **kwargs) for p in input_paths]

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'paddleocr', 'whisper', 'qwen25_vl'."""
        ...

    def is_available(self) -> bool:
        """Return False if required dependencies are not installed."""
        return True
