"""
OCR Extractor — Extract on-screen text from keyframe images using PaddleOCR.

Input:  Keyframe images (.jpg) from the Kaggle dataset
Output: JSON files per video with OCR text per keyframe

Output format (datasets/ocr/L21_V001.json):
    {
      "video_id": "L21_V001",
      "extractor": "paddleocr",
      "keyframes": [
        {
          "n": 1,
          "frame_idx": 0,
          "pts_time": 0.0,
          "texts": ["VTV1", "Bản tin 19h"],
          "raw": [{"text": "VTV1", "confidence": 0.98, "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.feature_extractors.base import BaseExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False


class OCRExtractor(BaseExtractor):
    """
    Extracts text from keyframe images using PaddleOCR.

    Supports both Vietnamese (vi) and English (en) text.
    Results are filtered by a minimum confidence threshold.

    Usage:
        extractor = OCRExtractor(lang="vi", min_confidence=0.7)
        extractor.load()
        result = extractor.extract_one("keyframes/L21/V001/5.jpg",
                                       n=5, frame_idx=450, pts_time=18.24)
    """

    def __init__(
        self,
        lang: str = "vi",           # PaddleOCR language code
        min_confidence: float = 0.6,
        use_gpu: bool = True,
        use_angle_cls: bool = True,  # Detect rotated text
    ):
        self.lang = lang
        self.min_confidence = min_confidence
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self._ocr = None

    @property
    def name(self) -> str:
        return "paddleocr"

    def is_available(self) -> bool:
        return _PADDLE_AVAILABLE

    def load(self) -> "OCRExtractor":
        """Load PaddleOCR model."""
        if not _PADDLE_AVAILABLE:
            raise ImportError("paddleocr not installed. Run: pip install paddleocr")
        logger.info(f"Loading PaddleOCR (lang={self.lang}, gpu={self.use_gpu})")
        self._ocr = _PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
        )
        logger.info("PaddleOCR loaded.")
        return self

    def extract_one(
        self,
        input_path: str,
        n: int = 0,
        frame_idx: int = 0,
        pts_time: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run OCR on a single keyframe image.

        Returns:
            {
              "n": 5,
              "frame_idx": 450,
              "pts_time": 18.24,
              "texts": ["VTV1", "Bản tin 19h"],
              "raw": [{"text": ..., "confidence": ..., "bbox": ...}]
            }
        """
        if self._ocr is None:
            raise RuntimeError("OCRExtractor not loaded. Call load() first.")

        texts = []
        raw = []

        try:
            result = self._ocr.ocr(input_path, cls=self.use_angle_cls)
            if result and result[0]:
                for line in result[0]:
                    bbox, (text, confidence) = line[0], line[1]
                    if confidence >= self.min_confidence and text.strip():
                        texts.append(text.strip())
                        raw.append({
                            "text": text.strip(),
                            "confidence": round(float(confidence), 4),
                            "bbox": bbox,
                        })
        except Exception as e:
            logger.warning(f"OCR failed for {input_path}: {e}")

        return {
            "n":         n,
            "frame_idx": frame_idx,
            "pts_time":  pts_time,
            "texts":     texts,
            "raw":       raw,
        }

    # ----------------------------------------------------------
    # Video-level batch processing
    # ----------------------------------------------------------

    def extract_video(
        self,
        video_id: str,
        keyframes_dir: str,
        map_keyframes_csv: str,
        output_dir: str,
        overwrite: bool = False,
    ) -> Path:
        """
        Extract OCR for all keyframes of one video and save to JSON.

        Args:
            video_id:         e.g. "L21_V001"
            keyframes_dir:    Root dir of keyframe images (Keyframes_L21/keyframes/L21_V001/)
            map_keyframes_csv: Path to L21_V001.csv
            output_dir:       Where to save L21_V001.json
            overwrite:        Skip if output already exists

        Returns:
            Path to the written JSON file
        """
        import pandas as pd

        out_path = Path(output_dir) / f"{video_id}.json"
        if out_path.exists() and not overwrite:
            logger.debug(f"[OCR] Skipping {video_id} (already exists)")
            return out_path

        # Parse CSV
        df = pd.read_csv(map_keyframes_csv)
        batch_id = video_id.split("_")[0]   # e.g. "L21"

        keyframe_results = []
        for _, row in df.iterrows():
            n = int(row["n"])
            img_path = str(
                Path(keyframes_dir)
                / f"Keyframes_{batch_id}"
                / "keyframes"
                / video_id
                / f"{n}.jpg"
            )
            if not Path(img_path).exists():
                logger.warning(f"Image not found: {img_path}")
                keyframe_results.append({
                    "n": n, "frame_idx": int(row["frame_idx"]),
                    "pts_time": float(row["pts_time"]),
                    "texts": [], "raw": [],
                })
                continue

            kf_result = self.extract_one(
                img_path,
                n=n,
                frame_idx=int(row["frame_idx"]),
                pts_time=float(row["pts_time"]),
            )
            keyframe_results.append(kf_result)

        # Save JSON
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "video_id":   video_id,
            "extractor":  self.name,
            "total":      len(keyframe_results),
            "keyframes":  keyframe_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        n_with_text = sum(1 for kf in keyframe_results if kf["texts"])
        logger.info(f"[OCR] {video_id}: {n_with_text}/{len(keyframe_results)} frames have text → {out_path}")
        return out_path
