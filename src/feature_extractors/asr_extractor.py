"""
ASR Extractor — Extract speech transcripts from video audio using Faster-Whisper.

Input:  Raw video files (.mp4) from the Kaggle dataset
Output: JSON files per video with timestamped transcript segments

Output format (datasets/subtitles/L21_V001.json):
    {
      "video_id": "L21_V001",
      "extractor": "faster_whisper",
      "language": "vi",
      "segments": [
        {"start": 0.0,  "end": 3.2,  "text": "Hôm nay tại hội nghị..."},
        {"start": 3.2,  "end": 7.1,  "text": "Bộ trưởng cho biết..."},
        ...
      ],
      "keyframe_asr": [
        {"n": 1, "frame_idx": 0,  "pts_time": 0.0,  "asr_text": "Hôm nay tại hội nghị..."},
        {"n": 2, "frame_idx": 90, "pts_time": 3.0,  "asr_text": "Bộ trưởng cho biết..."},
        ...
      ]
    }

The "keyframe_asr" list maps each keyframe to the ASR segment that covers its timestamp,
making it directly joinable with keyframe_master.parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.feature_extractors.base import BaseExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from faster_whisper import WhisperModel as _WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False


class ASRExtractor(BaseExtractor):
    """
    Extracts speech transcripts from video audio using Faster-Whisper.

    Faster-Whisper is a CTranslate2-based implementation of OpenAI Whisper,
    offering 2-4× speed improvement with the same accuracy.

    Usage:
        extractor = ASRExtractor(model_size="large-v3", device="cuda")
        extractor.load()
        result = extractor.extract_video(
            video_id="L21_V001",
            video_path="/kaggle/input/.../L21_V001.mp4",
            map_keyframes_csv="/kaggle/input/.../L21_V001.csv",
            output_dir="/kaggle/working/subtitles",
        )
    """

    def __init__(
        self,
        model_size: str = "large-v3",  # large-v3 for best accuracy
        device: str = "cuda",
        compute_type: str = "float16",  # float16 for GPU, int8 for CPU
        language: str = "vi",           # Vietnamese
        beam_size: int = 5,
    ):
        self.model_size   = model_size
        self.device       = device
        self.compute_type = compute_type
        self.language     = language
        self.beam_size    = beam_size
        self._model = None

    @property
    def name(self) -> str:
        return "faster_whisper"

    def is_available(self) -> bool:
        return _WHISPER_AVAILABLE

    def load(self) -> "ASRExtractor":
        """Load Faster-Whisper model."""
        if not _WHISPER_AVAILABLE:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")
        logger.info(
            f"Loading Faster-Whisper {self.model_size} "
            f"({self.device}, {self.compute_type})"
        )
        self._model = _WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Faster-Whisper loaded.")
        return self

    def extract_one(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe a single video file.
        Returns list of {start, end, text} segments.
        """
        if self._model is None:
            raise RuntimeError("ASRExtractor not loaded. Call load() first.")

        segments_out = []
        try:
            segments, info = self._model.transcribe(
                input_path,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,           # Skip silence
                vad_parameters={"min_silence_duration_ms": 500},
            )
            for seg in segments:
                segments_out.append({
                    "start": round(float(seg.start), 3),
                    "end":   round(float(seg.end), 3),
                    "text":  seg.text.strip(),
                })
            logger.debug(f"[ASR] {input_path}: {len(segments_out)} segments")
        except Exception as e:
            logger.warning(f"[ASR] Failed for {input_path}: {e}")

        return {"segments": segments_out}

    # ----------------------------------------------------------
    # Video-level processing with keyframe alignment
    # ----------------------------------------------------------

    def extract_video(
        self,
        video_id: str,
        video_path: str,
        map_keyframes_csv: str,
        output_dir: str,
        overwrite: bool = False,
    ) -> Path:
        """
        Transcribe a video and align transcript to keyframe timestamps.

        For each keyframe n with pts_time T, finds the ASR segment that
        contains T (start ≤ T < end). If no segment covers T, uses the
        nearest segment by start time.

        Args:
            video_id:          e.g. "L21_V001"
            video_path:        Absolute path to .mp4 file
            map_keyframes_csv: Path to L21_V001.csv
            output_dir:        Where to save L21_V001.json
            overwrite:         Skip if output exists

        Returns:
            Path to the written JSON file
        """
        import pandas as pd

        out_path = Path(output_dir) / f"{video_id}.json"
        if out_path.exists() and not overwrite:
            logger.debug(f"[ASR] Skipping {video_id} (already exists)")
            return out_path

        # 1. Transcribe video
        asr_data = self.extract_one(video_path)
        segments = asr_data["segments"]

        # 2. Load keyframe timestamps from CSV
        df = pd.read_csv(map_keyframes_csv)

        # 3. Align each keyframe to nearest ASR segment
        keyframe_asr = []
        for _, row in df.iterrows():
            pts = float(row["pts_time"])
            asr_text = self._find_segment_text(segments, pts)
            keyframe_asr.append({
                "n":         int(row["n"]),
                "frame_idx": int(row["frame_idx"]),
                "pts_time":  pts,
                "asr_text":  asr_text,
            })

        # 4. Save JSON
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "video_id":    video_id,
            "extractor":   self.name,
            "language":    self.language,
            "segments":    segments,
            "keyframe_asr": keyframe_asr,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        n_with_text = sum(1 for kf in keyframe_asr if kf["asr_text"])
        logger.info(
            f"[ASR] {video_id}: {len(segments)} segments, "
            f"{n_with_text}/{len(keyframe_asr)} keyframes aligned → {out_path}"
        )
        return out_path

    def _find_segment_text(self, segments: List[Dict], pts_time: float) -> str:
        """
        Find the ASR text covering a given timestamp.
        Falls back to the nearest segment if no exact match.
        """
        if not segments:
            return ""
        # Exact cover
        for seg in segments:
            if seg["start"] <= pts_time < seg["end"]:
                return seg["text"]
        # Nearest by start time
        nearest = min(segments, key=lambda s: abs(s["start"] - pts_time))
        return nearest["text"]
