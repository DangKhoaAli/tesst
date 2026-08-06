"""
Qwen2.5-VL Client — Shared VLM inference engine for Q&A and TRAKE alignment.

This client is a lightweight wrapper around the loaded Qwen2.5-VL model,
used by both:
  - QAPipeline     (answer extraction from keyframes)
  - TRAKEPipeline  (event alignment verification)

Separating the model loading from business logic allows the model to be
loaded once and reused across both pipelines without duplication.

Model VRAM requirements:
  - Qwen2.5-VL-7B-Instruct (fp16):  ~14 GB VRAM (T4 on Kaggle fits)
  - Qwen2.5-VL-7B-Instruct (4bit):  ~8  GB VRAM (safer for T4)
  - Qwen2.5-VL-3B-Instruct (fp16):  ~7  GB VRAM (faster, slightly lower quality)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.prompt_templates import (
    build_qa_answer_prompt,
    build_qa_relevance_prompt,
    build_trake_align_prompt,
    SYSTEM_QA, SYSTEM_VERIFY,
)
from src.llm.response_parser import ResponseParser, QAAnswer, RelevanceScore, AlignmentScore
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    _QWEN_AVAILABLE = True
except ImportError:
    _QWEN_AVAILABLE = False


class QwenVLClient:
    """
    Qwen2.5-VL inference client for Visual Question Answering.

    Provides three high-level methods used by the Q&A and TRAKE pipelines:
      - answer_question()    → QAAnswer
      - score_relevance()    → RelevanceScore
      - score_alignment()    → AlignmentScore  (for TRAKE)

    Usage:
        client = QwenVLClient(load_in_4bit=True)
        client.load()

        answer = client.answer_question(
            image_path="keyframes/L21/V001/5.jpg",
            event_description="Lễ trao giải âm nhạc...",
            question="Có bao nhiêu người lên sân khấu?",
        )
        # answer.answer → "5"
        # answer.confidence → 0.85
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 200,
    ):
        self.model_name     = model_name
        self.device         = device
        self.load_in_4bit   = load_in_4bit
        self.max_new_tokens = max_new_tokens

        self._model     = None
        self._processor = None
        self._parser    = ResponseParser()

    def load(self) -> "QwenVLClient":
        """Load Qwen2.5-VL model and processor."""
        if not _QWEN_AVAILABLE:
            raise ImportError(
                "transformers or qwen-vl-utils not installed. "
                "Run: pip install transformers qwen-vl-utils accelerate bitsandbytes"
            )

        logger.info(f"Loading {self.model_name} (4bit={self.load_in_4bit}, device={self.device})")

        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.float16}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.device

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, **model_kwargs
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        logger.info(f"Qwen2.5-VL ready ({self.model_name})")
        return self

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def answer_question(
        self,
        image_path: str,
        event_description: str,
        question: str,
        answer_language: str = "auto",
    ) -> QAAnswer:
        """
        Answer a question about the event shown in the keyframe.

        Args:
            image_path:        Path to keyframe .jpg
            event_description: Context about the event
            question:          The question to answer
            answer_language:   "vi" | "en" | "auto"

        Returns:
            QAAnswer with answer text and confidence
        """
        messages = build_qa_answer_prompt(
            image_path, event_description, question, answer_language
        )
        raw = self._infer(messages, system=SYSTEM_QA)
        return self._parser.parse_qa_answer(raw)

    def score_relevance(
        self,
        image_path: str,
        event_description: str,
        question: str,
    ) -> RelevanceScore:
        """
        Score how relevant a keyframe is for answering the question.
        Used to re-rank candidate frames in the Q&A pipeline.

        Returns:
            RelevanceScore with relevant bool and 0-1 confidence
        """
        messages = build_qa_relevance_prompt(image_path, event_description, question)
        raw = self._infer(messages, system=SYSTEM_VERIFY)
        return self._parser.parse_relevance(raw)

    def score_alignment(
        self,
        image_path: str,
        event_name: str,
        semantic_keyframe_hint: str,
    ) -> AlignmentScore:
        """
        Verify if a keyframe matches a specific TRAKE event moment.
        Used in TRAKE Phase 2 per-event alignment (Sprint 5).

        Returns:
            AlignmentScore with match bool and 0-1 confidence
        """
        messages = build_trake_align_prompt(image_path, event_name, semantic_keyframe_hint)
        raw = self._infer(messages, system=SYSTEM_VERIFY)
        return self._parser.parse_alignment(raw)

    # ----------------------------------------------------------
    # Internal inference
    # ----------------------------------------------------------

    def _infer(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> str:
        """
        Run one VLM forward pass and return the generated text.

        Args:
            messages: Chat messages list (user turn with image + text)
            system:   Optional system prompt prepended to the conversation
        """
        self._check_loaded()

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        try:
            text = self._processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(full_messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            result = self._processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
            return result

        except Exception as e:
            logger.warning(f"[QwenVLClient] Inference failed: {e}")
            return ""

    def _check_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("QwenVLClient not loaded. Call client.load() first.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
