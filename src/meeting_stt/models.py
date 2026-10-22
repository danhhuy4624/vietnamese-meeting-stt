from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    WhisperForAudioClassification,
    WhisperForConditionalGeneration,
)

from .constants import WHISPER_SAMPLE_RATE
from .schemas import Classification

logger = logging.getLogger(__name__)

VI_CHARS = set(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
VI_WORDS = {
    "anh",
    "chị",
    "chúng",
    "của",
    "được",
    "không",
    "mình",
    "những",
    "phòng",
    "tôi",
    "và",
}
EN_WORDS = {
    "a", "and", "are", "for", "in", "is", "meeting", "of", "the", "this", "to", "we", "you",
}


@dataclass(slots=True)
class ASRCandidate:
    text: str
    confidence: float
    language: str
    model_id: str


def detect_text_language(text: str) -> str:
    words = set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))
    vi_score = sum(character in VI_CHARS for character in text.lower()) + 2 * len(words & VI_WORDS)
    en_score = 2 * len(words & EN_WORDS)
    if vi_score and en_score:
        return "mixed"
    if vi_score:
        return "vi"
    if en_score:
        return "en"
    return "unknown"


class WhisperASR:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype) -> None:
        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRCandidate:
        if sample_rate != WHISPER_SAMPLE_RATE:
            raise ValueError(
                f"Whisper ASR requires {WHISPER_SAMPLE_RATE} Hz audio, got {sample_rate} Hz."
            )
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(
            self.device,
            dtype=self.model.dtype,
        )
        attention_mask = inputs.attention_mask.to(self.device)

        output = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            task="transcribe",
            language=None,
            max_new_tokens=160,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        text = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        confidence = self._confidence(output)
        return ASRCandidate(
            text=text,
            confidence=confidence,
            language=detect_text_language(text),
            model_id=self.model_id,
        )

    def _confidence(self, output: object) -> float:
        scores = getattr(output, "scores", None)
        sequences = getattr(output, "sequences", None)
        if not scores or sequences is None:
            return 0.5
        transition = self.model.compute_transition_scores(
            sequences, scores, normalize_logits=True
        )[0]
        valid = transition[torch.isfinite(transition)]
        if valid.numel() == 0:
            return 0.0
        # Geometric mean token probability. Clamp for a stable API value.
        return float(np.clip(math.exp(float(valid.mean().cpu())), 0.0, 1.0))


class SpeakerClassifier:
    """Gender/dialect tagging using the exact checkpoint requested by the user."""

    def __init__(self, model_id: str, device: str, dtype: torch.dtype) -> None:
        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = WhisperForAudioClassification.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def classify(self, audio: np.ndarray, sample_rate: int) -> Classification:
        if sample_rate != WHISPER_SAMPLE_RATE:
            raise ValueError(
                f"Speaker classification requires {WHISPER_SAMPLE_RATE} Hz audio, "
                f"got {sample_rate} Hz."
            )
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        features = inputs.input_features.to(self.device, dtype=self.model.dtype)
        logits = self.model(input_features=features).logits[0]
        probabilities = torch.softmax(logits.float(), dim=-1)
        index = int(probabilities.argmax())
        return Classification(
            label=self.model.config.id2label[index],
            confidence=float(probabilities[index].cpu()),
        )


def select_candidate(vietnamese: ASRCandidate, mixed: ASRCandidate | None) -> ASRCandidate:
    if mixed is None or not mixed.text:
        return vietnamese
    if not vietnamese.text:
        return mixed

    # PhoWhisper remains preferred for Vietnamese unless the multilingual pass is
    # clearly more confident or detects English/code-switching.
    if mixed.language in {"en", "mixed"} and mixed.confidence >= vietnamese.confidence * 0.82:
        return mixed
    if mixed.confidence > vietnamese.confidence + 0.08:
        return mixed
    return vietnamese
