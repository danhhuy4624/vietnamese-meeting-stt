from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import WHISPER_SAMPLE_RATE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    classifier_model_id: str = "hr16/PhoWhisper-small-vispeech-classifier-v3"
    vietnamese_asr_model_id: str = "vinai/PhoWhisper-medium"
    mixed_asr_model_id: str = "openai/whisper-small"
    device: str = "auto"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    enable_classifier: bool = True
    enable_mixed_asr: bool = True

    sample_rate: int = WHISPER_SAMPLE_RATE
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    min_utterance_ms: int = Field(default=400, ge=100)
    end_silence_ms: int = Field(default=700, ge=200)
    max_utterance_seconds: float = Field(default=28, gt=1, le=30)
    pre_roll_ms: int = Field(default=300, ge=0, le=1000)

    high_pass_hz: float = Field(default=80, ge=0, le=300)
    target_rms_db: float = Field(default=-22, ge=-45, le=-6)
    max_gain_db: float = Field(default=24, ge=0, le=36)

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65_535)
    log_level: str = "INFO"

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        if value != WHISPER_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate must be exactly {WHISPER_SAMPLE_RATE} Hz for this ASR pipeline; "
                "resample audio before sending it to the server."
            )
        return value

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def resolved_dtype(self) -> torch.dtype:
        if self.resolved_device == "cpu" and self.torch_dtype == "float16":
            return torch.float32
        return getattr(torch, self.torch_dtype)


@lru_cache
def get_settings() -> Settings:
    return Settings()
