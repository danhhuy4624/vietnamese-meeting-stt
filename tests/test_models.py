import numpy as np
import pytest

from meeting_stt.models import SpeakerClassifier, WhisperASR


@pytest.mark.parametrize(
    ("model_type", "method_name"),
    [(WhisperASR, "transcribe"), (SpeakerClassifier, "classify")],
)
def test_models_reject_non_16khz_audio_before_model_access(
    model_type: type,
    method_name: str,
) -> None:
    instance = object.__new__(model_type)
    method = getattr(instance, method_name)

    with pytest.raises(ValueError, match="16000"):
        method(np.zeros(320, dtype=np.float32), 8_000)
