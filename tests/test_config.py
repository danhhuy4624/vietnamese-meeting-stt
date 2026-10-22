import pytest
from pydantic import ValidationError

from meeting_stt.config import Settings
from meeting_stt.constants import WHISPER_SAMPLE_RATE


def test_default_sample_rate_matches_whisper_contract() -> None:
    assert Settings().sample_rate == WHISPER_SAMPLE_RATE == 16_000


@pytest.mark.parametrize("sample_rate", [8_000, 32_000, 48_000, 44_100])
def test_settings_reject_non_16khz_sample_rates(sample_rate: int) -> None:
    with pytest.raises(ValidationError, match="16000"):
        Settings(sample_rate=sample_rate)
