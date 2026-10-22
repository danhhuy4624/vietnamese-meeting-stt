from meeting_stt.models import ASRCandidate, detect_text_language, select_candidate


def candidate(text: str, confidence: float, language: str, model: str) -> ASRCandidate:
    return ASRCandidate(text, confidence, language, model)


def test_language_heuristic() -> None:
    assert detect_text_language("Chúng ta bắt đầu cuộc họp") == "vi"
    assert detect_text_language("We can start the meeting") == "en"
    assert detect_text_language("Chúng ta review the deployment") == "mixed"


def test_mixed_candidate_wins_for_code_switching() -> None:
    vi = candidate("Chúng ta triển khai", 0.82, "vi", "pho")
    mixed = candidate("Chúng ta review the deployment", 0.72, "mixed", "whisper")
    assert select_candidate(vi, mixed) is mixed


def test_phowhisper_remains_preferred_for_vietnamese() -> None:
    vi = candidate("Chúng ta bắt đầu", 0.82, "vi", "pho")
    mixed = candidate("Chúng ta bắt đầu", 0.84, "vi", "whisper")
    assert select_candidate(vi, mixed) is vi


def test_missing_mixed_candidate_falls_back_to_vietnamese() -> None:
    vi = candidate("Chúng ta bắt đầu", 0.82, "vi", "pho")
    assert select_candidate(vi, None) is vi


def test_empty_mixed_transcript_falls_back_to_vietnamese() -> None:
    vi = candidate("Chúng ta bắt đầu", 0.82, "vi", "pho")
    mixed = candidate("", 0.99, "mixed", "whisper")
    assert select_candidate(vi, mixed) is vi


def test_empty_vietnamese_transcript_falls_back_to_mixed_candidate() -> None:
    vi = candidate("", 0.0, "unknown", "pho")
    mixed = candidate("We should review this", 0.4, "en", "whisper")
    assert select_candidate(vi, mixed) is mixed


def test_mixed_candidate_below_relative_confidence_threshold_is_rejected() -> None:
    vi = candidate("Chúng ta triển khai", 0.80, "vi", "pho")
    mixed = candidate("Chúng ta deploy", 0.65, "mixed", "whisper")
    assert select_candidate(vi, mixed) is vi


def test_non_code_switched_candidate_needs_clear_confidence_margin() -> None:
    vi = candidate("Chúng ta bắt đầu", 0.80, "vi", "pho")
    mixed = candidate("Chúng ta bắt đầu", 0.879, "vi", "whisper")
    assert select_candidate(vi, mixed) is vi


def test_non_code_switched_candidate_wins_above_confidence_margin() -> None:
    vi = candidate("Chúng ta bắt đầu", 0.80, "vi", "pho")
    mixed = candidate("Chúng ta bắt đầu", 0.881, "vi", "whisper")
    assert select_candidate(vi, mixed) is mixed
