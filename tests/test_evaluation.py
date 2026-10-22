import pytest

from meeting_stt.evaluation import aggregate_scores, normalize_text, score_text
from scripts.compare_phowhisper import load_manifest


def test_normalization_is_case_and_punctuation_insensitive() -> None:
    assert normalize_text("  CHÚNG ta, review!  ") == "chúng ta review"


def test_normalization_preserves_vietnamese_diacritics() -> None:
    assert normalize_text("Tôi có cà phê") != normalize_text("Toi co ca phe")


def test_exact_transcript_has_zero_wer_and_cer() -> None:
    score = score_text("Chúng ta review kế hoạch.", "chúng ta review kế hoạch")
    assert score.wer == 0
    assert score.cer == 0


def test_known_word_error_rate_counts_substitution_and_insertion() -> None:
    score = score_text("the cat sat", "the bat sat down")
    assert score.word_errors == 2
    assert score.reference_words == 3
    assert score.wer == pytest.approx(2 / 3)


def test_wer_is_not_clamped_when_insertions_exceed_reference_words() -> None:
    score = score_text("a", "a b c")
    assert score.word_errors == 2
    assert score.wer == 2.0


def test_empty_reference_is_rejected_instead_of_reporting_a_misleading_rate() -> None:
    with pytest.raises(ValueError, match="Reference is empty"):
        score_text("!!!", "a transcript")


def test_corpus_wer_is_micro_averaged_not_mean_of_recording_rates() -> None:
    first = score_text("a", "x")
    second = score_text("a b c", "a b c")
    aggregate = aggregate_scores([first, second])
    assert aggregate.wer == pytest.approx(1 / 4)


def test_empty_score_collection_cannot_be_aggregated() -> None:
    with pytest.raises(ValueError, match="empty collection"):
        aggregate_scores([])


def test_manifest_resolves_relative_audio_paths(tmp_path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "id,audio_path,reference\nmeeting-1,meeting.wav,Chúng ta review\n",
        encoding="utf-8",
    )

    entries = load_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].recording_id == "meeting-1"
    assert entries[0].audio_path == audio.resolve()


def test_manifest_rejects_blank_reference(tmp_path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("audio_path,reference\nmeeting.wav,  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="reference is empty"):
        load_manifest(manifest)
