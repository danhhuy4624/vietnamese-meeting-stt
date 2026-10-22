#!/usr/bin/env python
"""Evaluate one or more Whisper checkpoints against a labeled CSV manifest.

Manifest columns: audio_path,reference; optional id.
Relative audio paths are resolved from the manifest's directory. Example:

    python scripts/compare_phowhisper.py --manifest data/eval.csv \
        --models vinai/PhoWhisper-small ./models/PhoWhisper-small-meetings \
        --device cuda --output evaluation-results.csv

WER/CER are emitted as ratios. The report includes per-recording predictions and
corpus-level micro-averaged rates. References must be human-verified and aligned
with their audio; a model-only run cannot establish real-world WER.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from meeting_stt.constants import WHISPER_SAMPLE_RATE
from meeting_stt.evaluation import (
    ErrorRateScore,
    aggregate_scores,
    normalize_text,
    score_text,
)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    recording_id: str
    audio_path: Path
    reference: str


RESULT_COLUMNS = [
    "recording_id",
    "model",
    "audio_path",
    "reference",
    "prediction",
    "wer",
    "cer",
    "word_errors",
    "reference_words",
    "character_errors",
    "reference_characters",
    "inference_seconds",
]


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Read and validate a CSV manifest before downloading/loading any models."""
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Manifest must have a CSV header.")

        columns = {
            name.strip().casefold(): name
            for name in reader.fieldnames
            if name and name.strip()
        }
        audio_column = columns.get("audio_path") or columns.get("audio")
        reference_column = columns.get("reference")
        id_column = columns.get("id")
        if audio_column is None or reference_column is None:
            raise ValueError(
                "Manifest needs audio_path (or audio) and reference columns; id is optional."
            )

        for line_number, row in enumerate(reader, start=2):
            audio_value = (row.get(audio_column) or "").strip()
            reference = (row.get(reference_column) or "").strip()
            if not audio_value:
                raise ValueError(f"{manifest_path}:{line_number}: audio path is empty.")
            if not reference or not normalize_text(reference):
                raise ValueError(
                    f"{manifest_path}:{line_number}: reference is empty after normalization."
                )

            audio_path = Path(audio_value).expanduser()
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path}:{line_number}: audio file not found: {audio_path}"
                )

            recording_id = (row.get(id_column) or "").strip() if id_column else ""
            if not recording_id:
                recording_id = audio_path.stem
            if recording_id in seen_ids:
                raise ValueError(
                    f"{manifest_path}:{line_number}: duplicate id {recording_id!r}; "
                    "provide unique values in the optional id column."
                )
            seen_ids.add(recording_id)
            entries.append(ManifestEntry(recording_id, audio_path, reference))

    if not entries:
        raise ValueError("Manifest contains no evaluation recordings.")
    return entries


def load_audio(path: Path) -> np.ndarray:
    """Load, downmix to mono, and resample an audio file to the 16 kHz ASR contract."""
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[0] == 0:
        raise ValueError(f"Audio file is empty: {path}")

    mono = audio.mean(axis=1, dtype=np.float32)
    if source_rate != WHISPER_SAMPLE_RATE:
        divisor = math.gcd(int(source_rate), WHISPER_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            up=WHISPER_SAMPLE_RATE // divisor,
            down=int(source_rate) // divisor,
        ).astype(np.float32)
    return mono


def load_asr_pipeline(
    model_id: str,
    device: str,
    dtype_name: str,
    chunk_length_s: float,
    stride_length_s: float,
) -> Any:
    """Load a chunked Transformers ASR pipeline, without importing models in unit tests."""
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline

    if dtype_name == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, dtype_name)

    processor = AutoProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device == "cuda" else -1,
        chunk_length_s=chunk_length_s,
        stride_length_s=(stride_length_s, stride_length_s),
    )


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Labeled CSV manifest.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Hugging Face model IDs or local checkpoint directories.",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--chunk-length-s", type=float, default=30.0)
    parser.add_argument("--stride-length-s", type=float, default=5.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args(argv)

    try:
        entries = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if len(set(args.models)) != len(args.models):
        parser.error("--models contains duplicate model IDs.")
    if args.chunk_length_s <= 0:
        parser.error("--chunk-length-s must be positive.")
    if args.stride_length_s < 0 or 2 * args.stride_length_s >= args.chunk_length_s:
        parser.error("--stride-length-s must be non-negative and less than half the chunk length.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is not available.")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    all_rows: list[dict[str, Any]] = []
    scores_by_model: dict[str, list[ErrorRateScore]] = {}
    for model_id in args.models:
        print(f"\nLoading {model_id} on {device}...")
        recognizer = load_asr_pipeline(
            model_id,
            device,
            args.dtype,
            args.chunk_length_s,
            args.stride_length_s,
        )
        model_scores: list[ErrorRateScore] = []
        try:
            for entry in entries:
                audio = load_audio(entry.audio_path)
                started = time.perf_counter()
                output = recognizer(
                    {"raw": audio, "sampling_rate": WHISPER_SAMPLE_RATE},
                    generate_kwargs={
                        "task": "transcribe",
                        "max_new_tokens": args.max_new_tokens,
                        "num_beams": 1,
                    },
                )
                elapsed = time.perf_counter() - started
                prediction = str(output.get("text", "")).strip()
                score = score_text(entry.reference, prediction)
                model_scores.append(score)
                all_rows.append(
                    {
                        "recording_id": entry.recording_id,
                        "model": model_id,
                        "audio_path": str(entry.audio_path),
                        "reference": entry.reference,
                        "prediction": prediction,
                        "wer": score.wer,
                        "cer": score.cer,
                        "word_errors": score.word_errors,
                        "reference_words": score.reference_words,
                        "character_errors": score.character_errors,
                        "reference_characters": score.reference_characters,
                        "inference_seconds": elapsed,
                    }
                )
                print(
                    f"  {entry.recording_id}: WER={score.wer:.2%}, "
                    f"CER={score.cer:.2%}, inference={elapsed:.2f}s"
                )
        finally:
            del recognizer
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        scores_by_model[model_id] = model_scores
        aggregate = aggregate_scores(model_scores)
        print(
            f"{model_id}: corpus WER={aggregate.wer:.2%} "
            f"({aggregate.word_errors}/{aggregate.reference_words} word errors), "
            f"CER={aggregate.cer:.2%} "
            f"({aggregate.character_errors}/{aggregate.reference_characters} character errors)"
        )

    output_path = args.output.expanduser().resolve()
    if output_path == args.manifest.expanduser().resolve():
        parser.error("--output must not overwrite the input manifest.")
    write_results(output_path, all_rows)
    print(f"\nPer-recording results written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
