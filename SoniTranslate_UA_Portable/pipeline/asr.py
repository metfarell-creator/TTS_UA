"""Automatic speech recognition helpers built on top of WhisperX."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import whisperx


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def transcribe(
    audio_path: str,
    *,
    model_name: str = "large-v3",
    language: str = "uk",
    device: str = "cuda",
    compute_type: str = "float16",
    diarization: bool = False,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe ``audio_path`` using WhisperX and optional diarization."""

    runtime_device = _resolve_device(device)
    model = whisperx.load_model(
        model_name,
        runtime_device,
        compute_type=compute_type,
        asr_options={"language": language},
    )
    audio = whisperx.load_audio(audio_path)
    result: Dict[str, Any] = model.transcribe(audio, language=language)

    align_model, metadata = whisperx.load_align_model(language_code=language, device=runtime_device)
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, runtime_device, return_char_alignments=False
    )
    result["segments"] = aligned["segments"]
    if "word_segments" in aligned:
        result["word_segments"] = aligned["word_segments"]

    if diarization:
        if not hf_token:
            raise ValueError("Hugging Face token is required for diarization.")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=runtime_device)
        diarization_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarization_segments, result)

    return result
