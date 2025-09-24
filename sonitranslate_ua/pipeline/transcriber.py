"""Модуль транскрипції за допомогою WhisperX."""

from typing import Any, List, Optional

import torch
import whisperx

from ..config.settings import config


class Transcriber:
    """Клас для транскрипції аудіо за допомогою WhisperX."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.alignment_model: Optional[Any] = None
        self.metadata: Optional[Any] = None
        self.diarize_model: Optional[Any] = None
        self.device = config.WHISPER_DEVICE
        self.compute_type = config.WHISPER_COMPUTE_TYPE
        self.language: Optional[str] = getattr(config, "SOURCE_LANGUAGE", None)
        self.alignment_language: Optional[str] = None

    def load_models(self) -> None:
        """Завантажує моделі WhisperX."""
        if self.model is None:
            print("Завантаження моделі WhisperX...")
            load_kwargs = {"compute_type": self.compute_type}
            if self.language:
                load_kwargs["language"] = self.language

            self.model = whisperx.load_model(
                config.WHISPER_MODEL,
                self.device,
                **load_kwargs,
            )

    def transcribe_audio(
        self, audio_path: str, enable_diarization: bool = False, hf_token: Optional[str] = None
    ) -> List[dict]:
        """Транскрибує аудіо файл."""
        self.load_models()

        try:
            result = self.model.transcribe(
                str(audio_path),
                batch_size=config.BATCH_SIZE,
                language=self.language,
            )

            detected_language = result.get("language") or self.language
            self._ensure_alignment_model(detected_language)

            result = whisperx.align(
                result["segments"],
                self.alignment_model,
                self.metadata,
                str(audio_path),
                self.device,
                return_char_alignments=False,
            )

            if enable_diarization and hf_token:
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=self.device,
                )
                diarize_segments = self.diarize_model(str(audio_path))
                result = whisperx.assign_word_speakers(diarize_segments, result)

            return result["segments"]

        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка транскрипції: {exc}") from exc

    def cleanup(self) -> None:
        """Очищує пам'ять від моделей."""
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        self.model = None
        self.alignment_model = None
        self.diarize_model = None
        self.metadata = None
        self.alignment_language = None

    def _ensure_alignment_model(self, language_code: Optional[str]) -> None:
        """Гарантує завантаження моделі вирівнювання для потрібної мови."""
        target_language = language_code or "en"

        if (
            self.alignment_model is None
            or self.alignment_language != target_language
        ):
            print("Завантаження моделі вирівнювання...")
            self.alignment_model, self.metadata = whisperx.load_align_model(
                language_code=target_language,
                device=self.device,
            )
            self.alignment_language = target_language


__all__ = ["Transcriber"]
