"""Модуль обробки аудіо та відео."""

from pathlib import Path
from typing import Iterable, List, Optional

import ffmpeg
import yt_dlp as youtube_dl
from pydub import AudioSegment

from config.settings import config


class AudioProcessor:
    """Клас для обробки аудіо та відео файлів."""

    def __init__(self) -> None:
        self.temp_dir = config.TEMP_DIR

    def download_youtube_audio(self, url: str, output_path: Optional[Path] = None) -> Path:
        """Завантажує аудіо з YouTube."""
        if output_path is None:
            output_path = self.temp_dir / "youtube_audio.wav"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_path.with_suffix("")),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return output_path
        except Exception as exc:  # pragma: no cover - захоплення помилок мережі
            raise Exception(f"Помилка завантаження з YouTube: {exc}") from exc

    def download_youtube_video(self, url: str, output_path: Optional[Path] = None) -> Path:
        """Завантажує відео з YouTube у форматі mp4."""
        if output_path is None:
            output_path = self.temp_dir / "youtube_video.mp4"

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": str(output_path.with_suffix("")),
            "merge_output_format": "mp4",
            "quiet": True,
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return output_path
        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка завантаження відео з YouTube: {exc}") from exc

    def extract_audio_from_video(
        self, video_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """Виділяє аудіо з відео файлу."""
        if output_path is None:
            output_path = self.temp_dir / "extracted_audio.wav"

        try:
            (
                ffmpeg.input(str(video_path))
                .output(str(output_path), ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка виділення аудіо: {exc}") from exc

    def convert_audio_format(
        self, input_path: Path, output_path: Path, target_sr: int = 16000
    ) -> Path:
        """Конвертує аудіо у потрібний формат."""
        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(target_sr).set_channels(1)
            audio.export(output_path, format="wav")
            return output_path
        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка конвертації аудіо: {exc}") from exc

    def split_audio_by_segments(
        self, audio_path: Path, segments: Iterable[dict], output_dir: Path
    ) -> List[dict]:
        """Розділяє аудіо на сегменти за тайм-кодами."""
        audio = AudioSegment.from_file(audio_path)
        segment_files: List[dict] = []

        for index, segment in enumerate(segments):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)

            # Додаємо невеликі паузи для природності
            padded_start = max(0, start_ms - 100)
            padded_end = min(len(audio), end_ms + 100)

            segment_audio = audio[padded_start:padded_end]
            segment_path = output_dir / f"segment_{index:04d}.wav"
            segment_audio.export(segment_path, format="wav")

            segment_files.append(
                {
                    "path": segment_path,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment.get("text", ""),
                }
            )

        return segment_files

    def combine_audio_segments(
        self, segment_files: Iterable[dict], output_path: Path, original_duration: float
    ) -> Path:
        """Об'єднує аудіо сегменти в одну доріжку."""
        combined = AudioSegment.silent(duration=int(original_duration * 1000))

        for segment in segment_files:
            segment_audio = AudioSegment.from_file(segment["path"])
            start_ms = int(segment["start"] * 1000)

            # Нормалізуємо гучність
            segment_audio = segment_audio.normalize()

            # Перекриваємо оригінальне аудіо
            combined = combined.overlay(segment_audio, position=start_ms)

        combined.export(output_path, format="wav")
        return output_path

    def get_audio_duration(self, audio_path: Path) -> float:
        """Отримує тривалість аудіо файлу."""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка отримання тривалості аудіо: {exc}") from exc


__all__ = ["AudioProcessor"]
