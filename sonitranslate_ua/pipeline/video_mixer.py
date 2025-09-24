"""Модуль для змішування відео та аудіо."""

from pathlib import Path
from typing import Iterable

import ffmpeg

from config.settings import config


class VideoMixer:
    """Клас для змішування аудіо та відео."""

    def __init__(self) -> None:
        self.temp_dir = config.TEMP_DIR

    def replace_audio_in_video(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        """Замінює аудіо доріжку у відео файлі."""
        try:
            temp_audio = self.temp_dir / "temp_audio.aac"

            (
                ffmpeg.input(str(audio_path))
                .output(str(temp_audio), ac=2, ar=44100, audio_codec="aac")
                .overwrite_output()
                .run(quiet=True)
            )

            output_kwargs = {"map": ["0:v:0", "1:a:0"], "shortest": None}
            (
                ffmpeg.input(str(video_path))
                .input(str(temp_audio))
                .output(
                    str(output_path),
                    vcodec="copy",
                    acodec="aac",
                    **output_kwargs,
                )
                .overwrite_output()
                .run(quiet=True)
            )

            temp_audio.unlink(missing_ok=True)
            return output_path

        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка змішування відео: {exc}") from exc

    def add_subtitles(self, video_path: Path, subtitles: Iterable[dict], output_path: Path) -> Path:
        """Додає субтитри до відео."""
        try:
            srt_path = self.temp_dir / "subtitles.srt"
            self._create_srt_file(srt_path, subtitles)

            (
                ffmpeg.input(str(video_path))
                .output(str(output_path), vf=f"subtitles={srt_path}")
                .overwrite_output()
                .run(quiet=True)
            )

            srt_path.unlink(missing_ok=True)
            return output_path

        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка додавання субтитрів: {exc}") from exc

    def _create_srt_file(self, srt_path: Path, subtitles: Iterable[dict]) -> None:
        """Створює файл субтитрів у форматі SRT."""
        with srt_path.open("w", encoding="utf-8") as handle:
            for index, subtitle in enumerate(subtitles):
                start_time = self._format_timestamp(subtitle["start"])
                end_time = self._format_timestamp(subtitle["end"])

                handle.write(f"{index + 1}\n")
                handle.write(f"{start_time} --> {end_time}\n")
                handle.write(f"{subtitle['text']}\n\n")

    def _format_timestamp(self, seconds: float) -> str:
        """Форматує час для SRT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"


__all__ = ["VideoMixer"]
