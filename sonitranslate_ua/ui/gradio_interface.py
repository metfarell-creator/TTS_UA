"""Модуль побудови графічного інтерфейсу Gradio для SoniTranslate."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gradio as gr
from transformers import pipeline

from ..config.settings import config
from ..pipeline.audio_processor import AudioProcessor
from ..pipeline.transcriber import Transcriber
from ..pipeline.tts_engine import UkrainianTTSEngine
from ..pipeline.video_mixer import VideoMixer


@dataclass
class PipelineResult:
    """Результат виконання основного конвеєра."""

    status: str
    audio_path: Optional[Path]
    video_path: Optional[Path]
    subtitles: List[Dict[str, Any]]


class SoniTranslateUI:
    """Графічний інтерфейс для SoniTranslate."""

    def __init__(self) -> None:
        self.audio_processor = AudioProcessor()
        self.transcriber = Transcriber()
        self.tts_engine = UkrainianTTSEngine()
        self.video_mixer = VideoMixer()

        self.translation_pipeline: Optional[Any] = None

    # ------------------------------------------------------------------
    # Публічні методи
    # ------------------------------------------------------------------
    def create_interface(self) -> gr.Blocks:
        """Створює та повертає екземпляр інтерфейсу Gradio."""
        with gr.Blocks(theme=config.GRADIO_THEME, title="SoniTranslate Український") as interface:
            gr.Markdown("# 🎯 SoniTranslate - Український дубляж")
            gr.Markdown("Інструмент для автоматичного дубляжу відео українською мовою")

            with gr.Tab("📁 Завантаження та налаштування"):
                with gr.Row():
                    with gr.Column():
                        input_type = gr.Radio(
                            choices=["Файл відео", "YouTube посилання"],
                            label="Тип вхідних даних",
                            value="Файл відео",
                        )
                        video_file = gr.File(
                            label="Відео файл",
                            file_types=[".mp4", ".avi", ".mov", ".mkv"],
                        )
                        youtube_url = gr.Textbox(
                            label="YouTube URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                            visible=False,
                        )

                        voice_option = gr.Radio(
                            choices=["Стандартний український голос", "Клонування голосу"],
                            label="Вибір голосу",
                            value="Стандартний український голос",
                        )
                        voice_reference = gr.File(
                            label="Зразок голосу для клонування (10-20 секунд)",
                            file_types=[".wav", ".mp3"],
                            visible=False,
                        )

                    with gr.Column():
                        with gr.Accordion("Розширені налаштування", open=False):
                            enable_diarization = gr.Checkbox(
                                label="Увімкнути діаризацію (розділення голосів)",
                                value=False,
                            )
                            hf_token = gr.Textbox(
                                label="HuggingFace Token (для діаризації)",
                                type="password",
                                visible=False,
                            )
                            diffusion_steps = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                label="Кількість дифузійних кроків (якість/швидкість)",
                            )
                            embedding_scale = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=1,
                                label="Масштаб ембеддингу (стиль)",
                            )

                input_type.change(
                    self._toggle_input_type,
                    inputs=[input_type],
                    outputs=[video_file, youtube_url],
                )

                voice_option.change(
                    self._toggle_voice_option,
                    inputs=[voice_option],
                    outputs=[voice_reference],
                )

                enable_diarization.change(
                    self._toggle_diarization,
                    inputs=[enable_diarization],
                    outputs=[hf_token],
                )

            with gr.Tab("🚀 Обробка"):
                with gr.Row():
                    process_button = gr.Button("Запустити дубляж", variant="primary")
                    clear_button = gr.Button("Очистити", variant="secondary")

                status_box = gr.Textbox(
                    label="Статус виконання",
                    lines=6,
                    interactive=False,
                )
                with gr.Row():
                    output_audio = gr.Audio(label="Згенероване аудіо", type="filepath")
                    output_video = gr.Video(label="Підсумкове відео")
                segments_table = gr.Dataframe(
                    headers=["Початок", "Кінець", "Оригінальний текст", "Український переклад"],
                    label="Сегменти",
                    interactive=False,
                    value=[],
                )

                process_button.click(
                    self._process_pipeline,
                    inputs=[
                        input_type,
                        video_file,
                        youtube_url,
                        voice_option,
                        voice_reference,
                        enable_diarization,
                        hf_token,
                        diffusion_steps,
                        embedding_scale,
                    ],
                    outputs=[status_box, output_audio, output_video, segments_table],
                )

                clear_button.click(
                    self._clear_outputs,
                    inputs=None,
                    outputs=[status_box, output_audio, output_video, segments_table],
                )

        return interface

    # ------------------------------------------------------------------
    # Обробники подій інтерфейсу
    # ------------------------------------------------------------------
    @staticmethod
    def _toggle_input_type(choice: str) -> Tuple[gr.Update, gr.Update]:
        """Перемикає видимість полів в залежності від типу вхідних даних."""
        if choice == "YouTube посилання":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False)

    @staticmethod
    def _toggle_voice_option(choice: str) -> gr.Update:
        """Перемикає видимість поля завантаження голосу."""
        return gr.update(visible=choice == "Клонування голосу")

    @staticmethod
    def _toggle_diarization(enabled: bool) -> gr.Update:
        """Перемикає видимість поля токена."""
        return gr.update(visible=enabled)

    @staticmethod
    def _clear_outputs(*_: Any) -> Tuple[str, None, None, List[List[str]]]:
        """Очищує вихідні поля."""
        return "", None, None, []

    # ------------------------------------------------------------------
    # Допоміжні методи
    # ------------------------------------------------------------------
    def _process_pipeline(
        self,
        input_type: str,
        video_file: Any,
        youtube_url: str,
        voice_option: str,
        voice_reference: Any,
        enable_diarization: bool,
        hf_token: str,
        diffusion_steps: int,
        embedding_scale: int,
    ) -> Tuple[str, Optional[str], Optional[str], List[List[str]]]:
        """Основний робочий процес інтерфейсу."""
        config.setup_directories()
        run_dir = config.TEMP_DIR / datetime.now().strftime("session_%Y%m%d_%H%M%S")
        run_dir.mkdir(exist_ok=True)

        try:
            pipeline_result, segments = self._execute_pipeline(
                input_type=input_type,
                video_file=video_file,
                youtube_url=youtube_url,
                voice_option=voice_option,
                voice_reference=voice_reference,
                enable_diarization=enable_diarization,
                hf_token=hf_token,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                working_dir=run_dir,
            )
        except Exception as exc:
            return (f"❌ Помилка: {exc}", None, None, [])
        finally:
            self.transcriber.cleanup()
            self.tts_engine.cleanup()

        status = pipeline_result.status
        audio_path = str(pipeline_result.audio_path) if pipeline_result.audio_path else None
        video_path = str(pipeline_result.video_path) if pipeline_result.video_path else None

        table_data = [
            [
                f"{segment['start']:.2f}",
                f"{segment['end']:.2f}",
                segment.get("text", ""),
                segment.get("translation", ""),
            ]
            for segment in segments
        ]

        return status, audio_path, video_path, table_data

    def _execute_pipeline(
        self,
        input_type: str,
        video_file: Any,
        youtube_url: str,
        voice_option: str,
        voice_reference: Any,
        enable_diarization: bool,
        hf_token: str,
        diffusion_steps: int,
        embedding_scale: int,
        working_dir: Path,
    ) -> Tuple[PipelineResult, List[Dict[str, Any]]]:
        """Запускає основні кроки обробки."""
        source_video = self._prepare_input(input_type, video_file, youtube_url, working_dir)
        original_audio = self.audio_processor.extract_audio_from_video(
            source_video, working_dir / "source_audio.wav"
        )
        original_duration = self.audio_processor.get_audio_duration(original_audio)

        segments = self.transcriber.transcribe_audio(
            str(original_audio),
            enable_diarization=enable_diarization,
            hf_token=hf_token if enable_diarization else None,
        )

        translated_segments = self._translate_segments(segments)

        tts_dir = working_dir / "tts_segments"
        tts_dir.mkdir(exist_ok=True)

        reference_path = None
        if voice_option == "Клонування голосу":
            resolved_reference = self._resolve_file(voice_reference)
            if resolved_reference:
                reference_destination = working_dir / f"voice_reference{resolved_reference.suffix or '.wav'}"
                if resolved_reference != reference_destination:
                    shutil.copy(resolved_reference, reference_destination)
                else:
                    reference_destination = resolved_reference
                reference_path = reference_destination
        tts_paths = self.tts_engine.batch_synthesize(
            [segment.get("translation") or segment.get("text", "") for segment in translated_segments],
            tts_dir,
            voice_reference=reference_path,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        )

        tts_segments = [
            {
                "path": path,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment.get("translation") or segment.get("text", ""),
            }
            for path, segment in zip(tts_paths, translated_segments)
        ]

        dubbed_audio_path = working_dir / "dubbed_audio.wav"
        self.audio_processor.combine_audio_segments(
            tts_segments,
            dubbed_audio_path,
            original_duration=original_duration,
        )

        dubbed_video_path = working_dir / "dubbed_video.mp4"
        final_video = self.video_mixer.replace_audio_in_video(
            source_video, dubbed_audio_path, dubbed_video_path
        )

        subtitles = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment.get("translation") or segment.get("text", ""),
            }
            for segment in translated_segments
        ]

        subtitle_video_path = working_dir / "dubbed_video_subtitles.mp4"
        try:
            final_video = self.video_mixer.add_subtitles(
                final_video, subtitles, subtitle_video_path
            )
        except Exception:
            final_video = final_video

        return (
            PipelineResult(
                status="✅ Дубляж успішно завершено",
                audio_path=dubbed_audio_path,
                video_path=final_video,
                subtitles=subtitles,
            ),
            translated_segments,
        )

    def _prepare_input(
        self,
        input_type: str,
        video_file: Any,
        youtube_url: str,
        working_dir: Path,
    ) -> Path:
        """Готує вхідні дані в залежності від обраного режиму."""
        if input_type == "Файл відео":
            resolved = self._resolve_file(video_file)
            if resolved is None:
                raise ValueError("Будь ласка, завантажте відео файл.")
            destination = working_dir / f"input{resolved.suffix}"
            if resolved != destination:
                shutil.copy(resolved, destination)
            return destination

        if not youtube_url:
            raise ValueError("Будь ласка, введіть YouTube URL.")

        video_path = self.audio_processor.download_youtube_video(
            youtube_url, working_dir / "youtube_source.mp4"
        )
        return video_path

    def _resolve_file(self, file_data: Any) -> Optional[Path]:
        """Допоміжний метод для отримання шляху до файлу Gradio."""
        if file_data is None:
            return None

        if isinstance(file_data, Path):
            return file_data

        if isinstance(file_data, str):
            potential = Path(file_data)
            if potential.exists():
                return potential

        if hasattr(file_data, "name") and isinstance(file_data.name, str):
            potential = Path(file_data.name)
            if potential.exists():
                return potential

        if isinstance(file_data, dict):
            for key in ("path", "name", "data"):
                value = file_data.get(key)
                if isinstance(value, str):
                    potential = Path(value)
                    if potential.exists():
                        return potential

        return None

    def _translate_segments(self, segments: Iterable[dict]) -> List[Dict[str, Any]]:
        """Перекладає текст сегментів українською мовою."""
        translator = self._get_translator()
        translated_segments: List[Dict[str, Any]] = []

        for segment in segments:
            original_text = segment.get("text", "").strip()
            if not original_text:
                translated_text = ""
            else:
                try:
                    result = translator(original_text)
                    translated_text = result[0]["translation_text"]
                except Exception:
                    translated_text = original_text

            translated_segment = dict(segment)
            translated_segment["translation"] = translated_text
            translated_segments.append(translated_segment)

        return translated_segments

    def _get_translator(self):
        """Повертає або ініціалізує пайплайн перекладу."""
        if self.translation_pipeline is None:
            try:
                import torch

                has_cuda = torch.cuda.is_available()
            except Exception:  # pragma: no cover - GPU може бути недоступний
                has_cuda = False

            try:
                device = 0 if config.WHISPER_DEVICE == "cuda" and has_cuda else -1
                self.translation_pipeline = pipeline(
                    "translation",
                    model="Helsinki-NLP/opus-mt-mul-uk",
                    device=device,
                )
            except Exception:
                self.translation_pipeline = lambda text: [  # type: ignore[assignment]
                    {"translation_text": text}
                ]
        return self.translation_pipeline


__all__ = ["SoniTranslateUI"]
