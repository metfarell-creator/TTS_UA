"""Gradio application for the portable SoniTranslate-UA workflow."""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
import srt
import yaml

from pipeline import align, asr, mixer, tts, utils

BASE_DIR = Path(__file__).resolve().parent.parent
PRESET_PATH = BASE_DIR / "config" / "presets" / "uk_to_uk.yaml"


def load_preset() -> Dict[str, Any]:
    with open(PRESET_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


PRESET = load_preset()


def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    entries: List[srt.Subtitle] = []
    for idx, segment in enumerate(segments, start=1):
        start = timedelta(seconds=float(segment["start"]))
        end = timedelta(seconds=float(segment["end"]))
        text = segment.get("text", "").strip()
        entries.append(srt.Subtitle(index=idx, start=start, end=end, content=text))
    return srt.compose(entries)


def step_transcribe(file, model, language, device, compute_type, diarization):
    if file is None:
        raise gr.Error("Завантажте аудіо або відео файл.")

    hf_token = os.getenv("HF_TOKEN")
    result = asr.transcribe(
        file,
        model_name=model,
        language=language,
        device=device,
        compute_type=compute_type,
        diarization=diarization,
        hf_token=hf_token if diarization else None,
    )
    return _segments_to_srt(result["segments"])


def step_synthesize(srt_text: str, repo: str, sample_rate: int, speed: float) -> str:
    if not srt_text.strip():
        raise gr.Error("Немає SRT контенту для синтезу.")

    entries = utils.srt_to_entries(srt_text)
    segments = []
    for entry in entries:
        waveform = tts.synthesize(
            entry.content,
            repo=repo,
            sample_rate=sample_rate,
            speed=speed,
        )
        segment = mixer.wav_from_array(waveform, sample_rate)
        slot_ms = int((entry.end - entry.start).total_seconds() * 1000)
        segment = align.fit_to_slot(segment, slot_ms)
        segments.append(segment)

    timeline = utils.concat_audio(segments)
    output_path = Path(tempfile.mkstemp(suffix=".wav")[1])
    timeline.export(output_path, format="wav")
    return str(output_path)


def step_mux(wav_path: str, video_file) -> str:
    if not wav_path:
        raise gr.Error("Спочатку згенеруйте дубльовану аудіодоріжку.")
    if video_file is None:
        raise gr.Error("Завантажте відео для збирання.")
    output_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
    mixer.mux_audio_video(wav_path, video_file, str(output_path))
    return str(output_path)


def build_ui() -> gr.Blocks:
    model_choices = [
        "tiny",
        "base",
        "small",
        "medium",
        "large-v2",
        "large-v3",
    ]

    with gr.Blocks(title="SoniTranslate UA Portable") as demo:
        gr.Markdown("# 🇺🇦 SoniTranslate UA Portable")

        with gr.Tab("1) Транскрипція"):
            in_file = gr.File(label="Аудіо / відео файл")
            model = gr.Dropdown(model_choices, value=PRESET.get("asr_model", "large-v3"), label="Whisper модель")
            language = gr.Dropdown(["uk", "en", "ru", "auto"], value=PRESET.get("language", "uk"), label="Мова")
            device = gr.Dropdown(["cuda", "cpu"], value=PRESET.get("device", "cuda"), label="Пристрій")
            compute_type = gr.Dropdown(["float16", "int8"], value=PRESET.get("compute_type", "float16"), label="Precision")
            diar = gr.Checkbox(value=PRESET.get("diarization", False), label="Діаризація")
            srt_out = gr.Code(label="SRT результат", language="srt")
            gr.Button("Транскрибувати").click(
                step_transcribe,
                inputs=[in_file, model, language, device, compute_type, diar],
                outputs=srt_out,
            )

        with gr.Tab("2) Озвучення"):
            repo = gr.Textbox(value=PRESET.get("tts", {}).get("repo", "patriotyk/styletts2-ukrainian"), label="Hugging Face репозиторій")
            sample_rate = gr.Slider(
                minimum=8000,
                maximum=48000,
                value=PRESET.get("tts", {}).get("sample_rate", 24_000),
                step=1000,
                label="Sample rate",
            )
            speed = gr.Slider(
                minimum=0.5,
                maximum=1.5,
                value=PRESET.get("tts", {}).get("speed", 1.0),
                step=0.05,
                label="Швидкість",
            )
            wav_out = gr.Audio(label="Синтезований WAV", type="filepath")
            gr.Button("Синтезувати WAV").click(
                step_synthesize,
                inputs=[srt_out, repo, sample_rate, speed],
                outputs=wav_out,
            )

        with gr.Tab("3) Відео"):
            in_video = gr.File(label="Оригінальне відео")
            final_video = gr.File(label="Дубльоване відео")
            gr.Button("Зібрати відео").click(
                step_mux,
                inputs=[wav_out, in_video],
                outputs=final_video,
            )

    return demo


def main() -> None:  # pragma: no cover - UI entry point
    build_ui().launch()


if __name__ == "__main__":
    main()
