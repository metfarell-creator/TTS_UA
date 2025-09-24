"""Audio/Video post-processing helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from pydub import AudioSegment


def wav_from_array(waveform: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert a ``numpy`` waveform to a mono :class:`AudioSegment`."""

    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    return AudioSegment(
        data=pcm16.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )


def mux_audio_video(audio_path: str, video_path: str, output_path: str) -> Path:
    """Replace the audio track of ``video_path`` with ``audio_path`` using FFmpeg."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output
