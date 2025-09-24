"""Utility helpers for subtitle handling and audio post-processing."""

from __future__ import annotations

import datetime as dt
from typing import Iterable, List

import srt
from pydub import AudioSegment


def to_ms(delta: dt.timedelta) -> int:
    """Convert :class:`datetime.timedelta` to milliseconds."""

    return int(delta.total_seconds() * 1000)


def srt_to_entries(srt_text: str) -> List[srt.Subtitle]:
    """Parse SRT text into a list of :class:`srt.Subtitle` entries."""

    return list(srt.parse(srt_text))


def entries_to_srt(entries: Iterable[srt.Subtitle]) -> str:
    """Compose SRT string from subtitle entries."""

    return srt.compose(entries)


def concat_audio(segments: Iterable[AudioSegment]) -> AudioSegment:
    """Concatenate multiple :class:`AudioSegment` objects into one timeline."""

    timeline = AudioSegment.silent(duration=0)
    for segment in segments:
        timeline += segment
    return timeline
