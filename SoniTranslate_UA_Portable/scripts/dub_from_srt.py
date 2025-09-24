#!/usr/bin/env python3
"""Generate a dubbed WAV file from an SRT subtitle track."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydub import AudioSegment
from ukrainian_word_stress import Stressifier

from pipeline import align, mixer, tts, utils


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srt", required=True, help="Шлях до файлу субтитрів у форматі SRT")
    parser.add_argument("--repo", default="patriotyk/styletts2-ukrainian", help="Hugging Face репозиторій із голосом")
    parser.add_argument("--out", default="dubbed.wav", help="Файл результату")
    parser.add_argument("--sr", type=int, default=24_000, help="Частота дискретизації вихідного WAV")
    parser.add_argument("--speed", type=float, default=1.0, help="Швидкість StyleTTS2")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    entries = utils.srt_to_entries(Path(args.srt).read_text(encoding="utf-8"))
    stressifier = Stressifier()

    segments: list[AudioSegment] = []
    for entry in entries:
        enriched_text = stressifier.process_text(entry.content)
        waveform = tts.synthesize(
            enriched_text,
            repo=args.repo,
            sample_rate=args.sr,
            speed=args.speed,
        )
        segment = mixer.wav_from_array(waveform, args.sr)
        slot_ms = int((entry.end - entry.start).total_seconds() * 1000)
        segment = align.fit_to_slot(segment, slot_ms)
        segments.append(segment)

    timeline = utils.concat_audio(segments)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeline.export(output_path, format="wav")
    print(output_path)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
