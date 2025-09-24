#!/usr/bin/env python3
"""Mux a dubbed audio track with the original video using FFmpeg."""

from __future__ import annotations

import argparse

from pipeline.mixer import mux_audio_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Шлях до готової аудіодоріжки")
    parser.add_argument("--video", required=True, help="Вихідне відео")
    parser.add_argument("--out", default="output.mp4", help="Файл результату")
    args = parser.parse_args()

    output = mux_audio_video(args.audio, args.video, args.out)
    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()
