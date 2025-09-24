"""Helpers for fitting synthesized audio into subtitle time-slots."""

from pydub import AudioSegment

HEAD_PAD_MS = 70
TAIL_PAD_MS = 120
FADE_MS = 40
MAX_TRIM_OVERFLOW_MS = 220


def fit_to_slot(segment: AudioSegment, slot_ms: int) -> AudioSegment:
    """Pad and trim ``segment`` so that it fits into a subtitle slot."""

    padded = AudioSegment.silent(duration=HEAD_PAD_MS) + segment + AudioSegment.silent(
        duration=TAIL_PAD_MS
    )

    if len(padded) > slot_ms + MAX_TRIM_OVERFLOW_MS:
        return padded.fade_out(FADE_MS)[:slot_ms]
    if len(padded) < slot_ms:
        return padded + AudioSegment.silent(duration=slot_ms - len(padded))
    return padded
