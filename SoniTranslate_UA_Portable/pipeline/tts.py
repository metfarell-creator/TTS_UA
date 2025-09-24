"""Text-to-Speech helpers built on top of StyleTTS2."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Tuple

import numpy as np
from huggingface_hub import snapshot_download

try:
    from styletts2 import tts as _styletts2
except ImportError as exc:  # pragma: no cover - optional dependency at runtime
    _styletts2 = None
    _STYLETTS2_IMPORT_ERROR = exc
else:
    _STYLETTS2_IMPORT_ERROR = None


@functools.lru_cache(maxsize=4)
def _resolve_resources(repo: str) -> Tuple[Path, Path]:
    """Download StyleTTS2 checkpoints and configs from Hugging Face."""

    local_dir = Path(
        snapshot_download(repo_id=repo, local_dir_use_symlinks=False)
    )
    checkpoint = next(local_dir.rglob("*.pth"))
    config = next(local_dir.rglob("*config*.yml"))
    return checkpoint, config


def synthesize(
    text: str,
    *,
    repo: str = "patriotyk/styletts2-ukrainian",
    sample_rate: int = 24_000,
    speed: float = 1.0,
) -> np.ndarray:
    """Synthesize ``text`` into a waveform sampled at ``sample_rate`` Hz."""

    if _styletts2 is None:
        raise RuntimeError("StyleTTS2 is not installed.") from _STYLETTS2_IMPORT_ERROR

    checkpoint_path, config_path = _resolve_resources(repo)
    model = _styletts2.StyleTTS2(
        model_checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
    )
    waveform = model.inference(
        text,
        output_sample_rate=sample_rate,
        speed=speed,
    )
    return waveform.astype(np.float32)
