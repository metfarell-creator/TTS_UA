"""Модуль синтезу мовлення на базі StyleTTS2."""

from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torchaudio
from styletts2 import StyleTTS2

from ..config.settings import config


class UkrainianTTSEngine:
    """Двигун синтезу української мови з використанням StyleTTS2."""

    def __init__(self) -> None:
        self.model: Optional[StyleTTS2] = None
        self.device = config.STYLETTS2_CONFIG["device"]

    def load_model(self) -> None:
        """Завантажує модель StyleTTS2."""
        if self.model is not None:
            return

        try:
            print("Завантаження української моделі StyleTTS2...")

            if not config.STYLETTS2_CONFIG["checkpoint_path"].exists():
                raise FileNotFoundError(
                    "Модель StyleTTS2 не знайдена: "
                    f"{config.STYLETTS2_CONFIG['checkpoint_path']}"
                )

            self.model = StyleTTS2(
                model_checkpoint_path=str(config.STYLETTS2_CONFIG["checkpoint_path"]),
                config_path=str(config.STYLETTS2_CONFIG["model_path"]),
                device=self.device,
            )

            print("Модель StyleTTS2 успішно завантажена")

        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка завантаження моделі TTS: {exc}") from exc

    def synthesize_text(
        self,
        text: str,
        output_path: Optional[Path] = None,
        voice_reference: Optional[Path] = None,
        diffusion_steps: int = 5,
        embedding_scale: int = 1,
    ) -> Path:
        """Синтезує текст у мову."""
        self.load_model()

        try:
            if len(text) > config.MAX_TEXT_LENGTH:
                text = text[: config.MAX_TEXT_LENGTH] + "..."

            if voice_reference:
                audio_tensor = self.model.inference(
                    text,
                    target_voice_path=voice_reference,
                    diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                )
            else:
                audio_tensor = self.model.inference(
                    text,
                    diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                )

            if output_path is None:
                output_path = config.TEMP_DIR / "tts_output.wav"

            if isinstance(audio_tensor, torch.Tensor):
                audio_tensor = audio_tensor.detach().cpu()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

            torchaudio.save(
                str(output_path),
                audio_tensor,
                config.SAMPLE_RATE,
            )

            return output_path

        except Exception as exc:  # pragma: no cover
            raise Exception(f"Помилка синтезу мови: {exc}") from exc

    def batch_synthesize(
        self,
        texts: Iterable[str],
        output_dir: Path,
        voice_reference: Optional[Path] = None,
        diffusion_steps: int = 5,
        embedding_scale: int = 1,
    ) -> List[Path]:
        """Синтезує кілька текстів за один раз."""
        self.load_model()

        output_paths: List[Path] = []
        for index, text in enumerate(texts):
            output_path = output_dir / f"tts_{index:04d}.wav"
            try:
                self.synthesize_text(
                    text,
                    output_path,
                    voice_reference,
                    diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                )
                output_paths.append(output_path)
            except Exception as exc:  # pragma: no cover
                print(f"Помилка синтезу для тексту {index}: {exc}")
                silent_audio = torch.zeros(1, config.SAMPLE_RATE)
                torchaudio.save(str(output_path), silent_audio, config.SAMPLE_RATE)
                output_paths.append(output_path)

        return output_paths

    def cleanup(self) -> None:
        """Очищує пам'ять."""
        self.model = None
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()


__all__ = ["UkrainianTTSEngine"]
