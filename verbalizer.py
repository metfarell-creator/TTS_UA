import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from transformers import MBart50TokenizerFast
except Exception:  # transformers version without MBart50TokenizerFast
    MBart50TokenizerFast = None


class Verbalizer:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model_path = model_path

        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                use_fast=True,
            )
        except Exception as auto_exc:
            print(f"[verbalizer] AutoTokenizer failed: {auto_exc}")
            if MBart50TokenizerFast is None:
                raise RuntimeError(
                    "MBart50TokenizerFast недоступний у вашій версії transformers."
                )
            try:
                tokenizer = MBart50TokenizerFast.from_pretrained(
                    model_path,
                    local_files_only=True,
                )
            except Exception as mbart_exc:
                raise RuntimeError(
                    f"Не вдалося завантажити токенайзер: {mbart_exc}"
                ) from mbart_exc

        self.tokenizer = tokenizer

        if getattr(self.tokenizer, "pad_token", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            self.tokenizer.pad_token = eos_token if eos_token else "</s>"

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                local_files_only=True,
            )
        except Exception as model_exc:
            raise RuntimeError(
                f"Не вдалося завантажити модель вербалізатора: {model_exc}"
            ) from model_exc

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate_text(self, text: str, **generate_kwargs) -> str:
        if not text:
            return ""

        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generated = self.model.generate(**inputs, **generate_kwargs)
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded[0] if decoded else ""
