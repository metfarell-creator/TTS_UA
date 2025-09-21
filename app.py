import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import glob
import re
from typing import List

import gradio as gr
import torch
from unicodedata import normalize

from ipa_uk import ipa
from huggingface_hub import snapshot_download
from ukrainian_word_stress import Stressifier, StressSymbol
from styletts2_inference.models import StyleTTS2
from verbalizer import Verbalizer

# =====================
#   Global init
# =====================

stressify = Stressifier()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(ROOT, "voices")
MODELS_DIR = os.path.join(ROOT, "models")
HF_REPOS = {
    "verbalizer": (
        "skypro1111/mbart-large-50-verbalization",
        "models--skypro1111--mbart-large-50-verbalization",
    ),
    "multi": (
        "patriotyk/styletts2_ukrainian_multispeaker_hifigan",
        "models--patriotyk--styletts2_ukrainian_multispeaker",
    ),
    "single": (
        "patriotyk/styletts2_ukrainian_single",
        "models--patriotyk--styletts2_ukrainian_single",
    ),
}


def ensure_local_repo(repo_key: str) -> str:
    repo_id, folder_name = HF_REPOS[repo_key]
    target_dir = os.path.join(MODELS_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    has_files = False
    try:
        with os.scandir(target_dir) as entries:
            has_files = any(entries)
    except FileNotFoundError:
        has_files = False

    if has_files:
        return target_dir

    print(f"[download] fetching '{repo_id}' into '{target_dir}'")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # pragma: no cover - network failures surfaced to user
        raise RuntimeError(
            "Не вдалося завантажити модель з Hugging Face. "
            "Переконайтеся у наявності інтернету та доступу до репозиторію "
            f"'{repo_id}'."
        ) from exc

    return target_dir


VERBALIZER_PATH = ensure_local_repo("verbalizer")
MULTI_MODEL_PATH = ensure_local_repo("multi")
SINGLE_MODEL_PATH = ensure_local_repo("single")
SINGLE_STYLE_PATH = os.path.join(ROOT, "filatov.pt")


# =====================
#   Helpers
# =====================

def split_to_parts(text: str, max_len: int = 220) -> List[str]:
    """Break text into manageable parts respecting punctuation and length."""

    if not text:
        return []

    normalized_text = normalize("NFKC", text.strip())
    sentences = re.split(r"(?<=[.?!:])\s+", normalized_text)

    parts: List[str] = []
    current = ""

    def flush(segment: str) -> None:
        segment = segment.strip()
        if segment:
            parts.append(segment)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_len:
            current = candidate
            continue

        if current:
            flush(current)
            current = ""

        words = sentence.split()
        chunk = ""
        for word in words:
            addition = word if not chunk else f" {word}"
            if len(chunk) + len(addition) <= max_len:
                chunk += addition
                continue

            flush(chunk)
            chunk = ""

            if len(word) > max_len:
                for start in range(0, len(word), max_len):
                    flush(word[start : start + max_len])
            else:
                chunk = word

        if chunk:
            current = chunk

    flush(current)
    return parts if parts else [normalized_text]


def preprocess_text(text: str) -> str:
    """Normalize text, unify dashes and replace custom stress markers."""

    if not text:
        return ""

    text = text.strip().replace('"', "")
    text = text.replace("+", StressSymbol.CombiningAcuteAccent)
    text = normalize("NFKC", text)
    text = re.sub(r"[᠆‐‑‒–—―⁻₋−⸺⸻]", "-", text)
    text = re.sub(r"\s-\s", ": ", text)
    return text


# =====================
#   Load models
# =====================

verbalizer = Verbalizer(model_path=VERBALIZER_PATH, device=device)

single_model = StyleTTS2(hf_path=SINGLE_MODEL_PATH, device=device)
if not os.path.isfile(SINGLE_STYLE_PATH):
    raise FileNotFoundError(f"Single speaker style not found: {SINGLE_STYLE_PATH}")
single_style = torch.load(SINGLE_STYLE_PATH, map_location=device)

multi_model = StyleTTS2(hf_path=MULTI_MODEL_PATH, device=device)

multi_styles = {}
prompt_files = sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.pt")))
for path in prompt_files:
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        multi_styles[name] = torch.load(path, map_location=device)
        print(f"[voices] loaded: {name}")
    except Exception as exc:
        print(f"[voices] skip {name}: {exc}")

VOICE_NAMES = sorted(list(multi_styles.keys()))

MODELS = {
    "multi": {"model": multi_model, "styles": multi_styles},
    "single": {"model": single_model, "style": single_style},
}


# =====================
#   Core functions
# =====================

@torch.inference_mode()
def verbalize(text: str) -> str:
    parts = split_to_parts(text)
    outputs = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        outputs.append(verbalizer.generate_text(cleaned))
    return " ".join(outputs).strip()


@torch.inference_mode()
def synthesize(
    model_name: str,
    text: str,
    speed: float,
    voice_name: str | None = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    if not text or not text.strip():
        raise gr.Error("Введіть текст, будь ласка.")
    if len(text) > 50_000:
        raise gr.Error("Текст має бути коротшим за 50 000 символів.")

    model_entry = MODELS[model_name]
    model = model_entry["model"]

    parts = split_to_parts(text)
    if not parts:
        raise gr.Error("Немає тексту для синтезу.")

    wave_chunks: List[torch.Tensor] = []
    iterator = progress.tqdm(parts, desc="Синтез...") if progress else parts

    for chunk in iterator:
        chunk = preprocess_text(chunk)
        if not chunk:
            continue
        stressed = stressify(chunk)
        phonemes = ipa(stressed)
        if not phonemes:
            continue

        tokens = model.tokenizer.encode(phonemes)

        if model_name == "multi":
            style = None
            if voice_name and voice_name in model_entry["styles"]:
                style = model_entry["styles"][voice_name]
            elif VOICE_NAMES:
                style = model_entry["styles"][VOICE_NAMES[0]]
            if style is None:
                raise gr.Error("Не знайдено стилі в папці 'voices/'.")
        else:
            style = model_entry["style"]

        wav = model(tokens, speed=speed, s_prev=style)
        wav = wav.squeeze()
        if wav.dim() == 0:
            wav = wav.unsqueeze(0)
        wave_chunks.append(wav.to("cpu"))

    if not wave_chunks:
        raise gr.Error("Немає результату синтезу — перевірте текст.")

    audio = torch.cat(wave_chunks)
    return 24_000, audio.numpy()


# =====================
#   UI
# =====================

def build_ui():
    with gr.Blocks(title="StyleTTS2 Ukrainian EXP", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # StyleTTS 2 Ukrainian EXP 🇺🇦
        Високоякісний синтез українського мовлення на базі StyleTTS 2.
        """
        )

        with gr.Tabs():
            with gr.TabItem("Multi speaker"):
                with gr.Row():
                    txt_multi = gr.Textbox(
                        label="Text", lines=6, placeholder="Введіть текст українською…"
                    )
                with gr.Row():
                    btn_verbalize_m = gr.Button("Вербалізувати")
                    out_verbal_m = gr.Textbox(label="Вербалізований текст", lines=6)
                with gr.Row():
                    voice_dd = gr.Dropdown(
                        choices=VOICE_NAMES,
                        value=(VOICE_NAMES[0] if VOICE_NAMES else None),
                        label="Голос",
                    )
                    speed_m = gr.Slider(
                        minimum=0.6,
                        maximum=1.4,
                        value=1.0,
                        step=0.05,
                        label="Швидкість",
                    )
                btn_synth_m = gr.Button("Синтезувати")
                audio_m = gr.Audio(label="Audio", type="numpy")

                btn_verbalize_m.click(verbalize, inputs=txt_multi, outputs=out_verbal_m)
                btn_synth_m.click(
                    synthesize,
                    inputs=[gr.State("multi"), txt_multi, speed_m, voice_dd],
                    outputs=audio_m,
                )

            with gr.TabItem("Single speaker"):
                with gr.Row():
                    txt_single = gr.Textbox(
                        label="Text", lines=6, placeholder="Введіть текст українською…"
                    )
                with gr.Row():
                    btn_verbalize_s = gr.Button("Вербалізувати")
                    out_verbal_s = gr.Textbox(label="Вербалізований текст", lines=6)
                with gr.Row():
                    speed_s = gr.Slider(
                        minimum=0.6,
                        maximum=1.4,
                        value=1.0,
                        step=0.05,
                        label="Швидкість",
                    )
                btn_synth_s = gr.Button("Синтезувати")
                audio_s = gr.Audio(label="Audio", type="numpy")

                btn_verbalize_s.click(verbalize, inputs=txt_single, outputs=out_verbal_s)
                btn_synth_s.click(
                    synthesize,
                    inputs=[gr.State("single"), txt_single, speed_s, gr.State(None)],
                    outputs=audio_s,
                )

        return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(api_open=True, max_size=20).launch(show_api=True)
