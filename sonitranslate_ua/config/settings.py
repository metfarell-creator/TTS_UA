from pathlib import Path


class Config:
    """Конфігураційні налаштування проекту"""

    # Шляхи
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    TEMP_DIR = BASE_DIR / "temp"
    EXAMPLES_DIR = BASE_DIR / "examples"

    # Моделі Whisper
    WHISPER_MODEL = "large-v2"
    WHISPER_DEVICE = "cuda"  # "cpu" або "cuda"
    WHISPER_COMPUTE_TYPE = "float16"  # "int8" для економії пам'яті

    # Модель StyleTTS2 для української мови
    STYLETTS2_CONFIG = {
        "model_path": MODELS_DIR / "styletts2_ua" / "config.yml",
        "checkpoint_path": MODELS_DIR / "styletts2_ua" / "epochs_2nd_00020.pth",
        "device": "cuda" if WHISPER_DEVICE == "cuda" else "cpu",
    }

    # Налаштування аудіо
    SAMPLE_RATE = 24000
    TARGET_LANGUAGE = "uk"

    # Оптимізація
    BATCH_SIZE = 4
    MAX_TEXT_LENGTH = 200

    # Налаштування Gradio
    GRADIO_THEME = "soft"
    MAX_CONTENT_LENGTH = 500  # MB

    @classmethod
    def setup_directories(cls) -> None:
        """Створює необхідні директорії"""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.TEMP_DIR.mkdir(exist_ok=True)
        cls.EXAMPLES_DIR.mkdir(exist_ok=True)

        # Піддиректорії для моделей
        (cls.MODELS_DIR / "styletts2_ua").mkdir(exist_ok=True)
        (cls.MODELS_DIR / "whisper").mkdir(exist_ok=True)


config = Config()
