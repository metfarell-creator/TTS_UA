# SoniTranslate_UA_Portable (WhisperX + StyleTTS2 Ukrainian)

Портативний конвеєр для транскрипції та дубляжу **українською** мовою. У проєкті поєднані:

- **ASR:** [WhisperX](https://github.com/m-bain/whisperX) (CUDA 12.1 / cuDNN 9 через CTranslate2 4.5.0)
- **TTS:** [StyleTTS2](https://github.com/KevinMIN95/StyleTTS2) з українськими чекпойнтами Hugging Face
- **UI:** [Gradio 5.x](https://www.gradio.app)
- **FFmpeg:** портативні білди для Windows/macOS/Linux
- **Preset:** `uk → uk` дубляж із попередньо налаштованими параметрами

## Швидкий старт

### Windows
```bat
install_portable.bat
run_ui.bat
```

### Linux/macOS
```bash
bash install_portable.sh
bash run_ui.sh
```

Після запуску відкрийте браузер за адресою <http://127.0.0.1:7860>.

## Налаштування оточення

1. Скопіюйте файл `.env.example` у `.env` та пропишіть свій `HF_TOKEN`, якщо плануєте використовувати діаризацію.
2. За потреби вкажіть альтернативний шлях для кешу моделей Hugging Face (`HF_HOME`).
3. Скрипт `install_portable` створює віртуальне середовище `.venv`, встановлює залежності з `requirements.txt` та префетчить основні моделі Whisper і StyleTTS2.

## Структура проєкту

```
SoniTranslate_UA_Portable/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── install_portable.bat
├── install_portable.sh
├── run_ui.bat
├── run_ui.sh
├── config/
│   └── presets/
│       └── uk_to_uk.yaml
├── pipeline/
│   ├── __init__.py
│   ├── align.py
│   ├── asr.py
│   ├── mixer.py
│   ├── tts.py
│   └── utils.py
├── scripts/
│   ├── dub_from_srt.py
│   └── export_video.py
├── tools/
│   ├── ffmpeg_setup.bat
│   └── prefetch_models.py
└── ui/
    └── gradio_app.py
```

## Ліцензія

Проєкт розповсюджується за ліцензією MIT. Деталі див. у файлі [LICENSE](LICENSE).
