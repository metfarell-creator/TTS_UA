"""Скрипт встановлення залежностей для SoniTranslate."""

import subprocess
import sys
from pathlib import Path


def install_requirements() -> None:
    """Встановлює залежності з файлу requirements.txt."""
    requirements_path = Path(__file__).with_name("requirements.txt")
    if not requirements_path.exists():
        raise FileNotFoundError("Файл requirements.txt не знайдено")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])


if __name__ == "__main__":
    install_requirements()
