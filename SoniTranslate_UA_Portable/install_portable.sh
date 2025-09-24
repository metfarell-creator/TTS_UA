#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv ]; then
  echo "[1/4] Створюємо віртуальне середовище .venv ..."
  python3 -m venv .venv
else
  echo "Знайдено існуюче середовище .venv"
fi

if [ ! -f .env ]; then
  echo "Копіюємо .env.example у .env"
  cp .env.example .env
fi

source .venv/bin/activate

echo "[2/4] Оновлюємо pip ..."
pip install --upgrade pip

echo "[3/4] Встановлюємо залежності ..."
pip install -r requirements.txt

echo "[4/4] Попередньо завантажуємо моделі WhisperX та StyleTTS2 ..."
python tools/prefetch_models.py

echo
printf '✅ Готово! Запускайте run_ui.sh\n'
