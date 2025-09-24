#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv ]; then
  echo "❌ Середовище .venv не знайдено. Спочатку виконайте bash install_portable.sh" >&2
  exit 1
fi

source .venv/bin/activate
python ui/gradio_app.py
