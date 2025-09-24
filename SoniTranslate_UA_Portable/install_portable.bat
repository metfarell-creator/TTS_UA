@echo off
setlocal EnableDelayedExpansion
title SoniTranslate_UA_Portable - Installer

if not exist .venv (
    echo [1/4] Створюємо віртуальне середовище .venv ...
    python -m venv .venv || goto :error
) else (
    echo Знайдено існуюче середовище .venv
)

if not exist .env (
    echo Копіюємо .env.example у .env
    copy /Y .env.example .env >NUL
)

echo [2/4] Оновлюємо pip ...
call .venv\Scripts\activate.bat && python -m pip install --upgrade pip || goto :error

echo [3/4] Встановлюємо залежності ...
call .venv\Scripts\activate.bat && pip install -r requirements.txt || goto :error

echo [4/4] Попередньо завантажуємо моделі WhisperX та StyleTTS2 ...
call .venv\Scripts\activate.bat && python tools\prefetch_models.py || goto :error

echo.
echo ✅ Готово! Запускайте run_ui.bat
exit /b 0

:error
echo.
echo ❌ Сталася помилка. Перевірте лог вище.
exit /b 1
