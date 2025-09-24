@echo off
setlocal
echo Запуск UI...
call ".venv\Scripts\activate.bat" || goto :error
python ui\gradio_app.py
exit /b 0

:error
echo ❌ Не вдалося активувати середовище. Запустіть install_portable.bat
exit /b 1
