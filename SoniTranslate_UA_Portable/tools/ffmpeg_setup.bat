@echo off
setlocal

set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.0.1-essentials_build.zip"
set "OUT_ZIP=ffmpeg-portable.zip"
set "TARGET_DIR=bin"

echo Завантаження портативного FFmpeg...
powershell -NoProfile -Command "Invoke-WebRequest -Uri %FFMPEG_URL% -OutFile %OUT_ZIP%" || goto :error

echo Розпаковка у папку %TARGET_DIR% ...
powershell -NoProfile -Command "Expand-Archive -Force %OUT_ZIP% %TARGET_DIR%" || goto :error

echo Очищення...
del %OUT_ZIP%

echo ✅ FFmpeg встановлено. Додайте %TARGET_DIR% до PATH або скопіюйте виконувані файли в корінь проєкту.
exit /b 0

:error
echo ❌ Не вдалося завантажити або розпакувати FFmpeg. Перевірте підключення до інтернету.
exit /b 1
