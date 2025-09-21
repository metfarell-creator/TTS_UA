"""Utility script to bootstrap the StyleTTS 2 Ukrainian project.

The installer performs the following steps:
- installs Python dependencies from requirements.txt;
- ensures local directories configured in config.yml exist;
- downloads the Hugging Face snapshots required for inference.

Run via `python install.py`. Use `--help` for options.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - handled after deps install
    yaml = None  # type: ignore


ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ROOT / "requirements.txt"
CONFIG_FILE = ROOT / "config.yml"


DEFAULT_CONFIG: Dict[str, object] = {
    "sample_rate": 24_000,
    "text": {"max_length": 50_000, "chunk_length": 220},
    "speed": {"min": 0.6, "max": 1.4, "default": 1.0, "step": 0.05},
    "paths": {
        "voices_dir": "voices",
        "models_dir": "models",
        "single_style": "filatov.pt",
    },
    "huggingface": {
        "repos": {
            "verbalizer": {
                "repo_id": "skypro1111/mbart-large-50-verbalization",
                "folder": "models--skypro1111--mbart-large-50-verbalization",
            },
            "multi": {
                "repo_id": "patriotyk/styletts2_ukrainian_multispeaker_hifigan",
                "folder": "models--patriotyk--styletts2_ukrainian_multispeaker",
            },
            "single": {
                "repo_id": "patriotyk/styletts2_ukrainian_single",
                "folder": "models--patriotyk--styletts2_ukrainian_single",
            },
        }
    },
}


class InstallError(RuntimeError):
    """Custom exception used to signal installation failures."""


def log(message: str) -> None:
    print(f"[install] {message}")


def run_command(cmd: Tuple[str, ...], description: str) -> None:
    log(f"{description}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaces to user
        raise InstallError(
            f"Команда '{' '.join(cmd)}' завершилась з помилкою (код {exc.returncode})."
        ) from exc


def ensure_requirements(skip: bool) -> None:
    global yaml
    if skip:
        log("Пропускаю встановлення залежностей (skip-deps).")
    elif not REQUIREMENTS_FILE.exists():
        log("Файл requirements.txt не знайдено, пропускаю встановлення залежностей.")
    else:
        run_command(
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(REQUIREMENTS_FILE),
            ),
            "Встановлення залежностей",
        )

    # yaml може бути не встановлений до інсталяції залежностей
    if yaml is None:
        try:
            import yaml as yaml_mod  # type: ignore
        except ImportError as exc:
            raise InstallError(
                "Модуль PyYAML не встановлено. Переконайтесь, що виконали `pip install -r requirements.txt`."
            ) from exc
        yaml = yaml_mod  # type: ignore


def _merge_dict(base: Dict[str, object], updates: Dict[str, object]) -> Dict[str, object]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def load_config() -> Dict[str, object]:
    config = DEFAULT_CONFIG.copy()

    if not CONFIG_FILE.exists():
        log("config.yml не знайдено — використовуються стандартні значення.")
        return config

    with CONFIG_FILE.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise InstallError("config.yml має містити словник верхнього рівня.")

    return _merge_dict(config, loaded)


def ensure_directories(paths_cfg: Dict[str, object]) -> Tuple[Path, Path]:
    voices_dir = ROOT / str(paths_cfg.get("voices_dir", "voices"))
    models_dir = ROOT / str(paths_cfg.get("models_dir", "models"))

    voices_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    log(f"Гарантовано існує папка голосів: {voices_dir.relative_to(ROOT)}")
    log(f"Гарантовано існує папка моделей: {models_dir.relative_to(ROOT)}")

    return voices_dir, models_dir


def download_models(models_dir: Path, huggingface_cfg: Dict[str, object], force: bool) -> None:
    repos_cfg = huggingface_cfg.get("repos", {}) if isinstance(huggingface_cfg, dict) else {}
    if not isinstance(repos_cfg, dict) or not repos_cfg:
        raise InstallError("У config.yml не описані Hugging Face репозиторії (huggingface.repos).")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - surfaces to user
        raise InstallError(
            "Бібліотеку huggingface_hub не знайдено. Переконайтесь, що залежності встановлено." \
        ) from exc

    for key, data in repos_cfg.items():
        if not isinstance(data, dict):
            log(f"Пропускаю ключ '{key}': очікувався словник з repo_id та folder.")
            continue

        repo_id = data.get("repo_id")
        folder = data.get("folder")
        if not repo_id or not folder:
            log(f"Пропускаю ключ '{key}': repo_id або folder не задані.")
            continue

        target_dir = models_dir / str(folder)
        target_dir.mkdir(parents=True, exist_ok=True)

        if not force:
            # якщо в каталозі вже є файли, пропускаємо скачування
            try:
                has_files = any(target_dir.iterdir())
            except FileNotFoundError:
                has_files = False

            if has_files:
                log(f"Модель '{key}' вже завантажена ({target_dir}).")
                continue

        log(f"Завантаження '{repo_id}' у '{target_dir}'.")
        try:
            snapshot_download(
                repo_id=str(repo_id),
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as exc:  # pragma: no cover - network errors bubble up
            raise InstallError(
                "Не вдалося завантажити модель з Hugging Face: "
                f"{repo_id}. Перевірте підключення до інтернету."
            ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Автовстановлення залежностей і моделей для StyleTTS 2 Ukrainian EXP")
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Пропустити встановлення залежностей з requirements.txt",
    )
    parser.add_argument(
        "--force-models",
        action="store_true",
        help="Перезапустити завантаження моделей навіть якщо вони вже існують",
    )
    parser.add_argument(
        "--only-models",
        action="store_true",
        help="Завантажити лише моделі (залежності не встановлюються)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.only_models:
        skip_deps = True
    else:
        skip_deps = args.skip_deps

    try:
        ensure_requirements(skip_deps)

        config = load_config()
        paths_cfg = config.get("paths", {})
        if not isinstance(paths_cfg, dict):
            raise InstallError("config.yml: секція paths повинна бути словником.")

        _, models_dir = ensure_directories(paths_cfg)

        huggingface_cfg = config.get("huggingface", {})
        if not isinstance(huggingface_cfg, dict):
            raise InstallError("config.yml: секція huggingface повинна бути словником.")

        download_models(models_dir, huggingface_cfg, force=args.force_models)

    except InstallError as exc:
        log(str(exc))
        return 1

    log("Готово. Можна запускати `python app.py` або `run.bat`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
