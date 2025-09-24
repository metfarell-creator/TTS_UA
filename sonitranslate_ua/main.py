"""Точка входу для портативної версії SoniTranslate."""

from config.settings import config
from ui import SoniTranslateUI


def main() -> None:
    """Запускає веб-інтерфейс SoniTranslate."""
    config.setup_directories()
    ui = SoniTranslateUI()
    interface = ui.create_interface()
    interface.queue().launch(
        server_name="0.0.0.0",
        max_file_size=config.MAX_CONTENT_LENGTH * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
