import logging

from app.config.settings import settings


def get_logger():

    settings.LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger("DeepFakeDetector")

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        settings.LOG_DIR / "application.log"
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger