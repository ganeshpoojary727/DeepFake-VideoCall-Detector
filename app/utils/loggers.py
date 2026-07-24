import logging

from app.config.settings import settings


def setup_logger():

    settings.LOG_DIR.mkdir(
        exist_ok=True
    )

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

        handlers=[
            logging.FileHandler(
                settings.LOG_DIR / "application.log"
            ),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("DeepFakeDetector")