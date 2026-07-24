from app.config.settings import settings
from app.utils.logger import get_logger


def main():

    logger = get_logger()

    logger.info("----------------------------------")
    logger.info("DeepFake Video Call Detector")
    logger.info("Application Started")
    logger.info(f"Running on: {settings.DEVICE}")
    logger.info("----------------------------------")


if __name__ == "__main__":
    main()