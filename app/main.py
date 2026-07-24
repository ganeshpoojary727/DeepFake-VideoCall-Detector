from app.config.settings import settings
from app.utils.logger import setup_logger


def main():

    logger = setup_logger()

    logger.info("===================================")
    logger.info("DeepFake Video Call Detector")
    logger.info("Application Started")
    logger.info(f"Device : {settings.DEVICE}")
    logger.info("===================================")


if __name__ == "__main__":
    main()