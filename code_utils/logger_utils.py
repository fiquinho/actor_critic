import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


FORMATTER = logging.Formatter('%(asctime)s: - %(message)s', '%m/%d/%Y %I:%M:%S %p')


def prepare_stream_logger(logger: logging.Logger, level: int=logging.INFO) -> None:
    """
    Configure a logger to print to the console.
    :param logger: The Logger object to configure.
    :param level: The threshold of the logger.
    :return:
    """
    logger.setLevel(level)

    formatter = FORMATTER
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def prepare_file_logger(logger: logging.Logger, level: int, log_file: Path) -> None:
    """
        Configure a logger to print to a logs file.
        :param logger: The Logger object to configure.
        :param level: The threshold for the logger messages.
        :param log_file: The text file where to print the logger messages.
        :return:
        """
    formatter = FORMATTER
    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=1)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)
