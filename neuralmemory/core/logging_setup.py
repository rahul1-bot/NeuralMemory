from __future__ import annotations
import logging
from pathlib import Path


class LoggerSetup:
    @staticmethod
    def get_logger(name: str, log_file: Path) -> logging.Logger:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger: logging.Logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        file_handler: logging.FileHandler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)

        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger
