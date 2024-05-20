import logging
import logging.config
import os
import sys
from typing import Any

LOGGERS_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"default": {"format": "[%(asctime)s][%(levelname)s]: %(message)s"}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": logging.DEBUG,
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": "",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "template": {"handlers": ["console", "file"], "level": logging.DEBUG},
    },
}


def get_logging_config_copy() -> dict[str, Any]:
    config = LOGGERS_CONFIG.copy()
    config["loggers"] = LOGGERS_CONFIG["loggers"].copy()
    config["handlers"] = LOGGERS_CONFIG["handlers"].copy()
    config["formatters"] = LOGGERS_CONFIG["formatters"].copy()
    return config


def init_logger(filename: str, logger_name: str = "filter_logger", logging_dir: str = "./logs/") -> logging.Logger:
    os.makedirs(logging_dir, exist_ok=True)

    config = get_logging_config_copy()

    config["handlers"]["file"]["filename"] = os.path.join(logging_dir, filename)
    config["loggers"][logger_name] = {
        "handlers": ["console", "file"],
        "level": logging.DEBUG,
    }

    logging.config.dictConfig(config)
    logger = logging.getLogger(logger_name)

    return logger


def init_stdout_logger(logger_name: str = "filter_logger") -> logging.Logger:
    config = get_logging_config_copy()

    config["loggers"][logger_name] = {
        "handlers": ["console"],
        "level": logging.DEBUG,
    }
    config["handlers"].pop("file")
    config["loggers"].pop("template")

    logging.config.dictConfig(config)
    logger = logging.getLogger(logger_name)

    return logger
