import logging

from sanic.log import logger

from engine.supports import env

log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s"


class CustomFormatter(logging.Formatter):
    orange = "\x1b[38;5;208m"
    green = "\x1b[32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: orange + log_format + reset,
        logging.INFO: green + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset
    }

    def format(self, record):
        name_max_length = 15
        if len(record.name) > name_max_length:
            record.name = record.name[:name_max_length]
        else:
            record.name = record.name.ljust(name_max_length)

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_log():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())

    # Main logger (Sanic and your app)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    # Logging for external libraries
    external_loggers = ['sentence_transformers', 'faster_whisper', 'sanic']

    for logger_name in external_loggers:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(logging.DEBUG)
        lib_logger.addHandler(console_handler)

    if env.DEBUG_MODE in ["on", "yes", "enabled"]:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
