import logging


# Credits to https://stackoverflow.com/a/56944256
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    """Configure and return a logger with consistent formatting.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        A configured Logger instance with ISO 8601 timestamp formatting.
    """
    logger = logging.getLogger(name)

    # Only add handler if the logger doesn't already have handlers
    if not logger.handlers:
        # Set logger string format
        formatter = CustomFormatter()

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
