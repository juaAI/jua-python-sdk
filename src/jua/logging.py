import logging


def get_logger(name: str) -> logging.Logger:
    """Configure and return a logger with consistent formatting.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        A configured Logger instance with ISO 8601 timestamp formatting.
    """
    logger = logging.getLogger(name)

    # Set logger string format
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
