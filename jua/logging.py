import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Set logger string format
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] [%(name)s] "
        "[%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",  # Example ISO 8601 like format
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
