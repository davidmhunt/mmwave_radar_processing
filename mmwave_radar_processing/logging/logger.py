import logging
import sys

# logging levels
# logging.INFO
# logging.DEBUG

_DEFAULT_LOGGER: logging.Logger | None = None


def setup_logger(
    name="mmwave_radar_processing",
    level=logging.DEBUG,
    *,
    set_default: bool = True,
) -> logging.Logger:
    """
    Set up and return a logger that streams to stdout with a consistent formatter.

    Args:
        name: Name of the logger (defaults to ``"mmwave_radar_processing"``).
        level: Logging level (defaults to ``logging.DEBUG``).
        set_default: When True, assign the configured logger to ``_DEFAULT_LOGGER``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    logger.addHandler(handler)
    logger.propagate = False

    global _DEFAULT_LOGGER
    if set_default:
        _DEFAULT_LOGGER = logger

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a logger that has been configured via ``setup_logger``.

    If the requested logger has not yet been configured, ``setup_logger`` is
    invoked transparently so that any helper or submodule can log immediately.
    """
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        setup_logger()

    if name is None or name == "mmSplat":
        return _DEFAULT_LOGGER

    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        setup_logger(name=name, level=_DEFAULT_LOGGER.level, set_default=False)
    return logger
