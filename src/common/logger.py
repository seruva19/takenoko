import logging


def _is_main_process() -> bool:
    """Best-effort check for Accelerate rank-0. Falls back to True if unavailable."""
    try:
        from accelerate import PartialState  # type: ignore

        state = PartialState()
        return bool(getattr(state, "is_main_process", True))
    except Exception:
        return True


class PrefixFormatter(logging.Formatter):
    def format(self, record):
        original = super().format(record)
        return f"☄️ {original}"


def get_logger(name: str = "takenoko", level=logging.DEBUG) -> logging.Logger:
    """
    Get a logger with custom formatting and prevent duplicate handlers.

    This function ensures that each logger only has one handler to prevent
    duplicate log messages. It also sets propagate=False to prevent
    duplicate logging from parent loggers.

    Args:
        name: Logger name (default: "takenoko")
        level: Logging level (default: logging.DEBUG)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Demote noisy logs on non-main processes in distributed runs
    if not _is_main_process() and level < logging.WARNING:
        level = logging.WARNING

    logger.setLevel(level)

    # Only add handler if no handlers exist and propagate is True (to avoid duplicate handlers)
    if not logger.handlers and logger.propagate:
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = PrefixFormatter("%(levelname)s: %(message)s")
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        # Set propagate to False to prevent duplicate logging from parent loggers
        logger.propagate = False

    return logger
