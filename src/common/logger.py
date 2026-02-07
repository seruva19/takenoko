import logging
import multiprocessing
import os


def _is_worker_process() -> bool:
    """Return True for subprocesses (e.g., DataLoader workers) spawned by Python multiprocessing."""
    try:
        return multiprocessing.current_process().name != "MainProcess"
    except Exception:
        return False


def _is_main_process() -> bool:
    """Best-effort check for rank-0 while avoiding expensive imports in worker subprocesses."""
    if _is_worker_process():
        return False

    for key in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value) == 0
            except ValueError:
                break

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
