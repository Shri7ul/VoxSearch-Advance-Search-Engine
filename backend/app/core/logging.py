"""Centralized logging configuration."""
import logging

from app.core.config import Settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_LOG_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def resolve_level(value: str) -> int:
    """Map a configured level name to a numeric level, defaulting to INFO."""
    return _LOG_LEVELS.get(value.strip().upper(), logging.INFO)


def setup_logging(settings: Settings) -> None:
    """Configure root and application logging from the current settings."""
    level = resolve_level(settings.log_level)
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger("app").setLevel(level)
