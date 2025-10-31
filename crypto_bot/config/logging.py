"""Centralized logging configuration."""

from __future__ import annotations

import logging
from typing import Iterable, Optional


def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: str = "trading.log",
    format_string: str = "%(asctime)s - %(levelname)s - %(message)s",
    extra_handlers: Optional[Iterable[logging.Handler]] = None,
) -> None:
    """Configure application-wide logging.

    Parameters
    ----------
    level:
        Logging level to apply globally.
    log_file:
        Path to the log file where records should be stored.
    format_string:
        Format string shared by all handlers.
    extra_handlers:
        Optional extra handlers that should be attached alongside the default
        file and stream handlers.
    """

    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]

    if extra_handlers:
        handlers.extend(extra_handlers)

    logging.basicConfig(level=level, format=format_string, handlers=handlers)
