from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", *, log_dir: Optional[Path] = None, console: bool = True, file: bool = True) -> Logger:
    """Configure logging for an experiment run."""
    logger = logging.getLogger("experiments")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # prevent duplicate handlers when reconfiguring (e.g., resume)
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "logs.txt", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging configured: level=%s console=%s file=%s", level, console, file)
    return logger


__all__ = ["setup_logging"]
