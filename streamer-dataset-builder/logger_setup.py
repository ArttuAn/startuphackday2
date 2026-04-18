"""
logger_setup.py — Structured logging for the pipeline.
Creates a rotating file log + coloured console output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from config import LOGS_DIR


def get_logger(name: str = "pipeline") -> logging.Logger:
    log_file = LOGS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(ch)

    # File handler (DEBUG+)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(fh)

    return logger
