from __future__ import annotations
import logging, os, sys
from dataclasses import dataclass
from pathlib import Path
from logging.handlers import RotatingFileHandler

@dataclass
class LogConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt: str = os.getenv("LOG_FMT", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    logfile: Path = Path(os.getenv("LOG_FILE", "logs/app.log"))

_cfg = LogConfig()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(_cfg.level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(_cfg.level)
    ch.setFormatter(logging.Formatter(_cfg.fmt, _cfg.datefmt))
    logger.addHandler(ch)

    _cfg.logfile.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(_cfg.logfile, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(_cfg.level)
    fh.setFormatter(logging.Formatter(_cfg.fmt, _cfg.datefmt))
    logger.addHandler(fh)

    logger.propagate = False
    return logger