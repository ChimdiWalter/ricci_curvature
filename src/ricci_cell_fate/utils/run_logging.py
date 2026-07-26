from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Iterator

from ricci_cell_fate.utils.provenance import utc_now


def configure_logger(name: str, log_path: str | Path) -> logging.Logger:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


@contextlib.contextmanager
def logged_step(name: str, log_path: str | Path) -> Iterator[logging.Logger]:
    logger = configure_logger(name, log_path)
    logger.info("START %s at %s", name, utc_now())
    try:
        yield logger
    except Exception:
        logger.exception("FAILED %s", name)
        raise
    finally:
        logger.info("END %s at %s", name, utc_now())

