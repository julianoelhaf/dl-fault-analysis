# Vendored from the private `psp_helper` package (see docs/PROVENANCE.md) to
# remove the public repository's dependency on that internal package.
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # guarantees no duplicate logs
    return logger
