"""Structured logging.

Every stage of the pipeline logs duration and a few cardinalities. The point is
that a research run can be explained after the fact: which provider answered,
how many candidates were considered, how long the first token took, which
optional steps degraded.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.basicConfig(level=logging.INFO, handlers=[handler])
        _configured = True
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info("%s %s", event, json.dumps(fields, sort_keys=True, default=str))


@contextmanager
def timed(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log an event with its wall-clock duration and any extra fields."""

    started = time.perf_counter()
    payload: dict[str, Any] = dict(fields)
    try:
        yield payload
    finally:
        payload["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        log_event(logger, event, **payload)
