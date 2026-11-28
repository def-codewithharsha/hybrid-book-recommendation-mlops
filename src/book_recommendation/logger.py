"""Application logging configuration with JSON output for deployed services."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any

from book_recommendation.config import get_settings

_SAFE_CONTEXT_FIELDS = (
    "request_id",
    "trace_id",
    "span_id",
    "event",
    "http_method",
    "http_path",
    "status_code",
    "duration_ms",
    "model_version",
)


class _JsonFormatter(logging.Formatter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        payload: dict[str, Any] = {
            "timestamp": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "process_id": record.process,
        }
        for field in _SAFE_CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str | None = None,
    output_format: str | None = None,
    *,
    force: bool = True,
) -> None:
    """Configure root logging from validated settings, with explicit overrides.

    The force option is intended for application entry points that need to
    replace a preconfigured root handler. Libraries should avoid calling this.
    """
    settings = get_settings()
    configured_level = (level or settings.logging.level).upper()
    numeric_level = logging.getLevelNamesMapping().get(configured_level)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {configured_level!r}")

    configured_format = (output_format or settings.logging.format).lower()
    if configured_format not in {"json", "text"}:
        raise ValueError("output_format must be 'json' or 'text'")

    service_name = os.getenv("SERVICE_NAME", settings.project)
    handler = logging.StreamHandler(sys.stdout)
    if configured_format == "json":
        handler.setFormatter(_JsonFormatter(service_name))
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)sZ %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)

    logging.basicConfig(level=numeric_level, handlers=[handler], force=force)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; configure handlers once in the application entry point."""
    return logging.getLogger(name)
