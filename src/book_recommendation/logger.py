import logging
import sys
import os
import time
import json

from datetime import UTC,datetime
from typing import Any

_DEFAULT_SERVICE_NAME = "book-recommendation"

class _JsonFormatter(logging.Formatter):
    
    def format(self,record :logging.LogRecord) -> str:
        timestamp =datetime.fromtimestamp(record.created,tz = UTC)
        payload = {
            "timestamp" : timestamp.isoformat(timespec="milliseconds").replace("+00:00","Z"),
            "level"     : record.levelname,
            "logger"    : record.name,
            "message"   : record.getMessage(),
            "service"   :os.getenv("SERVICE_NAME",_DEFAULT_SERVICE_NAME),
            "process_id":record.process,
                                    }

        for field in ("request_id","trace_id"):
            value = getattr(record , field , None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload , ensure_ascii= False,default=str)
        
def configure_logging(level: str | None = None) -> None:

    configured_level = os.getenv("LOG_LEVEL", level or "INFO").upper()

    numeric_level = logging.getLevelNamesMapping().get(configured_level)

    if not isinstance(numeric_level, int):
        raise ValueError(f" Invalid log Level {configured_level!r}.")

    output_format = os.getenv("LOG_FORMAT","text").lower()

    if output_format not in {"json","text"}:
        raise ValueError("LOG_FORMAT must be either 'json' or 'text'.")

    handler = logging.StreamHandler(sys.stdout)

    if output_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        formatter = logging.Formatter(
            fmt = "%(asctime)sZ %(levelname)s %(name)s: %(message)s", datefmt= "%Y-%m-%dT%H:%M:%S",
            )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)

    logging.basicConfig(
        level = numeric_level ,
        handlers = [handler] ,
        force = True ,
        
        )

def get_logger(name:str) -> logging.Logger:
    return logging.getLogger(name)
    
