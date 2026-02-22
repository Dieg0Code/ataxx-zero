from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from time import perf_counter

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from api.config import Settings

_BASE_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _BASE_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.app_log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    if settings.app_log_json:
        stream_handler.setFormatter(JsonFormatter())
    else:
        stream_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    root_logger.addHandler(stream_handler)


def register_request_logging(app: FastAPI) -> None:
    logger = logging.getLogger("api.request")

    @app.middleware("http")
    async def request_timing_log(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (perf_counter() - start) * 1000.0
            request_id = getattr(request.state, "request_id", "") or request.headers.get(
                "x-request-id", ""
            )
            logger.info(
                "request_completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "request_id": request_id,
                },
            )
