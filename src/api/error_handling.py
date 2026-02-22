from __future__ import annotations

from http import HTTPStatus
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, OperationalError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response


def _resolve_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str) and request_id:
        return request_id
    return str(uuid4())


def _status_to_error_code(status_code: int) -> str:
    if status_code == 400:
        return "bad_request"
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 422:
        return "validation_error"
    if status_code == 429:
        return "too_many_requests"
    if status_code == 503:
        return "service_unavailable"
    if 500 <= status_code <= 599:
        return "internal_error"
    return f"http_{status_code}"


def _build_error_body(
    *,
    status_code: int,
    request_id: str,
    detail: object,
    details: object | None = None,
) -> dict[str, object]:
    message = detail if isinstance(detail, str) else HTTPStatus(status_code).phrase
    body: dict[str, object] = {
        "error_code": _status_to_error_code(status_code),
        "message": message,
        # Backward compatibility for existing clients/tests.
        "detail": detail,
        "request_id": request_id,
    }
    if details is not None:
        body["details"] = details
    return body


def register_error_handlers(app: FastAPI) -> None:
    @app.middleware("http")
    async def attach_request_id(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        incoming_request_id = request.headers.get("x-request-id")
        request.state.request_id = incoming_request_id or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = _resolve_request_id(request)
        detail = exc.detail if exc.detail is not None else HTTPStatus(exc.status_code).phrase
        details = detail if isinstance(detail, (dict, list)) else None
        body = _build_error_body(
            status_code=exc.status_code,
            request_id=request_id,
            detail=detail,
            details=details,
        )
        return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = _resolve_request_id(request)
        detail = "Validation failed"
        body = _build_error_body(
            status_code=422,
            request_id=request_id,
            detail=detail,
            details=exc.errors(),
        )
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(OperationalError)
    @app.exception_handler(DBAPIError)
    @app.exception_handler(OSError)
    async def database_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
        del exc
        request_id = _resolve_request_id(request)
        body = _build_error_body(
            status_code=503,
            request_id=request_id,
            detail="Database unavailable",
        )
        return JSONResponse(status_code=503, content=body)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        del exc
        request_id = _resolve_request_id(request)
        body = _build_error_body(
            status_code=500,
            request_id=request_id,
            detail="Internal server error",
        )
        return JSONResponse(status_code=500, content=body)
