from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import text

from api.config import Settings, get_settings
from api.db.session import get_engine

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app: str
    env: str


class HealthReadyResponse(BaseModel):
    status: str
    app: str
    env: str
    checks: dict[str, bool]


def _resolve_settings(request: Request) -> Settings:
    state_settings = getattr(request.app.state, "settings", None)
    if isinstance(state_settings, Settings):
        return state_settings
    return get_settings()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns basic application health and environment metadata.",
    responses={
        200: {
            "description": "Service is healthy.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "app": "ataxx-zero-api",
                        "env": "development",
                    }
                }
            },
        }
    },
)
def get_health(request: Request) -> HealthResponse:
    settings = _resolve_settings(request)
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        env=settings.app_env,
    )


async def _check_db_ready() -> bool:
    engine = get_engine()
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    return True


@router.get(
    "/ready",
    response_model=HealthReadyResponse,
    summary="Readiness Check",
    description="Checks whether the API is ready to serve traffic (includes DB connectivity).",
    responses={
        200: {
            "description": "Service is ready.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ready",
                        "app": "ataxx-zero-api",
                        "env": "development",
                        "checks": {"db": True},
                    }
                }
            },
        },
        503: {
            "description": "Service is not ready.",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "service_unavailable",
                        "message": "Database not ready",
                        "detail": "Database not ready",
                        "request_id": "req-123",
                    }
                }
            },
        },
    },
)
async def get_ready(request: Request) -> HealthReadyResponse:
    settings = _resolve_settings(request)
    if not await _check_db_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not ready",
        )
    return HealthReadyResponse(
        status="ready",
        app=settings.app_name,
        env=settings.app_env,
        checks={"db": True},
    )
