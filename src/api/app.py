from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import Settings, get_settings
from api.error_handling import register_error_handlers
from api.modules.auth.rate_limit import AuthRateLimiter
from api.modules.auth.router import router as auth_router
from api.modules.gameplay.router import router as gameplay_router
from api.modules.health.router import router as health_router
from api.modules.identity.router import router as identity_router
from api.modules.matches.router import router as matches_router
from api.modules.model_versions.router import router as model_versions_router
from api.modules.ranking.router import router as ranking_router
from api.modules.training_samples.router import router as training_samples_router
from api.modules.web.router import STATIC_DIR
from api.modules.web.router import router as web_router
from api.observability import configure_logging, register_request_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    configure_logging(cfg)
    app = FastAPI(
        title=cfg.app_name,
        debug=cfg.app_debug,
        docs_url=cfg.docs_url,
        redoc_url=cfg.redoc_url,
    )
    register_error_handlers(app)
    if cfg.app_log_requests:
        register_request_logging(app)
    if cfg.app_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.app_cors_origins,
            allow_credentials=cfg.app_cors_allow_credentials,
            allow_methods=cfg.app_cors_allow_methods,
            allow_headers=cfg.app_cors_allow_headers,
        )
    app.state.settings = cfg
    app.state.auth_rate_limiter = AuthRateLimiter(
        enabled=cfg.auth_rate_limit_enabled,
        login_max_requests=cfg.auth_login_rate_limit_requests,
        login_window_s=cfg.auth_login_rate_limit_window_s,
        refresh_max_requests=cfg.auth_refresh_rate_limit_requests,
        refresh_window_s=cfg.auth_refresh_rate_limit_window_s,
    )
    app.mount("/web/static", StaticFiles(directory=STATIC_DIR), name="web-static")
    app.include_router(web_router)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(health_router)
    app.include_router(gameplay_router, prefix="/api/v1")
    app.include_router(identity_router, prefix="/api/v1")
    app.include_router(matches_router, prefix="/api/v1")
    app.include_router(model_versions_router, prefix="/api/v1")
    app.include_router(ranking_router, prefix="/api/v1")
    app.include_router(training_samples_router, prefix="/api/v1")
    return app


app = create_app()
