from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote, urlsplit, urlunsplit

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Ataxx Zero API"
    app_env: str = "development"
    app_debug: bool = False
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_log_level: str = "INFO"
    app_log_json: bool = True
    app_log_requests: bool = True
    app_docs_enabled: bool = True
    app_cors_origins: list[str] = []
    app_cors_allow_credentials: bool = True
    app_cors_allow_methods: list[str] = ["*"]
    app_cors_allow_headers: list[str] = ["*"]

    # If set, this value has priority over component-based database settings.
    database_url: str = ""
    supabase_db_password: str = ""
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # Supabase/PostgreSQL components
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ataxx_zero"
    db_user: str = "postgres"
    db_password: str = ""
    db_timezone: str = "UTC"

    # SQLAlchemy/asyncpg runtime tuning
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_pool_timeout_s: int = 30
    db_pool_recycle_s: int = 1800

    # Inference/runtime configuration
    model_checkpoint_path: str = "checkpoints/last.ckpt"
    inference_mode_default: str = "fast"
    inference_device: str = "auto"
    inference_mcts_sims: int = 160
    inference_c_puct: float = 1.5

    # Auth/JWT
    auth_jwt_secret: str = ""
    auth_jwt_algorithm: str = "HS256"
    auth_access_token_ttl_minutes: int = 30
    auth_refresh_token_ttl_days: int = 14
    auth_rate_limit_enabled: bool = True
    auth_login_rate_limit_requests: int = 10
    auth_login_rate_limit_window_s: int = 60
    auth_refresh_rate_limit_requests: int = 20
    auth_refresh_rate_limit_window_s: int = 60

    @model_validator(mode="after")
    def validate_auth_security(self) -> Settings:
        env = self.app_env.strip().lower()
        if env not in {"production", "prod"}:
            return self

        secret = self.auth_jwt_secret.strip()
        weak_values = {
            "",
            "changeme",
            "change-me",
            "dev-change-me",
            "secret",
            "jwt-secret",
        }
        if secret.lower() in weak_values:
            raise ValueError(
                "auth_jwt_secret is required in production and cannot be empty/weak."
            )
        if len(secret) < 32:
            raise ValueError(
                "auth_jwt_secret must be at least 32 characters in production."
            )
        return self

    @computed_field
    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url.strip():
            if self.supabase_db_password.strip() and self._looks_like_supabase_url(
                self.database_url
            ):
                return self._with_overridden_password(
                    self.database_url,
                    self.supabase_db_password,
                )
            return self.database_url
        return (
            f"postgresql+asyncpg://{quote(self.db_user, safe='')}:{quote(self.db_password, safe='')}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @computed_field
    @property
    def docs_url(self) -> str | None:
        return "/docs" if self.app_docs_enabled else None

    @computed_field
    @property
    def redoc_url(self) -> str | None:
        return "/redoc" if self.app_docs_enabled else None

    @staticmethod
    def _with_overridden_password(database_url: str, raw_password: str) -> str:
        parsed = urlsplit(database_url)
        if "@" not in parsed.netloc:
            return database_url
        auth_part, host_part = parsed.netloc.rsplit("@", 1)
        if ":" in auth_part:
            user_part, _ = auth_part.split(":", 1)
        else:
            user_part = auth_part
        safe_password = quote(raw_password, safe="")
        new_netloc = f"{user_part}:{safe_password}@{host_part}"
        return urlunsplit(
            (
                parsed.scheme,
                new_netloc,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )

    @staticmethod
    def _looks_like_supabase_url(database_url: str) -> bool:
        host = (urlsplit(database_url).hostname or "").lower()
        return host.endswith("pooler.supabase.com") or host.endswith("supabase.co")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
