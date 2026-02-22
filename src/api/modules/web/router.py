from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

router = APIRouter(tags=["web"])

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"


@router.get("/", include_in_schema=False)
def web_root() -> RedirectResponse:
    return RedirectResponse(url="/web", status_code=307)


@router.get("/web", include_in_schema=False)
def web_index() -> FileResponse:
    return FileResponse(INDEX_FILE)
