"""initial schema

Revision ID: 20260222_0001
Revises:
Create Date: 2026-02-22 00:00:00.000000
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlmodel import SQLModel

from alembic import op

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.db import models as _models  # noqa: E402,F401

# revision identifiers, used by Alembic.
revision = "20260222_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    SQLModel.metadata.create_all(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    SQLModel.metadata.drop_all(bind=bind, checkfirst=True)
