"""add_game_created_by_user

Revision ID: d4a9f17e2c31
Revises: c3d52d7a1b90
Create Date: 2026-02-23 01:55:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import context, op

# revision identifiers, used by Alembic.
revision = "d4a9f17e2c31"
down_revision = "c3d52d7a1b90"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not context.is_offline_mode():
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        columns = {column["name"] for column in inspector.get_columns("game")}
        if "created_by_user_id" in columns:
            return

    op.add_column(
        "game",
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_game_created_by_user_id_user",
        "game",
        "user",
        ["created_by_user_id"],
        ["id"],
    )
    op.create_index("ix_game_created_by_user_id", "game", ["created_by_user_id"], unique=False)


def downgrade() -> None:
    if not context.is_offline_mode():
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        columns = {column["name"] for column in inspector.get_columns("game")}
        if "created_by_user_id" not in columns:
            return

    op.drop_index("ix_game_created_by_user_id", table_name="game")
    op.drop_constraint("fk_game_created_by_user_id_user", "game", type_="foreignkey")
    op.drop_column("game", "created_by_user_id")
