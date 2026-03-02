"""add_queueentry_table

Revision ID: c3d52d7a1b90
Revises: 8f2a3d1c4b9e
Create Date: 2026-02-23 01:40:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import context, op

# revision identifiers, used by Alembic.
revision = "c3d52d7a1b90"
down_revision = "8f2a3d1c4b9e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Offline migrations use a mock connection that cannot be inspected.
    if not context.is_offline_mode():
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        if inspector.has_table("queueentry"):
            return

    op.create_table(
        "queueentry",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("season_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("rating_snapshot", sa.Float(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("WAITING", "MATCHED", "CANCELED", name="queueentrystatus"),
            nullable=False,
        ),
        sa.Column("matched_game_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("matched_at", sa.DateTime(), nullable=True),
        sa.Column("canceled_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["matched_game_id"],
            ["game.id"],
        ),
        sa.ForeignKeyConstraint(
            ["season_id"],
            ["season.id"],
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("season_id", "user_id", name="uq_queueentry_season_user"),
    )
    op.create_index("ix_queueentry_matched_game_id", "queueentry", ["matched_game_id"], unique=False)
    op.create_index("ix_queueentry_season_id", "queueentry", ["season_id"], unique=False)
    op.create_index("ix_queueentry_status", "queueentry", ["status"], unique=False)
    op.create_index("ix_queueentry_user_id", "queueentry", ["user_id"], unique=False)


def downgrade() -> None:
    if not context.is_offline_mode():
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        if not inspector.has_table("queueentry"):
            return

    op.drop_index("ix_queueentry_user_id", table_name="queueentry")
    op.drop_index("ix_queueentry_status", table_name="queueentry")
    op.drop_index("ix_queueentry_season_id", table_name="queueentry")
    op.drop_index("ix_queueentry_matched_game_id", table_name="queueentry")
    op.drop_table("queueentry")
