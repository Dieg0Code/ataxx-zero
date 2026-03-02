"""add_botprofile_table

Revision ID: 8f2a3d1c4b9e
Revises: 476fb61b6b10
Create Date: 2026-02-23 01:30:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import context, op

# revision identifiers, used by Alembic.
revision = "8f2a3d1c4b9e"
down_revision = "476fb61b6b10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Offline migrations run against a mock connection, so table inspection is unavailable.
    if not context.is_offline_mode():
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        if inspector.has_table("botprofile"):
            return

    op.create_table(
        "botprofile",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("agent_type", sa.Enum("HUMAN", "HEURISTIC", "MODEL", name="agenttype"), nullable=False),
        sa.Column("heuristic_level", sa.String(length=16), nullable=True),
        sa.Column("model_mode", sa.String(length=16), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], name="fk_botprofile_user_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", name="pk_botprofile"),
        sa.UniqueConstraint("user_id", name="uq_bot_profile_user_id"),
    )
    op.create_index("ix_botprofile_agent_type", "botprofile", ["agent_type"], unique=False)
    op.create_index("ix_botprofile_enabled", "botprofile", ["enabled"], unique=False)


def downgrade() -> None:
    if not context.is_offline_mode():
        bind = op.get_bind()
        inspector = sa.inspect(bind)
        if not inspector.has_table("botprofile"):
            return
    op.drop_index("ix_botprofile_enabled", table_name="botprofile")
    op.drop_index("ix_botprofile_agent_type", table_name="botprofile")
    op.drop_table("botprofile")
