"""Frozen inconsistency verdict on submitted cases

Adds `cases.inconsistency_verdict` (JSONB): the coherence status of the
documents as submitted — status and counts only, computed by the deterministic
rules of `ai.inconsistency` at submission. Cases submitted before keep NULL
("not computed"), which is the honest state: no verdict was frozen for them.

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-10-05 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c9d0e1f2a3b4'
down_revision: str | None = 'b8c9d0e1f2a3'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'cases',
        sa.Column('inconsistency_verdict', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('cases', 'inconsistency_verdict')
