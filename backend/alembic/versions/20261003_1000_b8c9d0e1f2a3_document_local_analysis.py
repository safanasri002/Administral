"""Local document analysis for inconsistency detection

Adds `application_documents.local_analysis` (JSONB: extracted facts + text,
produced on the server by `ai.inconsistency`) and the `local_ocr` extraction
method, used when a scan is read by Tesseract/PaddleOCR instead of the cloud.

Existing rows keep NULL: their analysis is computed lazily from the stored
file the first time a report is requested.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-10-03 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'b8c9d0e1f2a3'
down_revision: str | None = 'a7b8c9d0e1f2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'application_documents',
        sa.Column('local_analysis', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    # ADD VALUE cannot run inside a transaction block on older PostgreSQL.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE extraction_method ADD VALUE IF NOT EXISTS 'local_ocr'")


def downgrade() -> None:
    # PostgreSQL cannot drop an enum value; rows using it are reset first so a
    # re-upgrade stays consistent, and the value itself is left in place.
    op.execute(
        "UPDATE application_documents SET extraction_method = NULL "
        "WHERE extraction_method = 'local_ocr'"
    )
    op.drop_column('application_documents', 'local_analysis')
