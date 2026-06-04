"""Remove fall_count column from participant_session.

fall_count was incremented by record_fall() which was removed when fall_history
was dropped (migration 0003). The column was always 0 after that point.
Fall counts are now derived from InfluxDB fall_events at query time in
fall_dashboard/db.py::_get_fall_counts().

Uses IF EXISTS guards because migration 0005 removes participant_session from
the inference layer entirely — on fresh deployments the table is never created
(migration 0001 no longer creates it), so a bare ALTER TABLE would fail.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-03

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE IF EXISTS participant_session DROP COLUMN IF EXISTS fall_count")


def downgrade() -> None:
    op.execute("ALTER TABLE IF EXISTS participant_session ADD COLUMN IF NOT EXISTS fall_count INTEGER")
