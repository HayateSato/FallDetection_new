"""Remove fall_count column from participant_session.

fall_count was incremented by record_fall() which was removed when fall_history
was dropped (migration 0003). The column was always 0 after that point.
Fall counts are now derived from InfluxDB fall_events at query time in
fall_dashboard/db.py::_get_fall_counts().

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
    op.drop_column("participant_session", "fall_count")


def downgrade() -> None:
    op.add_column(
        "participant_session",
        sa.Column("fall_count", sa.Integer(), nullable=True),
    )
