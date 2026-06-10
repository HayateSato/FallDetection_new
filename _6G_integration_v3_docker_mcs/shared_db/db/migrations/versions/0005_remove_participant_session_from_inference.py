"""Remove participant_session from the inference layer Postgres.

participant_session is owned by fall_dashboard (caregiver layer) and is created
there via SQLAlchemy create_all against the caregiver layer's own Postgres
(focus_postgres / caregiver_layer/docker-compose.yml).

It must not exist in the inference layer's Postgres (mcs_fall_postgres). This
migration drops it with IF EXISTS so it is a no-op on fresh deployments where
migration 0001 never created the table.

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-04

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # CASCADE drops the associated participant_session_id_seq automatically.
    op.execute("DROP TABLE IF EXISTS participant_session CASCADE")


def downgrade() -> None:
    op.create_table(
        "participant_session",
        sa.Column("id",               sa.Integer(),               nullable=False),
        sa.Column("participant_name", sa.String(length=100),      nullable=False),
        sa.Column("gender",           sa.String(length=10),       nullable=True),
        sa.Column("start_time",       sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("end_time",         sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_participant_session_participant_name",
        "participant_session",
        ["participant_name"],
    )
