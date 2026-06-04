"""Drop fall_history table; add patient_confirmed and needs_help to inference_log.

Architecture change (2026-06): FOCUS now hosts InfluxDB + caregiver dashboard.
Fall timestamps are injected into FOCUS InfluxDB by the mobile app.
Patient confirmation (patient_confirmed, needs_help) is stored directly on
inference_log — updated via POST /inference/{observation_id}/confirm from the
mobile app after the patient responds to the confirmation popup.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-02

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add confirmation columns to inference_log
    op.add_column("inference_log",
                  sa.Column("patient_confirmed", sa.String(length=20), nullable=True))
    op.add_column("inference_log",
                  sa.Column("needs_help", sa.Boolean(), nullable=True))

    # Drop fall_history (indices first, then table)
    op.drop_index("ix_fall_history_detection_time", table_name="fall_history")
    op.drop_index("ix_fall_history_patient_id",     table_name="fall_history")
    op.drop_index("ix_fall_history_observation_id", table_name="fall_history")
    op.drop_table("fall_history")


def downgrade() -> None:
    # Remove columns from inference_log
    op.drop_column("inference_log", "needs_help")
    op.drop_column("inference_log", "patient_confirmed")

    # Recreate fall_history
    op.create_table(
        "fall_history",
        sa.Column("id",                sa.Integer(),               nullable=False),
        sa.Column("observation_id",    sa.String(length=36),       nullable=True),
        sa.Column("patient_id",        sa.String(length=100),      nullable=False),
        sa.Column("fall_detected",     sa.Boolean(),               nullable=False),
        sa.Column("patient_confirmed", sa.String(length=20),       nullable=True),
        sa.Column("needs_help",        sa.Boolean(),               nullable=True),
        sa.Column("detection_time",    sa.DateTime(timezone=True), nullable=True),
        sa.Column("alert_time",        sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["observation_id"], ["inference_log.observation_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fall_history_observation_id", "fall_history", ["observation_id"])
    op.create_index("ix_fall_history_patient_id",     "fall_history", ["patient_id"])
    op.create_index("ix_fall_history_detection_time", "fall_history", ["detection_time"])
