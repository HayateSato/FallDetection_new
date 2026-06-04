"""Initial schema — inference layer tables for fall detection 6G integration.

Tables created:
  inference_log      — one row per /predict call (written by inference_server)
  feature_snapshot   — one row per feature per inference (FK -> inference_log)
  fall_history       — one row per MQTT alert received (written by fall_dashboard)

Note: participant_session belongs to the caregiver layer (fall_dashboard) and is
created there via SQLAlchemy create_all. It is NOT created by this migration.

Revision ID: 0001
Revises: —
Create Date: 2026-04-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── inference_log ──────────────────────────────────────────────────────
    op.create_table(
        "inference_log",
        sa.Column("id",             sa.Integer(),              nullable=False),
        sa.Column("observation_id", sa.String(length=36),      nullable=False),
        sa.Column("patient_id",     sa.String(length=100),     nullable=False),
        sa.Column("device_id",      sa.String(length=100),     nullable=True),
        sa.Column("model_version",  sa.String(length=20),      nullable=True),
        sa.Column("fall_detected",  sa.Boolean(),              nullable=False),
        sa.Column("confidence",     sa.Float(),                nullable=True),
        sa.Column("window_size",    sa.Integer(),              nullable=True),
        sa.Column("latency_ms",     sa.Integer(),              nullable=True),
        sa.Column("detection_time", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("observation_id"),
    )
    op.create_index("ix_inference_log_observation_id", "inference_log", ["observation_id"])
    op.create_index("ix_inference_log_patient_id",     "inference_log", ["patient_id"])

    # ── feature_snapshot ──────────────────────────────────────────────────
    op.create_table(
        "feature_snapshot",
        sa.Column("id",            sa.Integer(),          nullable=False),
        sa.Column("inference_id",  sa.Integer(),          nullable=False),
        sa.Column("feature_name",  sa.String(length=50),  nullable=False),
        sa.Column("feature_value", sa.Float(),            nullable=True),
        sa.ForeignKeyConstraint(["inference_id"], ["inference_log.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feature_snapshot_inference_id", "feature_snapshot", ["inference_id"])

    # ── fall_history ──────────────────────────────────────────────────────
    op.create_table(
        "fall_history",
        sa.Column("id",                sa.Integer(),              nullable=False),
        sa.Column("observation_id",    sa.String(length=36),      nullable=True),
        sa.Column("patient_id",        sa.String(length=100),     nullable=False),
        sa.Column("fall_detected",     sa.Boolean(),              nullable=False),
        sa.Column("patient_confirmed", sa.String(length=20),      nullable=True),
        sa.Column("needs_help",        sa.Boolean(),              nullable=True),
        sa.Column("detection_time",    sa.DateTime(timezone=True), nullable=True),
        sa.Column("alert_time",        sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["observation_id"], ["inference_log.observation_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fall_history_observation_id", "fall_history", ["observation_id"])
    op.create_index("ix_fall_history_patient_id",     "fall_history", ["patient_id"])
    op.create_index("ix_fall_history_detection_time", "fall_history", ["detection_time"])

def downgrade() -> None:
    op.drop_table("fall_history")
    op.drop_table("feature_snapshot")
    op.drop_table("inference_log")
