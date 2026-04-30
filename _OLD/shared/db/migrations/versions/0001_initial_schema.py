"""Initial schema — inference_log, feature_snapshot, participant_session, api_request_log

Revision ID: 0001
Revises: (none)
Create Date: 2026-03-20
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inference_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=True),
        sa.Column("fall_detected", sa.Boolean(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("window_size", sa.Integer(), nullable=True),
        sa.Column("inference_mode", sa.String(10), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("participant", sa.String(100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_inference_log_timestamp", "inference_log", ["timestamp"])
    op.create_index("ix_inference_log_participant", "inference_log", ["participant"])
    op.create_index("ix_inference_log_fall_detected", "inference_log", ["fall_detected"])

    op.create_table(
        "feature_snapshot",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("inference_id", sa.Integer(), sa.ForeignKey("inference_log.id"), nullable=False),
        sa.Column("feature_name", sa.String(50), nullable=False),
        sa.Column("feature_value", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feature_snapshot_inference_id", "feature_snapshot", ["inference_id"])

    op.create_table(
        "participant_session",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("participant_name", sa.String(100), nullable=False),
        sa.Column("gender", sa.String(10), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fall_count", sa.Integer(), nullable=True, default=0),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "api_request_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("client_ip", sa.String(45), nullable=True),
        sa.Column("endpoint", sa.String(100), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
        sa.Column("api_key_hash", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_request_log_timestamp", "api_request_log", ["timestamp"])


def downgrade() -> None:
    op.drop_table("api_request_log")
    op.drop_table("participant_session")
    op.drop_table("feature_snapshot")
    op.drop_table("inference_log")
