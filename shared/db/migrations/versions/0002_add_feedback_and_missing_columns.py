"""Add user_fall, need_help columns + missing inference_log columns

Adds patient feedback columns (user_fall, need_help) and the columns that
were added to the ORM model after migration 0001 but never migrated:
step_seconds, resampling_method, acc_sensor_type.

Revision ID: 0002
Revises:     0001
Create Date: 2026-03-26
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Columns added to the ORM after 0001 but never migrated
    op.add_column("inference_log",
        sa.Column("step_seconds", sa.Float(), nullable=True))
    op.add_column("inference_log",
        sa.Column("resampling_method", sa.String(20), nullable=True))
    op.add_column("inference_log",
        sa.Column("acc_sensor_type", sa.String(20), nullable=True))

    # Patient feedback columns
    # 0 = pending/default  1 = yes  2 = no  3 = no_answer (timeout)
    op.add_column("inference_log",
        sa.Column("user_fall", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("inference_log",
        sa.Column("need_help", sa.Integer(), nullable=True, server_default="0"))

    op.create_index("ix_inference_log_user_fall", "inference_log", ["user_fall"])


def downgrade() -> None:
    op.drop_index("ix_inference_log_user_fall", "inference_log")
    op.drop_column("inference_log", "need_help")
    op.drop_column("inference_log", "user_fall")
    op.drop_column("inference_log", "acc_sensor_type")
    op.drop_column("inference_log", "resampling_method")
    op.drop_column("inference_log", "step_seconds")
