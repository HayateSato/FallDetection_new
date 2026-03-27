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
    # These three columns may already exist if the table was created directly from
    # the ORM model before this migration was written — use IF NOT EXISTS to be safe.
    op.execute("ALTER TABLE inference_log ADD COLUMN IF NOT EXISTS step_seconds FLOAT")
    op.execute("ALTER TABLE inference_log ADD COLUMN IF NOT EXISTS resampling_method VARCHAR(20)")
    op.execute("ALTER TABLE inference_log ADD COLUMN IF NOT EXISTS acc_sensor_type VARCHAR(20)")

    # Patient feedback columns
    # 0 = pending/default  1 = yes  2 = no  3 = no_answer (timeout)
    op.execute("ALTER TABLE inference_log ADD COLUMN IF NOT EXISTS user_fall INTEGER DEFAULT 0")
    op.execute("ALTER TABLE inference_log ADD COLUMN IF NOT EXISTS need_help INTEGER DEFAULT 0")

    # Index for filtering by feedback status in the caregiver dashboard
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_inference_log_user_fall
        ON inference_log (user_fall)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_inference_log_user_fall")
    op.drop_column("inference_log", "need_help")
    op.drop_column("inference_log", "user_fall")
    # Note: step_seconds/resampling_method/acc_sensor_type are NOT dropped on downgrade
    # because they may have pre-dated this migration and dropping them would be destructive.
