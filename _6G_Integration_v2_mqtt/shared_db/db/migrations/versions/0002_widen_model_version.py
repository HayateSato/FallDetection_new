"""Widen inference_log.model_version from VARCHAR(20) to VARCHAR(64).

Reason: when /model/switch loads from the MLflow registry, the stored
model_version becomes a fully-qualified label like "mlflow:Production:v8(v0)"
which exceeds the original 20-char limit. Without this, inference_log
inserts silently fail and retraining loses those rows.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-28

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "inference_log",
        "model_version",
        type_=sa.String(length=64),
        existing_type=sa.String(length=20),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "inference_log",
        "model_version",
        type_=sa.String(length=20),
        existing_type=sa.String(length=64),
        existing_nullable=True,
    )
