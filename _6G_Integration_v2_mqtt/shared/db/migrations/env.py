"""
Alembic env.py — reads DATABASE_URL from the environment so the same
migration script works against both SQLite (local dev) and Postgres
(production).

Running migrations:
    # From _6G_Integration_v2_mqtt/ directory:
    alembic upgrade head

    # In Docker (production):
    docker exec <container_name> alembic upgrade head

    # Or inside a Python entrypoint (auto-migrate on startup):
    from alembic.config import Config
    from alembic import command
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
"""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── Import the ORM Base so Alembic can autogenerate migrations ────────────────
# sys.path manipulation is not needed when alembic is run from the
# _6G_Integration_v2_mqtt/ directory (which is the project root).
from shared.db.models import Base

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# Override sqlalchemy.url from environment so credentials never live in INI
database_url = os.getenv("DATABASE_URL", "sqlite:///./caregiver.db")
config.set_main_option("sqlalchemy.url", database_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── Offline mode (generate SQL without a live connection) ─────────────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (apply directly against a live DB) ───────────────────────────
def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
