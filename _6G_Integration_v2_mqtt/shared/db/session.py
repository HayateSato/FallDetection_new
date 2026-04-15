"""
SQLAlchemy session factory — shared between inference_server and caregiver_client.

Supports both SQLite (local dev, zero setup) and Postgres (production).
The engine is configured once at module import time based on DATABASE_URL.

Usage in FastAPI endpoints:
    from shared.db.session import get_db
    def my_route(db: Session = Depends(get_db)): ...

Usage in background tasks and non-FastAPI code:
    from shared.db.session import SessionLocal
    db = SessionLocal()
    try:
        ...
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Default to SQLite so the system runs without any Docker dependency locally.
# In production set DATABASE_URL=postgresql+psycopg2://user:pass@postgres:5432/fall_detection
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./caregiver.db")

# SQLite requires check_same_thread=False because paho callbacks, FastAPI handlers,
# and BackgroundTasks all run in different threads.
# pool_pre_ping=True tests the Postgres connection before use — prevents errors
# after a DB restart or idle connection timeout.
_engine_kwargs: dict = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session and closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all tables if they do not exist.
    Safe to call on every startup — does nothing if tables already exist.
    In production use Alembic migrations instead (alembic upgrade head).
    """
    from shared.db.models import Base
    Base.metadata.create_all(engine)
