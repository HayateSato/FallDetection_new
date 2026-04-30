"""
SQLAlchemy session factory.

Usage in FastAPI:
    from shared.db.session import get_db
    def my_route(db: Session = Depends(get_db)): ...

Usage in background tasks (non-FastAPI):
    from shared.db.session import SessionLocal
    db = SessionLocal()
    try:
        ...
    finally:
        db.close()
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# pool_pre_ping=True: test the connection before using it,
# preventing errors when Postgres restarts during long-running processes.
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL is not configured")
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
