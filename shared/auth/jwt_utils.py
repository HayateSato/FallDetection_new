"""
JWT auth utilities for dashboard login.

Usage — create a token (on login):
    from shared.auth.jwt_utils import create_token
    token = create_token({"sub": "operator1", "role": "operator"})

Usage — verify a token (FastAPI dependency):
    from shared.auth.jwt_utils import require_role
    @app.get("/admin")
    async def admin(payload = Depends(require_role("operator"))): ...

Roles:
    "operator"  — system operator, can switch models, view all metrics
    "caregiver" — care-giver, can view patient list and fall events

Token lifetime: 8 hours (configurable via JWT_TOKEN_EXPIRE_HOURS env var)
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
JWT_ALGORITHM = "HS256"
JWT_TOKEN_EXPIRE_HOURS = int(os.environ.get("JWT_TOKEN_EXPIRE_HOURS", "8"))


def create_token(data: dict, expires_in_hours: Optional[int] = None) -> str:
    """Encode a JWT token with an expiry time."""
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not set in environment")
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose not installed. Run: pip install python-jose[cryptography]")

    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours or JWT_TOKEN_EXPIRE_HOURS)
    payload["exp"] = expire
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    Raises ValueError if invalid or expired.
    """
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not set")
    try:
        from jose import jwt, JWTError
    except ImportError:
        raise RuntimeError("python-jose not installed")

    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}")


def hash_password(plain: str) -> str:
    """Hash a password with bcrypt. Store the result, never the plain text."""
    try:
        import bcrypt
        return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except ImportError:
        raise RuntimeError("bcrypt not installed. Run: pip install bcrypt")


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plain-text password against a bcrypt hash."""
    try:
        import bcrypt
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ImportError:
        raise RuntimeError("bcrypt not installed. Run: pip install bcrypt")


def require_role(role: str):
    """
    FastAPI dependency factory. Returns a dependency that:
    1. Reads Bearer token from Authorization header
    2. Verifies JWT
    3. Checks that payload['role'] matches the required role

    Usage:
        @app.get("/admin", dependencies=[Depends(require_role("operator"))])
    """
    from fastapi import HTTPException, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    bearer = HTTPBearer()

    async def _check(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        try:
            payload = verify_token(credentials.credentials)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=401, detail=str(e))
        if payload.get("role") != role:
            raise HTTPException(status_code=403, detail=f"Role '{role}' required")
        return payload

    return _check
