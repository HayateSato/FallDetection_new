"""
API key utilities — hashing for audit logs.
The raw API key is never stored; only the SHA-256 hex digest is saved to api_request_log.
"""

import hashlib


def hash_api_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of an API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()
