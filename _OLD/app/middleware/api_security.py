"""
API Security: Authentication and Rate Limiting.

Provides the @require_api_key decorator and rate limiting for Flask routes.
"""
import time
import threading
import logging
from functools import wraps
from collections import defaultdict
from flask import request, jsonify

from config.settings import (
    PUBLIC_ENDPOINT_ENABLED,
    API_KEYS,
    RATE_LIMIT_PER_MINUTE,
    CORS_ALLOWED_ORIGINS,
)

logger = logging.getLogger(__name__)

# Rate limiting storage (in-memory, per IP)
_rate_limit_storage = defaultdict(list)
_rate_limit_lock = threading.Lock()


def get_client_ip():
    """Get client IP address, handling proxies."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'


def check_rate_limit(ip: str) -> bool:
    """Check if IP is within rate limit. Returns True if allowed."""
    if not PUBLIC_ENDPOINT_ENABLED:
        return True

    now = time.time()
    window_start = now - 60  # 1 minute window

    with _rate_limit_lock:
        _rate_limit_storage[ip] = [t for t in _rate_limit_storage[ip] if t > window_start]

        if len(_rate_limit_storage[ip]) >= RATE_LIMIT_PER_MINUTE:
            return False

        _rate_limit_storage[ip].append(now)
        return True


def require_api_key(f):
    """Decorator to require API key authentication for endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not PUBLIC_ENDPOINT_ENABLED:
            return f(*args, **kwargs)

        client_ip = get_client_ip()
        if not check_rate_limit(client_ip):
            return jsonify({
                "error": "Rate limit exceeded",
                "message": f"Maximum {RATE_LIMIT_PER_MINUTE} requests per minute allowed"
            }), 429

        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

        if not API_KEYS:
            logger.warning("PUBLIC_ENDPOINT_ENABLED but no API_KEYS configured!")
            return f(*args, **kwargs)

        if not api_key:
            return jsonify({
                "error": "Authentication required",
                "message": "Missing API key. Provide via X-API-Key header or api_key query parameter"
            }), 401

        if api_key not in API_KEYS:
            logger.warning(f"Invalid API key attempt from {client_ip}")
            return jsonify({
                "error": "Invalid API key",
                "message": "The provided API key is not valid"
            }), 403

        return f(*args, **kwargs)
    return decorated_function


def setup_cors(app):
    """Configure CORS headers if in public endpoint mode."""
    if not PUBLIC_ENDPOINT_ENABLED:
        return

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get('Origin', '*')
        if CORS_ALLOWED_ORIGINS == '*':
            response.headers['Access-Control-Allow-Origin'] = origin
        elif origin in CORS_ALLOWED_ORIGINS.split(','):
            response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response
