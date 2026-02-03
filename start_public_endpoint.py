#!/usr/bin/env python3
"""
Public Endpoint Launcher for Fall Detection API

This script:
1. Starts ngrok tunnel to expose the local Flask server
2. Displays the public URL for partners to use
3. Starts the Fall Detection Flask server

Prerequisites:
1. Install ngrok: https://ngrok.com/download
2. Sign up for free ngrok account and get auth token
3. Run: ngrok config add-authtoken <your-token>

Usage:
    python start_public_endpoint.py
"""
import subprocess
import sys
import time
import os
import signal
import json
import secrets
from pathlib import Path

# Configuration
FLASK_PORT = 8000
NGROK_REGION = "eu"  # Change to 'us', 'ap', 'au', 'sa', 'jp', 'in' as needed


def generate_api_key():
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


def check_ngrok_installed():
    """Check if ngrok is installed and accessible."""
    try:
        result = subprocess.run(
            ["ngrok", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"[OK] ngrok found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[ERROR] Error checking ngrok: {e}")

    print("[ERROR] ngrok not found!")
    print("\nTo install ngrok:")
    print("  1. Download from: https://ngrok.com/download")
    print("  2. Extract and add to PATH")
    print("  3. Sign up at ngrok.com and get your auth token")
    print("  4. Run: ngrok config add-authtoken <your-token>")
    return False


def get_ngrok_url(max_retries=10):
    """Get the public URL from ngrok API."""
    import urllib.request
    import urllib.error

    for i in range(max_retries):
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as response:
                data = json.loads(response.read().decode())
                for tunnel in data.get("tunnels", []):
                    if tunnel.get("proto") == "https":
                        return tunnel.get("public_url")
        except (urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(1)

    return None


def update_env_file(api_key: str):
    """Add public endpoint settings to .env if not present."""
    env_path = Path(__file__).parent / ".env"

    if not env_path.exists():
        print("[WARNING] .env file not found")
        return

    content = env_path.read_text()

    # Check if public endpoint settings already exist
    if "PUBLIC_ENDPOINT_ENABLED" in content:
        print("[INFO] Public endpoint settings already in .env")
        return

    # Add new settings
    new_settings = f"""

# =============================================================================
# PUBLIC ENDPOINT CONFIGURATION (Added by start_public_endpoint.py)
# =============================================================================
# Enable public endpoint mode (authentication + rate limiting)
PUBLIC_ENDPOINT_ENABLED=true

# API Keys (comma-separated). Share these with your partners.
# Generate new keys: python -c "import secrets; print(secrets.token_urlsafe(32))"
API_KEYS={api_key}

# Rate limit: max requests per minute per IP
RATE_LIMIT_PER_MINUTE=30

# CORS allowed origins (* for all, or comma-separated list)
CORS_ALLOWED_ORIGINS=*

# Debug mode (automatically disabled when PUBLIC_ENDPOINT_ENABLED=true)
FLASK_DEBUG=false
"""

    with open(env_path, "a") as f:
        f.write(new_settings)

    print(f"[OK] Added public endpoint settings to .env")
    print(f"[OK] Generated API key: {api_key}")


def main():
    print("=" * 60)
    print("Fall Detection API - Public Endpoint Launcher")
    print("=" * 60)

    # Check ngrok
    if not check_ngrok_installed():
        sys.exit(1)

    # Generate API key if needed
    api_key = generate_api_key()
    update_env_file(api_key)

    # Set environment variable for this session
    os.environ["PUBLIC_ENDPOINT_ENABLED"] = "true"
    os.environ["API_KEYS"] = api_key
    os.environ["FLASK_DEBUG"] = "false"

    print("\n[INFO] Starting ngrok tunnel...")

    # Start ngrok in background
    ngrok_process = subprocess.Popen(
        ["ngrok", "http", str(FLASK_PORT), "--region", NGROK_REGION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for ngrok to start and get URL
    time.sleep(2)
    public_url = get_ngrok_url()

    if not public_url:
        print("[ERROR] Failed to get ngrok public URL")
        print("[INFO] Check ngrok dashboard at http://127.0.0.1:4040")
        ngrok_process.terminate()
        sys.exit(1)

    # Display connection info
    print("\n" + "=" * 60)
    print("PUBLIC ENDPOINT READY")
    print("=" * 60)
    print(f"\nPublic URL: {public_url}")
    print(f"API Key:    {api_key}")
    print(f"\nShare this with your partners:")
    print("-" * 60)
    print(f"Endpoint:   POST {public_url}/trigger")
    print(f"Header:     X-API-Key: {api_key}")
    print("-" * 60)
    print(f"\nngrok Dashboard: http://127.0.0.1:4040")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Start Flask app
    try:
        flask_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=Path(__file__).parent
        )

        # Wait for interrupt
        while True:
            time.sleep(1)
            # Check if processes are still running
            if flask_process.poll() is not None:
                print("\n[ERROR] Flask server stopped unexpectedly")
                break
            if ngrok_process.poll() is not None:
                print("\n[ERROR] ngrok tunnel stopped unexpectedly")
                break

    except KeyboardInterrupt:
        print("\n\n[INFO] Shutting down...")

    finally:
        # Cleanup
        ngrok_process.terminate()
        try:
            flask_process.terminate()
        except:
            pass
        print("[OK] Services stopped")


if __name__ == "__main__":
    main()
