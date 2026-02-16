#!/usr/bin/env python3
"""
Unified Public Endpoint Launcher for Fall Detection API

Reads TUNNEL_MODE from .env to decide how to expose the service:
  - local:      Just starts Flask on localhost (no tunnel)
  - ngrok:      Starts ngrok tunnel + Flask
  - cloudflare: Starts Cloudflare Tunnel (cloudflared) + Flask

Prerequisites per mode:
  ngrok:
    1. Install ngrok: https://ngrok.com/download
    2. Sign up and get auth token
    3. Run: ngrok config add-authtoken <your-token>

  cloudflare:
    1. Install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
    2. (Optional) For persistent URL: create a named tunnel in Cloudflare Zero Trust dashboard

Usage:
    python start_public_endpoint.py
"""
import subprocess
import sys
import time
import os
import secrets
from pathlib import Path
# custom imports
from config.settings import FLASK_PORT
from API_helpers.ngrok_api import start_ngrok
from API_helpers.cloudflare_api import start_cloudflared

def generate_api_key():
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tunnel_mode = os.getenv("TUNNEL_MODE", "local").lower()

    print("=" * 60)
    print("Fall Detection API - Public Endpoint Launcher")
    print(f"  Tunnel mode: {tunnel_mode}")
    print("=" * 60)

    # Ensure public endpoint security is active for tunnel modes
    os.environ["PUBLIC_ENDPOINT_ENABLED"] = "true"
    os.environ["FLASK_DEBUG"] = "false"

    # Use existing API key from .env, only generate if not set
    api_key = os.getenv("API_KEYS", "").strip()
    if not api_key:
        api_key = generate_api_key()
        print(f"[INFO] Generated new API key (set API_KEYS in .env to keep it stable)")
    else:
        print(f"[INFO] Using API key from .env")
    os.environ["API_KEYS"] = api_key

    tunnel_proc = None
    public_url = None

    # --- Start tunnel based on mode ---
    if tunnel_mode == "ngrok":
        region = os.getenv("NGROK_REGION", "eu")
        tunnel_proc, public_url = start_ngrok(region)

    elif tunnel_mode == "cloudflare":
        token = os.getenv("CLOUDFLARE_TUNNEL_TOKEN", "")
        tunnel_proc, public_url = start_cloudflared(token)

    elif tunnel_mode == "local":
        public_url = f"http://localhost:{FLASK_PORT}"
        print(f"\n[INFO] Local mode - no tunnel. Access at {public_url}")

    else:
        print(f"[ERROR] Unknown TUNNEL_MODE: {tunnel_mode}")
        sys.exit(1)

    # --- Display connection info ---
    print("\n" + "=" * 60)
    print("PUBLIC ENDPOINT READY")
    print("=" * 60)
    print(f"  Public URL:  {public_url}")
    print(f"  API Key:     {api_key}")
    print(f"  Tunnel:      {tunnel_mode}")
    print()
    # print("Share with partners:")
    # print("-" * 60)
    # print(f"  POST {public_url}/trigger")
    # print(f"  Header: X-API-Key: {api_key}")
    # print("-" * 60)
    if tunnel_mode == "ngrok":
        print(f"  ngrok dashboard: http://127.0.0.1:4040")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    # --- Start Flask ---
    flask_process = None
    try:
        flask_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=Path(__file__).parent,
        )

        while True:
            time.sleep(1)
            if flask_process.poll() is not None:
                print("\n[ERROR] Flask server stopped unexpectedly")
                break
            if tunnel_proc and tunnel_proc.poll() is not None:
                print(f"\n[ERROR] {tunnel_mode} tunnel stopped unexpectedly")
                break

    except KeyboardInterrupt:
        print("\n\n[INFO] Shutting down...")

    finally:
        if tunnel_proc:
            tunnel_proc.terminate()
        if flask_process:
            flask_process.terminate()
        print("[OK] All services stopped")


if __name__ == "__main__":
    main()
