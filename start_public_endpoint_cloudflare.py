#!/usr/bin/env python3
"""
Cloudflare Tunnel Launcher for Fall Detection API (standalone)

A simpler standalone script that ONLY uses Cloudflare Tunnel.
For the configurable version that supports local/ngrok/cloudflare,
use start_public_endpoint.py instead.

This uses "Quick Tunnels" - no Cloudflare account needed.
cloudflared gives you a random *.trycloudflare.com URL each time.

Prerequisites:
  1. Install cloudflared:
     - Windows:  winget install --id Cloudflare.cloudflared
     - macOS:    brew install cloudflared
     - Linux:    See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

Usage:
    python "start_public_endpoint copy.py"
"""
import subprocess
import sys
import time
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

FLASK_PORT = 8000


def check_cloudflared():
    """Check if cloudflared CLI is installed."""
    try:
        result = subprocess.run(
            ["cloudflared", "version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"[OK] {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("[ERROR] cloudflared not found!")
    print()
    print("Install cloudflared:")
    print("  Windows:  winget install --id Cloudflare.cloudflared")
    print("  macOS:    brew install cloudflared")
    print("  Linux:    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
    return False


def wait_for_tunnel_url(proc, max_wait=15):
    """Parse the *.trycloudflare.com URL from cloudflared stderr."""
    start = time.time()
    while time.time() - start < max_wait:
        line = proc.stderr.readline()
        if not line:
            time.sleep(0.2)
            continue
        line = line.decode("utf-8", errors="replace").strip()
        match = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", line)
        if match:
            return match.group(1)
    return None


def main():
    print("=" * 60)
    print("Fall Detection API - Cloudflare Tunnel Launcher")
    print("=" * 60)

    if not check_cloudflared():
        sys.exit(1)

    # Enable public endpoint security
    os.environ["PUBLIC_ENDPOINT_ENABLED"] = "true"
    os.environ["FLASK_DEBUG"] = "false"
    os.environ["TUNNEL_MODE"] = "cloudflare"

    api_key = os.getenv("API_KEYS", "")
    if not api_key:
        import secrets
        api_key = secrets.token_urlsafe(32)
        os.environ["API_KEYS"] = api_key

    # Start cloudflared quick tunnel
    print("\n[INFO] Starting Cloudflare Quick Tunnel...")
    cf_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{FLASK_PORT}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    public_url = wait_for_tunnel_url(cf_proc)
    if not public_url:
        print("[ERROR] Could not get tunnel URL.")
        print("[INFO] Try manually: cloudflared tunnel --url http://localhost:8000")
        cf_proc.terminate()
        sys.exit(1)

    # Display info
    print("\n" + "=" * 60)
    print("CLOUDFLARE TUNNEL READY")
    print("=" * 60)
    print(f"  Public URL:  {public_url}")
    print(f"  API Key:     {api_key}")
    print()
    print("Share with partners:")
    print("-" * 60)
    print(f"  POST {public_url}/trigger")
    print(f"  Header: X-API-Key: {api_key}")
    print("-" * 60)
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Start Flask
    flask_proc = None
    try:
        flask_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=Path(__file__).parent,
        )

        while True:
            time.sleep(1)
            if flask_proc.poll() is not None:
                print("\n[ERROR] Flask server stopped unexpectedly")
                break
            if cf_proc.poll() is not None:
                print("\n[ERROR] Cloudflare tunnel stopped unexpectedly")
                break

    except KeyboardInterrupt:
        print("\n\n[INFO] Shutting down...")

    finally:
        cf_proc.terminate()
        if flask_proc:
            flask_proc.terminate()
        print("[OK] All services stopped")


if __name__ == "__main__":
    main()
