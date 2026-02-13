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
import json
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

FLASK_PORT = 8000


def generate_api_key():
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# ngrok helpers
# ---------------------------------------------------------------------------

def check_ngrok_installed():
    try:
        result = subprocess.run(["ngrok", "version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"[OK] ngrok found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")

    print("[ERROR] ngrok not found!")
    print("  Install: https://ngrok.com/download")
    print("  Then:    ngrok config add-authtoken <your-token>")
    return False


def get_ngrok_url(max_retries=10):
    import urllib.request
    import urllib.error

    for _ in range(max_retries):
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                for t in data.get("tunnels", []):
                    if t.get("proto") == "https":
                        return t.get("public_url")
        except (urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(1)
    return None


def start_ngrok(region="eu"):
    """Start ngrok and return (process, public_url) or sys.exit on failure."""
    if not check_ngrok_installed():
        sys.exit(1)

    print("\n[INFO] Starting ngrok tunnel...")
    proc = subprocess.Popen(
        ["ngrok", "http", str(FLASK_PORT), "--region", region],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    url = get_ngrok_url()
    if not url:
        print("[ERROR] Failed to get ngrok public URL")
        print("[INFO] Check ngrok dashboard at http://127.0.0.1:4040")
        proc.terminate()
        sys.exit(1)

    return proc, url


# ---------------------------------------------------------------------------
# Cloudflare Tunnel helpers
# ---------------------------------------------------------------------------

def check_cloudflared_installed():
    try:
        result = subprocess.run(["cloudflared", "version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"[OK] cloudflared found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")

    print("[ERROR] cloudflared not found!")
    print("  Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
    print("  Windows: winget install --id Cloudflare.cloudflared")
    return False


def get_cloudflared_url(proc, max_wait=15):
    """Read the public URL from cloudflared stderr output."""
    import re
    start = time.time()
    while time.time() - start < max_wait:
        line = proc.stderr.readline()
        if not line:
            time.sleep(0.2)
            continue
        line = line.decode("utf-8", errors="replace").strip()
        # cloudflared prints the URL like: https://xxx-xxx-xxx.trycloudflare.com
        match = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", line)
        if match:
            return match.group(1)
    return None


def start_cloudflared(token=""):
    """Start cloudflared and return (process, public_url) or sys.exit on failure."""
    if not check_cloudflared_installed():
        sys.exit(1)

    print("\n[INFO] Starting Cloudflare Tunnel...")

    if token:
        # Named tunnel with persistent URL (requires Cloudflare account + tunnel setup)
        cmd = ["cloudflared", "tunnel", "run", "--token", token]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # For named tunnels the URL is configured in the dashboard, print a note
        time.sleep(5)
        print("[INFO] Named tunnel started. URL is configured in your Cloudflare dashboard.")
        return proc, "(see Cloudflare dashboard for URL)"
    else:
        # Quick tunnel - free, no account needed, temporary URL
        cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{FLASK_PORT}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        url = get_cloudflared_url(proc)
        if not url:
            print("[ERROR] Failed to get Cloudflare Tunnel public URL")
            print("[INFO] Try running manually: cloudflared tunnel --url http://localhost:8000")
            proc.terminate()
            sys.exit(1)

        return proc, url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tunnel_mode = os.getenv("TUNNEL_MODE", "local").lower()
    api_key = os.getenv("API_KEYS", "") or generate_api_key()

    print("=" * 60)
    print("Fall Detection API - Public Endpoint Launcher")
    print(f"  Tunnel mode: {tunnel_mode}")
    print("=" * 60)

    # Ensure public endpoint security is active for tunnel modes
    os.environ["PUBLIC_ENDPOINT_ENABLED"] = "true"
    os.environ["FLASK_DEBUG"] = "false"
    if not os.getenv("API_KEYS"):
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
    print("Share with partners:")
    print("-" * 60)
    print(f"  POST {public_url}/trigger")
    print(f"  Header: X-API-Key: {api_key}")
    print("-" * 60)
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
