import subprocess
# import re
import sys
# import os
import time
# from pathlib import Path
from config.settings import FLASK_PORT
# from dotenv import load_dotenv
# # Load .env from project root
# load_dotenv(Path(__file__).parent.parent / ".env")
# FLASK_PORT = os.getenv("FLASK_PORT")


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

