import subprocess
import json
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv
# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")
FLASK_PORT = os.getenv("FLASK_PORT")
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
