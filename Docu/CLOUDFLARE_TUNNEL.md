# Cloudflare Tunnel - How It Works

## What is Cloudflare?

**Cloudflare** is a large internet infrastructure company that sits between your server and the internet. They provide CDN, DDoS protection, DNS, and many other services used by ~20% of all websites.

**Cloudflare Tunnel** (formerly "Argo Tunnel") is one specific product that lets you expose a local service (like your Flask server) to the internet **without opening any ports or configuring firewalls**.

## How Cloudflare Tunnel Works

```
Your Partner's Computer                   Your Machine
(anywhere in the world)                   (running Flask)

  Browser / Python script                  Flask :8000
        |                                      |
        v                                      v
  HTTPS request                           cloudflared
        |                                      |
        v                                      | (outbound connection)
   +-----------+                               |
   | Cloudflare|<------------------------------+
   | Edge      |  encrypted tunnel
   | Network   |  (your machine connects OUT to Cloudflare,
   +-----------+   NOT the other way around)
```

### Key concept: **Outbound-only connection**

Unlike traditional port forwarding where you open a port and let the internet IN:
- `cloudflared` creates an **outbound** connection from your machine to Cloudflare's edge
- Your firewall doesn't need any changes (outbound HTTPS is always allowed)
- Cloudflare receives requests from the internet and forwards them through the tunnel
- Your local service never touches the internet directly

This is why it works behind corporate firewalls, NATs, etc.

## Cloudflare Tunnel vs ngrok

| Feature | Cloudflare Tunnel | ngrok |
|---------|------------------|-------|
| **Free tier** | Unlimited bandwidth | 1 online agent, limited |
| **Quick tunnel** (no account) | Yes (`trycloudflare.com`) | No (account required) |
| **Persistent URL** | Yes (with free account) | Paid plans only |
| **Speed** | Very fast (global CDN) | Good |
| **Rate limits** | None on free tier | Yes on free tier |
| **Custom domain** | Yes (free, if domain is on Cloudflare) | Paid |
| **Authentication** | Built-in (Cloudflare Access) | Basic auth |
| **Setup complexity** | Simple | Simple |

**Summary**: Cloudflare Tunnel is more generous on the free tier and can give you persistent URLs for free, but requires a few more setup steps for named tunnels.

## Two Modes of Operation

### 1. Quick Tunnel (No Account Needed)

The simplest option. Run one command and get a temporary public URL.

```bash
cloudflared tunnel --url http://localhost:8000
```

- Gives you a random URL like `https://verb-noun-adjective.trycloudflare.com`
- URL changes every time you restart
- No account, no login, no configuration
- Perfect for testing and quick demos

### 2. Named Tunnel (Free Account, Persistent URL)

For production use. The URL stays the same across restarts.

Requires:
1. A free Cloudflare account
2. A domain on Cloudflare (can be a cheap one, or use a free subdomain)
3. One-time tunnel setup

## Installation

### Windows
```bash
winget install --id Cloudflare.cloudflared
```

### macOS
```bash
brew install cloudflared
```

### Linux (Debian/Ubuntu)
```bash
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install cloudflared
```

### Verify
```bash
cloudflared version
```

## Usage with Fall Detection API

### Option A: Using the unified launcher (reads TUNNEL_MODE from .env)

1. Set in `.env`:
   ```
   TUNNEL_MODE=cloudflare
   ```

2. Run:
   ```bash
   python start_public_endpoint.py
   ```

### Option B: Using the standalone Cloudflare script

```bash
python "start_public_endpoint copy.py"
```

Both will:
1. Start `cloudflared` quick tunnel
2. Print the public URL
3. Start the Flask server
4. Clean up everything on Ctrl+C

### Output example:
```
============================================================
CLOUDFLARE TUNNEL READY
============================================================
  Public URL:  https://some-random-words.trycloudflare.com
  API Key:     G508N-Z3Y4Vcds4awadPSMiBdNs5AQrKwhFzKKgJH6w

Share with partners:
------------------------------------------------------------
  POST https://some-random-words.trycloudflare.com/trigger
  Header: X-API-Key: G508N-Z3Y4Vcds4awadPSMiBdNs5AQrKwhFzKKgJH6w
------------------------------------------------------------
```

## Setting Up a Named Tunnel (Persistent URL)

If you want a permanent URL that doesn't change:

### Step 1: Login
```bash
cloudflared tunnel login
```
This opens a browser. Select the domain you want to use.

### Step 2: Create tunnel
```bash
cloudflared tunnel create fall-detection
```

### Step 3: Create DNS record
```bash
cloudflared tunnel route dns fall-detection fall-api.yourdomain.com
```

### Step 4: Get the tunnel token
Go to https://one.dash.cloudflare.com -> Zero Trust -> Networks -> Tunnels
-> Click your tunnel -> Get the token

### Step 5: Add to .env
```
CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoiYWJj...your-long-token...
```

Now your API is permanently at `https://fall-api.yourdomain.com/trigger`.

## Security Notes

- Quick tunnels are **public** - anyone with the URL can reach your server
- That's why API key authentication is enabled automatically
- Cloudflare adds DDoS protection and TLS by default
- For extra security with named tunnels, you can add Cloudflare Access policies
  (email verification, SSO, etc.) through the Zero Trust dashboard
