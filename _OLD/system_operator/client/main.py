"""
Fall Detection Inference Client — System Operator Component

Entry point for the client laptop (the machine that queries InfluxDB).
This is a thin wrapper around the original main.py logic.

Identical behaviour to root main.py — kept here so each component
has its own clear entry point under system_operator/client/.

Run from project root:
    python system_operator/client/main.py
"""

import sys
import os

# Ensure project root is on sys.path when invoked from subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import and run the original main app
from main import *  # noqa: F401, F403 — re-exports everything from root main.py
