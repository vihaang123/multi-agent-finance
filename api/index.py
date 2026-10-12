"""Vercel Python function entrypoint: exposes the FastAPI app from server/app.py (all routes live under /api)."""

import os
import sys

# The repo root holds the engine packages (config, environment, agents, ...); make them importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.app import app  # noqa: E402,F401
