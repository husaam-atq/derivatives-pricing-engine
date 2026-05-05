"""Project-wide configuration and optional acceleration discovery."""

from __future__ import annotations

from importlib.util import find_spec

HAS_NUMBA = find_spec("numba") is not None
HAS_CUPY = find_spec("cupy") is not None

DEFAULT_CONFIDENCE_LEVEL = 0.95
TRADING_DAYS_PER_YEAR = 252
