"""Entry point the official evaluator imports.

The implementation lives in `copilot/`; this module only adapts it to the
required `Agent` interface so the evaluator can be run unmodified.
"""

from __future__ import annotations

import os
from pathlib import Path

from copilot.agent import ShoppingCopilot
from copilot.config import DEFAULT, Config


def _config_from_env() -> Config:
    """Allow tools/ablation.py to vary one setting per run without editing code."""
    overrides: dict[str, object] = {}
    for name, value in os.environ.items():
        if not name.startswith("COPILOT_"):
            continue
        key = name[len("COPILOT_"):].lower()
        if not hasattr(DEFAULT, key):
            continue
        current = getattr(DEFAULT, key)
        if isinstance(current, bool):
            overrides[key] = value.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int) and not isinstance(current, bool):
            overrides[key] = int(value)
        elif isinstance(current, float):
            overrides[key] = float(value)
        else:
            overrides[key] = value
    from dataclasses import replace

    return replace(DEFAULT, **overrides) if overrides else DEFAULT


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._impl = ShoppingCopilot(catalog_path, _config_from_env())

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._impl.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._impl.respond(session_id, user_message, turn, top_k)
