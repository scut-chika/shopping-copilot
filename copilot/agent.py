"""The Shopping Copilot agent.

Per-turn flow:

    parse -> accumulate constraints -> multi-route retrieval -> rank
          -> estimate question value -> answer with both a ranked list and a question

Two invariants matter for scoring and are enforced here:

* **Always return a full ranked list.**  The evaluator checks for a hit *before*
  it generates the customer's reply, and returning recommendations costs
  nothing, so there is never a reason to answer with a question alone.
* **Never raise.**  The evaluator turns an exception into an empty turn, which
  wastes one of only ten.  Every turn is wrapped and falls back to the previous
  answer, then to a popularity-ordered list.
"""

from __future__ import annotations

from pathlib import Path

from .catalog import CatalogIndex
from .config import DEFAULT, Config
from .dialog import DialogParser, SessionState
from .questions import choose_attribute, phrase
from .retrieval import retrieve


class ShoppingCopilot:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: Config = DEFAULT,
    ) -> None:
        self.config = config
        self.index = CatalogIndex(catalog_path, config)
        self.parser = DialogParser(self.index)
        self.sessions: dict[str, SessionState] = {}
        self._fallback = [
            asin
            for asin, _ in sorted(
                self.index.prior.items(), key=lambda kv: -kv[1]
            )[:64]
        ]

    # ------------------------------------------------------------- interface

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            profile=user_profile if isinstance(user_profile, dict) else {},
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self.sessions[session_id] = state
        try:
            return self._respond(state, user_message, turn, top_k)
        except Exception:
            return self._fallback_response(state, top_k)

    # --------------------------------------------------------------- internal

    def _respond(self, state: SessionState, user_message: str, turn: int, top_k: int) -> dict:
        self.parser.ingest(state, user_message, turn, self.config)

        recommendations, candidates = retrieve(self.index, state, self.config, top_k)
        if len(recommendations) < top_k:
            recommendations = self._pad(recommendations, top_k)
        state.last_recommendations = recommendations

        attribute = None
        if turn < 10:
            attribute = choose_attribute(self.index, state, self.config, candidates)
            if attribute is not None:
                state.asked.append(attribute)
            message = phrase(attribute, len(candidates))
        else:
            message = "Here are my best matches based on everything you have told me."

        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _pad(self, recommendations: list[str], top_k: int) -> list[str]:
        seen = set(recommendations)
        padded = list(recommendations)
        for asin in self._fallback:
            if len(padded) >= top_k:
                break
            if asin not in seen:
                seen.add(asin)
                padded.append(asin)
        return padded

    def _fallback_response(self, state: SessionState, top_k: int) -> dict:
        recommendations = state.last_recommendations or self._fallback[:top_k]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": a} for a in recommendations[:top_k]],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
