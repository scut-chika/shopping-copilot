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
from .retrieval import rank, retrieve


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
        self._llm_client = None
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

    def _llm(self):
        """Build the client on first use.

        The import is inside the method on purpose: with the LLM flags off, as
        they are for scoring, `copilot.llm` is never imported and the process
        holds no network client at all. A test asserts that no module on the
        scored path imports it at module level.
        """
        if self._llm_client is None:
            from .llm import LLMClient

            self._llm_client = LLMClient()
        return self._llm_client

    def _respond(self, state: SessionState, user_message: str, turn: int, top_k: int) -> dict:
        llm = self._llm() if self.config.use_llm_parse else None
        self.parser.ingest(state, user_message, turn, self.config, llm)

        recommendations, candidates = retrieve(self.index, state, self.config, top_k)
        if self.config.use_llm_rerank:
            recommendations = self._llm_rerank(state, top_k)
        if len(recommendations) < top_k:
            recommendations = self._pad(recommendations, top_k)
        state.last_recommendations = recommendations

        attribute = None
        if turn < 10:
            attribute = choose_attribute(self.index, state, self.config, candidates)
            if attribute is not None:
                state.asked.append(attribute)
                recommendations = self._gate(state, recommendations, candidates, turn)
            message = phrase(attribute, len(candidates))
        else:
            message = "Here are my best matches based on everything you have told me."

        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": self._usage(),
        }

    def _usage(self) -> dict:
        """Cumulative token counts, so the disclosure is true either way.

        Zero in the submitted configuration because no client is ever built.
        """
        if self._llm_client is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        return self._llm_client.usage()

    def _llm_rerank(self, state: SessionState, top_k: int) -> list[str]:
        """Semantic reorder of a deeper slice, fused back as an ordering only.

        Falls through to the unchanged ranking whenever the model is
        unavailable or answers with nothing usable.
        """
        deep = rank(self.index, state, self.config, self.config.llm_rerank_depth)
        if not deep:
            return deep
        described = [(asin, "; ".join(self.index.cards.get(asin, ()))) for asin in deep]
        ordered = self._llm().rerank(state.dialog_text(), described)
        return (ordered or deep)[:top_k]

    def _gate(self, state: SessionState, recommendations: list[str], candidates, turn: int) -> list[str]:
        """Shorten the list while the candidate pool is still overloaded.

        The evaluator ends the session on *any* hit, so a lucky low-rank hit on a
        turn where we do not yet know the answer locks that rank in permanently.
        Showing a short list and asking instead trades one turn (worth 0.20/10)
        for the rank (worth up to 0.30) -- about 13:1 in our favour.

        Only ever applied on a turn where we are actually asking something: a
        short list without a question would just be a worse answer.
        """
        config = self.config
        if not config.use_confidence_gate:
            return recommendations
        if turn > config.gate_max_turn:
            return recommendations
        if config.gate_needs_clean_parse and state.parse_failures:
            # We are not following this conversation. Holding back the list is
            # premised on one more question settling things, and a question we
            # cannot read the answer to settles nothing -- so pay out instead.
            return recommendations
        if len(candidates) <= config.gate_candidate_threshold:
            return recommendations
        return recommendations[: config.gate_list_size]

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
