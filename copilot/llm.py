"""Optional LLM stage. Never imported unless a `use_llm_*` flag is on.

**This module is not on the scored path.** The submitted configuration has both
flags off, makes no network call, and reports zero tokens. `copilot/llm.py` is
imported lazily from inside the enabling branch so that the offline guarantee is
a property of the import graph rather than a promise -- `tests/test_copilot.py`
asserts that no module on the scored path imports it at all.

### What it is for

Not for writing the customer-facing prose: the evaluator never reads `message`,
only the structured `ask_attribute` field, so a better-worded question is worth
exactly nothing. And not for choosing the question either -- the simulator's
reply rule is public and deterministic, so the best question has a closed-form
answer that a prompt can only approximate.

The one place a model genuinely helps is the *inbound* direction. The competition
specification reserves the organizer's right to add natural-language paraphrasing
to the customer simulator. When that happens our template regexes stop matching
and the rule engine goes blind -- which is exactly what the L1-L4 robustness
numbers show. A model can map a reworded sentence back onto the catalog
constraint that produced it, and hand that to the exact index.

### Containment

The model is a *selector*, never a source. `select_constraint` is given a
shortlist drawn from the real catalog and its answer is rejected unless it is
one of them verbatim. A hallucinated constraint cannot reach retrieval, because
a string that is not already in `card_index` has no posting list to contribute.

Every entry point returns its input unchanged on any failure -- missing
credentials, HTTP error, timeout, malformed JSON -- and never raises. An
exception here would cost a turn, which is worth more than the stage itself.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

_SELECT_SYSTEM = (
    "You map a shopper's paraphrased sentence back to the exact product-catalog "
    "constraint string it came from. Answer with one line: the chosen candidate "
    "copied verbatim, or NONE if none of them is what the shopper meant. "
    "Never invent text that is not in the candidate list."
)

_RERANK_SYSTEM = (
    "You rank candidate products against what a shopper has said they want. "
    "Answer with one line: the candidate numbers, best first, comma separated. "
    "Include every number exactly once. No other text."
)


class LLMClient:
    """Minimal OpenAI-compatible chat client over the standard library.

    Deliberately not a dependency: `urllib` keeps the whole submission free of
    third-party packages, so the same tree runs with or without the network.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("COPILOT_LLM_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
        self.model = model or os.environ.get("COPILOT_LLM_MODEL") or DEFAULT_MODEL
        # Credentials come from the environment only. Never written to a file,
        # a log line, or any results artifact.
        self.api_key = api_key or os.environ.get("COPILOT_LLM_API_KEY") or ""
        self.timeout = timeout
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.failures = 0

    def available(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------- transport

    def _chat(self, system: str, user: str, max_tokens: int = 2048) -> str | None:
        if not self.available():
            return None
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Generous on purpose. This family is a reasoning model: it spends
            # the budget on `reasoning_content` first and only then emits an
            # answer, so a tight cap does not truncate the answer -- it returns
            # an empty string with no error at all. 512 silently lost roughly a
            # third of the calls before this was raised.
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode("utf-8")

        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            self.failures += 1
            return None

        usage = body.get("usage") or {}
        if isinstance(usage.get("prompt_tokens"), int):
            self.prompt_tokens += usage["prompt_tokens"]
        if isinstance(usage.get("completion_tokens"), int):
            self.completion_tokens += usage["completion_tokens"]

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            self.failures += 1
            return None
        return content.strip() if isinstance(content, str) else None

    # ------------------------------------------------------------- selection

    def select_constraint(self, message: str, candidates: list[str]) -> str | None:
        """Which catalog constraint did this paraphrased sentence come from?

        Returns one of `candidates` verbatim, or None. Anything the model says
        that is not on the list is discarded, so it can only select.
        """
        if not candidates or not self.available():
            return None
        listing = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(candidates[:24]))
        answer = self._chat(
            _SELECT_SYSTEM,
            f"Shopper said:\n{message}\n\nCandidates:\n{listing}\n\n"
            "Which candidate is it? Copy it exactly, or answer NONE.",
        )
        if not answer:
            return None
        answer = answer.strip().strip('"').strip()
        if answer.upper() == "NONE":
            return None
        for text in candidates:
            if answer == text:
                return text
        # Tolerate the model echoing the numbered form, but still only ever
        # return a string that came from the catalog.
        head, _, tail = answer.partition(". ")
        if head.isdigit():
            position = int(head) - 1
            if 0 <= position < len(candidates) and tail.strip() == candidates[position]:
                return candidates[position]
        return None

    # -------------------------------------------------------------- reranking

    def rerank(self, dialog: str, candidates: list[tuple[str, str]]) -> list[str] | None:
        """Reorder `(asin, description)` pairs. Returns asins, or None."""
        if not candidates or not self.available():
            return None
        listing = "\n".join(
            f"{i + 1}. {text[:180]}" for i, (_, text) in enumerate(candidates)
        )
        answer = self._chat(
            _RERANK_SYSTEM,
            f"Shopper has said:\n{dialog[:1500]}\n\nCandidates:\n{listing}\n\n"
            "Best first, comma separated numbers only.",
            max_tokens=4096,
        )
        if not answer:
            return None
        order: list[str] = []
        used: set[int] = set()
        for chunk in answer.replace("\n", ",").split(","):
            chunk = chunk.strip().rstrip(".")
            if not chunk.isdigit():
                continue
            position = int(chunk) - 1
            if 0 <= position < len(candidates) and position not in used:
                used.add(position)
                order.append(candidates[position][0])
        if not order:
            return None
        # Anything the model dropped keeps its original relative position at the
        # tail, so a partial answer degrades instead of losing candidates.
        for position, (asin, _) in enumerate(candidates):
            if position not in used:
                order.append(asin)
        return order

    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
