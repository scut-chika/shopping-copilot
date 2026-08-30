"""Shared evaluation harness for ablations and robustness stress tests.

This re-uses the organizer's own session protocol and scoring functions from
`evaluator/local_evaluator.py` (imported, never modified) and adds one hook:
a `perturb` callable applied to each customer utterance before the agent sees
it.  With `perturb=None` it reproduces the official score exactly.

The official number we report always comes from running the unmodified
`python -m evaluator.local_evaluator`.  This harness is for answering "what
happens when our assumptions break", which the official evaluator cannot.
"""

from __future__ import annotations

import json
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)


def load_samples(path="data/public_set.jsonl") -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def load_catalog(path="data/catalog.jsonl"):
    identifiers, categories, products = set(), {}, {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asin = str(product["parent_asin"])
            identifiers.add(asin)
            categories[asin] = [str(v) for v in (product.get("categories") or [])]
            products[asin] = product
    return identifiers, categories, products


def run(agent, samples, catalog_ids, categories, products, perturb=None) -> dict:
    """Replays the official session protocol, optionally perturbing utterances."""
    sessions: list[dict] = []
    for sample in samples:
        session_id = f"harness_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}

        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

        hit_turn = None
        best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            shown = perturb(message, turn) if perturb else message
            try:
                response = agent.respond(session_id, shown, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}

            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

        sessions.append({
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })

    return summarize(sessions)


def summarize(sessions: list[dict]) -> dict:
    def block(items):
        if not items:
            return {"n": 0, "hit_rate": 0.0, "mrr": 0.0, "mttc": None}
        return {
            "n": len(items),
            "hit_rate": round(sum(int(i["hit"]) for i in items) / len(items), 6),
            "mrr": round(statistics.fmean(i["reciprocal_rank"] for i in items), 6),
            "mttc": round(
                statistics.fmean(
                    i["first_hit_turn"] if i["first_hit_turn"] is not None else MAX_TURNS + 1
                    for i in items
                ),
                6,
            ),
        }

    overall = block(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - overall["mttc"]) / 10.0))
    grouped = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "technical_score": round(
            0.50 * overall["hit_rate"] + 0.30 * overall["mrr"] + 0.20 * efficiency, 6
        ),
        "by_scenario": {name: block(grouped[name]) for name in sorted(grouped)},
    }
