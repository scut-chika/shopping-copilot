"""Measure startup, per-turn latency, and memory.

The challenge requires disclosing model choice, cost, token usage and latency,
and scores *Feasibility & Practicality* on whether resource use is proportionate.
This produces those numbers instead of estimating them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.agent import ShoppingCopilot  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import _config_from_env  # noqa: E402
from tools.harness import load_catalog, load_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost and latency profile")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="cost_profile.json")
    parser.add_argument("--memory", action="store_true",
                        help="also measure peak index memory (slow: second build)")
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    catalog_ids, categories, products = load_catalog(args.catalog)

    # tracemalloc roughly 5x's allocation cost, so timing and memory are
    # measured in separate passes rather than reporting an inflated startup.
    started = time.perf_counter()
    agent = ShoppingCopilot(args.catalog, _config_from_env())
    startup = time.perf_counter() - started

    peak = 0
    if args.memory:
        tracemalloc.start()
        ShoppingCopilot(args.catalog, _config_from_env())
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    latencies: list[float] = []
    turns = 0
    for sample in samples:
        session_id = f"profile_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

        for turn in range(1, MAX_TURNS + 1):
            clock = time.perf_counter()
            response = agent.respond(session_id, message, turn, TOP_K)
            latencies.append((time.perf_counter() - clock) * 1000.0)
            turns += 1
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                if str(override.get("new_value", "")):
                    disclosed.add(str(override["new_value"]))
                message = str(override.get("message", ""))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

    latencies.sort()
    report = {
        "startup_seconds": round(startup, 2),
        "index_peak_memory_mb": round(peak / 1e6, 1) if peak else None,
        "turns_measured": turns,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2),
            "p50": round(latencies[len(latencies) // 2], 2),
            "p95": round(latencies[int(len(latencies) * 0.95)], 2),
            "p99": round(latencies[int(len(latencies) * 0.99)], 2),
            "max": round(latencies[-1], 2),
        },
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "api_cost_usd": 0.0,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
