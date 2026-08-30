"""Replay one session with the agent's reasoning exposed.

This is the "one demonstrated multi-turn session" deliverable, and the script to
screen-record for the demo video: the challenge puts UI/UX out of scope, so a
walkthrough of the session transcript and the agent's internal state is the
appropriate demonstration.

    python tools/demo.py                  # first buying session
    python tools/demo.py --scenario browsing
    python tools/demo.py --index 3 --scenario intent_override
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.retrieval import candidate_set  # noqa: E402
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

RULE = "=" * 78


def title_of(products, asin: str, width: int = 62) -> str:
    text = str(products[asin].get("title") or "")
    return text[:width] + ("..." if len(text) > width else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one session verbosely")
    parser.add_argument("--scenario", default="buying",
                        choices=["buying", "browsing", "intent_override", "boundary"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    catalog_ids, categories, products = load_catalog(args.catalog)
    pool = [s for s in samples if s["scenario_type"] == args.scenario]
    if not pool:
        raise SystemExit(f"no {args.scenario} sessions found")
    sample = pool[args.index % len(pool)]

    agent = ShoppingCopilot(args.catalog, _config_from_env())
    session_id = f"demo_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    print(RULE)
    print(f"scenario   : {sample['scenario_type']}   ({sample['sample_id']}, "
          f"difficulty={sample.get('difficulty_bucket')})")
    print(f"hidden target : {target}  {title_of(products, target)}")
    print(f"profile    : {sample['user_profile'].get('summary', '')}")
    print(RULE)
    print("The agent never sees the two lines above.\n")

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f"--- turn {turn} " + "-" * 63)
        print(f"customer > {message}")
        response = agent.respond(session_id, message, turn, TOP_K)
        state = agent.sessions[session_id]

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        alive = candidate_set(agent.index, state, agent.config, agent.config.eig_max_candidates)

        print(f"agent    > {response['message']}")
        print(f"           ask_attribute = {response['ask_attribute']!r}")
        print(f"  state  : category={state.category!r} scenario={state.scenario}")
        print(f"           constraints held ({len(state.constraints)}):")
        for item in state.constraints:
            flags = "".join(
                [("K" if item.known else "-"), ("M" if item.mined else "-"),
                 ("E" if item.emphasized else "-")]
            )
            print(f"             [{flags}] {item.text[:66]}")
        print(f"           candidates still consistent: {len(alive)}")
        print("  top 3  :")
        for rank, asin in enumerate(ranked[:3], start=1):
            mark = "  <== TARGET" if asin == target else ""
            print(f"             {rank}. {asin}  {title_of(products, asin, 48)}{mark}")

        if override_applied and target in ranked:
            position = ranked.index(target) + 1
            print(f"\n{RULE}\n HIT on turn {turn} at rank {position}  "
                  f"(reciprocal rank {1 / position:.3f})\n{RULE}")
            return
        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, please ignore my earlier preference."))
            print("           [simulator: intent override fires next turn; "
                  "hits do not count until now]")
        else:
            message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )
        print()

    print(f"\n{RULE}\n MISS after {MAX_TURNS} turns\n{RULE}")


if __name__ == "__main__":
    main()
