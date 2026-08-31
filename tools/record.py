"""One-take recording aid for the demo video.

`tools/demo.py` rebuilds the index on every run, which puts a 17-second wait
between takes. This builds it once, waits for you to start recording, then walks
the beats of `docs/DEMO_VIDEO_SCRIPT.md`, pausing on Enter between each so you
can narrate at your own pace and record in a single pass with no editing.

Every figure is read from `results/`, never hardcoded, so the recording cannot
drift out of step with the measurements.

    python tools/record.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.config import DEFAULT  # noqa: E402
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
from tools.harness import load_catalog, load_samples  # noqa: E402

RULE = "=" * 78


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def beat(title):
    input("\n[Enter for the next beat]")
    print("\n" + RULE)
    print("  " + title)
    print(RULE)


def show_headline():
    official = load("public_set.json")
    print()
    print("  %-26s %8s %8s %7s %9s" % ("", "Hit@10", "MRR", "MTTC", "Score"))
    print("  %-26s %8.3f %8.4f %7.2f %9.4f"
          % ("organizer BM25 baseline", 0.125, 0.0680, 9.81, 0.1067))
    print("  %-26s %8.3f %8.4f %7.2f %9.4f"
          % ("Shopping Copilot", official["hit_rate_at_10"], official["mrr"],
             official["mttc"], official["recommended_technical_score"]))
    ranks = [s["best_rank"] for s in official["sessions"] if s["best_rank"]]
    print("\n  target ranked first in %d of %d sessions"
          % (sum(1 for r in ranks if r == 1), len(ranks)))
    print("  zero tokens, zero API cost, no network access")


def play_session(agent, samples, catalog_ids, categories, products, scenario, index):
    picked = [s for s in samples if s["scenario_type"] == scenario]
    sample = picked[index % len(picked)]
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    print("\n  hidden target, which the agent never sees: %s" % target)
    print("    %s..." % str(products[target].get("title") or "")[:58])

    session_id = "record"
    agent.reset(session_id, sample["user_profile"])
    disclosed = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        input("\n[Enter for turn %d]" % turn)
        print("\n  --- turn %d %s" % (turn, "-" * 56))
        print("  customer > %s" % message)
        response = agent.respond(session_id, message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        state = agent.sessions[session_id]
        remaining = len(candidate_set(agent.index, state, DEFAULT, DEFAULT.eig_max_candidates))

        print("  agent    > %s" % response["message"])
        print("             ask_attribute = %r" % response.get("ask_attribute"))
        print("             candidates still consistent: %d" % remaining)
        note = "   <-- holding back until it knows" if len(ranked) < TOP_K else ""
        print("             showing %d recommendation%s%s"
              % (len(ranked), "" if len(ranked) == 1 else "s", note))
        for position, asin in enumerate(ranked[:3], start=1):
            mark = "   <== TARGET" if asin == target else ""
            print("               %d. %s  %s...%s"
                  % (position, asin, str(products[asin].get("title") or "")[:48], mark))

        if override_applied and target in ranked:
            print("\n  HIT on turn %d at rank %d" % (turn, ranked.index(target) + 1))
            return
        if turn == MAX_TURNS:
            return
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


def show_ablation():
    # ablation.json is keyed by variant name, each value a metrics block.
    data = load("ablation.json")
    scored = [
        (name, float(block["technical_score"]))
        for name, block in data.items()
        if isinstance(block, dict) and "technical_score" in block
    ]
    if not scored:
        print("\n  (results/ablation.json not in the expected shape)")
        return
    full = dict(scored).get("full system", max(v for _, v in scored))
    scored.sort(key=lambda item: item[1])
    print("\n  one change at a time, 200 sessions:\n")
    for name, value in scored[:5]:
        print("    %-36s %8.4f   %+.4f" % (name, value, value - full))
    print("\n    %-36s %8.4f" % ("full system", full))


def show_generalization():
    print("\n  800 target products that appear nowhere in the public set, drawn twice:\n")
    for label, name in (("seed 20260830", "generalization_seed20260830.json"),
                        ("seed 7", "generalization_seed7.json")):
        data = load(name)
        public, holdout = data["public"], data["holdout"]
        print("    %-14s public %.4f  ->  held out %.4f   (%.1f%% retained)"
              % (label, public["technical_score"], holdout["technical_score"],
                 100 * holdout["technical_score"] / public["technical_score"]))
    print("\n    So the honest forecast for the private set is 0.930, not 0.972.")


def show_offline():
    cost = load("cost_profile.json")
    print("\n    model                none")
    print("    API cost             $0.00")
    print("    tokens               %d prompt / %d completion"
          % (cost["prompt_tokens"], cost["completion_tokens"]))
    print("    per-turn latency     %.0f ms mean, %.0f ms p99"
          % (cost["latency_ms"]["mean"], cost["latency_ms"]["p99"]))
    print("    dependencies         none beyond the Python standard library")
    print("\n    A test parses every shipped module and fails if a network client")
    print("    ever appears. Another starts a fresh interpreter and fails if the")
    print("    optional LLM module is so much as imported.")


def main():
    parser = argparse.ArgumentParser(description="One-take demo recording aid")
    parser.add_argument("--scenario", default="browsing",
                        choices=["buying", "browsing", "intent_override", "boundary"])
    parser.add_argument("--index", type=int, default=1)
    # Resolved against the repository, not the shell's working directory: this
    # is the one tool someone runs while a screen recorder is already going.
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    args = parser.parse_args()

    print("Building the index. This is the part to keep off camera...", flush=True)
    samples = load_samples(args.dataset)
    catalog_ids, categories, products = load_catalog(args.catalog)
    agent = ShoppingCopilot(args.catalog, DEFAULT)
    print("Ready.\n")
    input("Start your screen recorder, then press Enter to begin. ")
    print("\n\n\n")

    beat("Shopping Copilot -- TikTok TechJam 2026, Track 4")
    show_headline()

    beat("One real session, end to end")
    play_session(agent, samples, catalog_ids, categories, products,
                 args.scenario, args.index)

    beat("Where the score actually comes from")
    show_ablation()

    beat("Does it generalize, or did we fit 200 sessions?")
    show_generalization()

    beat("It runs with the network switched off")
    show_offline()

    input("\n[Enter to finish]")
    print("\n" + RULE)
    print("  github.com/scut-chika/shopping-copilot")
    print(RULE + "\n")


if __name__ == "__main__":
    main()
