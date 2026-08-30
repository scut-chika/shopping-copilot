"""Ablation study: what is each component actually worth?

Builds the catalog index once and re-runs the official session protocol with one
setting changed at a time.  The index is config-independent, so swapping the
config between runs is equivalent to (and much faster than) rebuilding.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.config import DEFAULT  # noqa: E402
from tools.harness import load_catalog, load_samples, run  # noqa: E402

VARIANTS: list[tuple[str, dict]] = [
    ("full system", {}),
    ("- card-exact index", {"use_card_index": False}),
    ("- loose index", {"use_loose_index": False}),
    ("- category filter", {"use_category_filter": False}),
    ("- BM25 route", {"use_bm25": False}),
    ("- popularity prior", {"use_prior": False}),
    ("+ user profile (default off)", {"use_profile": True}),
    ("- constraint mining", {"use_constraint_mining": False}),
    ("mining ungated", {"mining_only_when_parse_fails": False}),
    ("questions: none", {"question_strategy": "none"}),
    ("questions: fixed cycle", {"question_strategy": "cycle"}),
    ("questions: always 'other'", {"question_strategy": "other"}),
    ("questions: EIG, no 'other' arm", {"allow_other_arm": False}),
    ("retrieval: BM25 only", {
        "use_card_index": False, "use_loose_index": False, "use_category_filter": False,
    }),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Component ablation")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="ablation.json")
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    catalog_ids, categories, products = load_catalog(args.catalog)
    agent = ShoppingCopilot(args.catalog, DEFAULT)

    report, full = {}, None
    header = f"{'variant':<32} {'score':>8} {'delta':>8} {'hit':>7} {'mrr':>8} {'mttc':>7}"
    print(header)
    print("-" * len(header))
    for name, overrides in VARIANTS:
        agent.config = replace(DEFAULT, **overrides) if overrides else DEFAULT
        agent.sessions.clear()
        result = run(agent, samples, catalog_ids, categories, products)
        report[name] = {"overrides": overrides, **result}
        if full is None:
            full = result["technical_score"]
        delta = result["technical_score"] - full
        print(
            f"{name:<32} {result['technical_score']:>8.4f} {delta:>+8.4f} "
            f"{result['hit_rate']:>7.3f} {result['mrr']:>8.4f} {result['mttc']:>7.3f}"
        )

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
