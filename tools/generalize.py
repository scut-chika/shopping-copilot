"""Held-out generalization test: does the public-set score survive new targets?

The public set is 200 sessions and we score 1.000 Hit Rate@10 on it, which means
it can no longer tell us anything -- any further tuning fits those 200 sessions.
The private set is 800 sessions built from *different users and different target
products*, and we cannot see it.

So we build our own. The organizer's session generator is deterministic and
public: given a product and a scenario type, `intent_card()` and `behavior_for()`
produce the whole conversation. This script samples target products that appear
nowhere in the public set, synthesizes user profiles by resampling profile fields
independently, and replays the official protocol over them at the official
scenario mix (40/40/15/5).

That reproduces the one property that matters -- unseen targets -- while holding
the generator fixed. It cannot tell us about organizer paraphrasing (see
`tools/robustness.py` for that axis), but it does answer "did we fit the 200
sessions, or the task?"

    python tools/generalize.py --sessions 800
    python tools/generalize.py --sessions 800 --seed 7   # a different draw
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.agent import ShoppingCopilot  # noqa: E402
from starter.agent import _config_from_env  # noqa: E402
from tools.harness import load_catalog, load_samples, run  # noqa: E402

SCENARIO_MIX = [
    ("buying", 0.40),
    ("browsing", 0.40),
    ("intent_override", 0.15),
    ("boundary", 0.05),
]


def profile_pools(samples: list[dict]) -> dict[str, list]:
    """Collect each profile field separately so we can recombine them."""
    pools: dict[str, list] = defaultdict(list)
    for sample in samples:
        profile = sample.get("user_profile") or {}
        for key in ("average_prior_rating", "purchase_frequency", "rating_style"):
            if profile.get(key) is not None:
                pools[key].append(profile[key])
        tags = profile.get("preference_tags")
        if isinstance(tags, list) and tags:
            pools["preference_tags"].extend(tags)
    return pools


def synth_profile(pools: dict[str, list], rng: random.Random, target_text: str = "") -> dict:
    """Synthesize a profile for a held-out session.

    `preference_tags` used to be drawn purely at random, which made them
    *uncorrelated with the target by construction*. That is not a harmless
    simplification: it silently zeroes out any signal that depends on the profile
    matching the product, so the harness could neither confirm nor refute a
    profile-driven feature -- it could only return noise.

    Measured on the public set, the real tags do carry signal: a target's tag
    overlap averages 0.371 against 0.237 for its own category peers, and the
    target scores above its peers in 135 of 200 sessions. We reproduce that
    rather than assume it away: roughly two thirds of sessions draw one tag that
    genuinely occurs in the target's text, the rest stay random.
    """
    # Sample from the raw pool, not a de-duplicated one: the public set's tags are
    # far from uniform, and flattening that distribution alone halved how often a
    # random tag matches any product at all.
    catalog_tags = pools["preference_tags"]
    distinct = list(dict.fromkeys(catalog_tags))
    tags: list[str] = []
    if target_text:
        grounded = [t for t in distinct if t and t.lower() in target_text]
        rng.shuffle(grounded)
        for tag in grounded[:2]:
            if rng.random() < 0.675:
                tags.append(tag)
    while len(tags) < 3 and len(tags) < len(distinct):
        pick = rng.choice(catalog_tags)
        if pick not in tags:
            tags.append(pick)
    return {
        "average_prior_rating": rng.choice(pools["average_prior_rating"]),
        "preference_tags": tags,
        "purchase_frequency": rng.choice(pools["purchase_frequency"]),
        "rating_style": rng.choice(pools["rating_style"]),
        "summary": f"Prior purchases emphasize {', '.join(tags)}; "
                   f"ratings are {rng.choice(pools['rating_style'])}.",
    }


def build_sessions(products, public_samples, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pools = profile_pools(public_samples)

    def text_of(product: dict) -> str:
        features = product.get("features") or []
        if not isinstance(features, list):
            features = [features]
        return f"{product.get('title') or ''} {' '.join(str(f) for f in features)}".lower()

    profile_text = {asin: text_of(item) for asin, item in products.items()}

    seen_targets = {str(s["ground_truth"]["parent_asin"]) for s in public_samples}
    eligible = [asin for asin in products if asin not in seen_targets]
    rng.shuffle(eligible)
    if len(eligible) < count:
        raise SystemExit(f"only {len(eligible)} unseen products available")

    scenarios: list[str] = []
    for name, share in SCENARIO_MIX:
        scenarios.extend([name] * round(count * share))
    while len(scenarios) < count:
        scenarios.append("buying")
    scenarios = scenarios[:count]
    rng.shuffle(scenarios)

    return [
        {
            "sample_id": f"holdout_{index:04d}",
            "scenario_type": scenario,
            "category_bucket": "clothing",
            "difficulty_bucket": "unknown",
            "ground_truth": {"parent_asin": asin},
            "user_profile": synth_profile(pools, rng, profile_text.get(asin, "")),
        }
        for index, (asin, scenario) in enumerate(zip(eligible[:count], scenarios), start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out generalization test")
    parser.add_argument("--sessions", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="generalization.json")
    parser.add_argument("--dump-sessions", default=None,
                        help="write the synthesized sessions to this jsonl path")
    args = parser.parse_args()

    public_samples = load_samples(args.dataset)
    catalog_ids, categories, products = load_catalog(args.catalog)

    sessions = build_sessions(products, public_samples, args.sessions, args.seed)
    if args.dump_sessions:
        with Path(args.dump_sessions).open("w", encoding="utf-8") as handle:
            for session in sessions:
                handle.write(json.dumps(session) + "\n")

    config = _config_from_env()
    agent = ShoppingCopilot(args.catalog, config)

    print(f"held-out sessions : {len(sessions)} (seed {args.seed})")
    print(f"targets overlap public set : "
          f"{len({s['ground_truth']['parent_asin'] for s in sessions} & {str(p['ground_truth']['parent_asin']) for p in public_samples})}")
    print(f"scenario mix      : {dict(Counter(s['scenario_type'] for s in sessions))}")
    print(f"config            : mining={config.use_constraint_mining} "
          f"profile={config.use_profile} questions={config.question_strategy}")
    print()

    public = run(agent, public_samples, catalog_ids, categories, products)
    agent.sessions.clear()
    holdout = run(agent, sessions, catalog_ids, categories, products)

    header = f"{'set':<12} {'n':>5} {'score':>8} {'hit':>7} {'mrr':>8} {'mttc':>7}"
    print(header)
    print("-" * len(header))
    for name, result in (("public", public), ("held-out", holdout)):
        print(f"{name:<12} {result['n']:>5} {result['technical_score']:>8.4f} "
              f"{result['hit_rate']:>7.3f} {result['mrr']:>8.4f} {result['mttc']:>7.3f}")

    gap = holdout["technical_score"] - public["technical_score"]
    print(f"\ngeneralization gap : {gap:+.4f} "
          f"({holdout['technical_score'] / public['technical_score']:.1%} retained)")

    print("\nheld-out by scenario:")
    for name, block in sorted(holdout["by_scenario"].items()):
        print(f"  {name:<16} n={block['n']:>4}  hit={block['hit_rate']:.3f}  "
              f"mrr={block['mrr']:.4f}  mttc={block['mttc']:.3f}")

    Path(args.output).write_text(
        json.dumps({"seed": args.seed, "public": public, "holdout": holdout,
                    "gap": round(gap, 6)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
