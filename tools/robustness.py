"""Stress test: how far does the score fall if the organizer paraphrases?

The competition specification says the organizer may add natural-language
paraphrasing to the private sessions ("If natural-language paraphrasing is added
by the organizer, it cannot decide correctness").  Our strongest retrieval route
matches constraint strings verbatim, so paraphrasing is the single most likely
way our public-set score fails to transfer.  This measures that exposure instead
of assuming it away.

Levels
    L0  official wording (sanity check: must equal the official score)
    L1  every template reworded, constraint text left verbatim
    L2  L1 plus surface edits to the constraint (case, punctuation, spacing)
    L3  L1 plus 25% of constraint words dropped
    L4  L1 plus 40% dropped and word order shuffled

L1 isolates parser brittleness; L3/L4 isolate reliance on exact matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.agent import ShoppingCopilot  # noqa: E402
from starter.agent import _config_from_env  # noqa: E402
from tools.harness import load_catalog, load_samples, run  # noqa: E402

TEMPLATES = [
    (re.compile(r"^I'm looking for (?P<c>.+?), but I'm still exploring\.$"),
     "Just browsing for {c} right now, nothing fixed yet."),
    (re.compile(r"^I'm looking for (?P<c>.+?)\. A key requirement is: (?P<b>.+)\.$"),
     "Hi! I need {c}. One thing that really matters: {b}."),
    (re.compile(r"^For that, what matters is: (?P<b>.+)\.$"),
     "Sure - for me it comes down to {b}."),
    (re.compile(r"^Actually, ignore my earlier preference\. What I need is: (?P<b>.+)\.$"),
     "Scratch that, forget what I said before. What I actually want is {b}."),
    (re.compile(r"^I don't have an additional preference for (?P<a>[\w_]+)\.$"),
     "No strong feelings about {a}, honestly."),
    (re.compile(r"^I don't have a preference for (?P<a>[\w_]+); please use your judgment\.$"),
     "No preference on {a} - you pick whatever works."),
    (re.compile(r"^Those options are not quite right yet\..*$"),
     "Hmm, none of those look right. Ask me about something specific."),
]


def _rng(text: str) -> random.Random:
    return random.Random(int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16))


def _mangle(body: str, level: int) -> str:
    if level <= 1:
        return body
    if level == 2:
        return re.sub(r"\s+", " ", body.lower().replace(",", "").replace(":", " ")).strip()
    drop = 0.25 if level == 3 else 0.40
    words = body.split()
    if len(words) <= 3:
        return body
    rng = _rng(body)
    kept = [w for w in words if rng.random() > drop] or words[:3]
    if level >= 4:
        rng.shuffle(kept)
    return " ".join(kept)


def make_perturb(level: int):
    if level <= 0:
        return None

    def perturb(message: str, turn: int) -> str:
        for pattern, template in TEMPLATES:
            match = pattern.match(message)
            if not match:
                continue
            fields = {k: v for k, v in match.groupdict().items() if v is not None}
            if "b" in fields:
                fields["b"] = _mangle(fields["b"], level)
            if "c" in fields and level >= 3:
                fields["c"] = _mangle(fields["c"], level)
            return template.format(**fields)
        return message

    return perturb


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase robustness stress test")
    parser.add_argument("--levels", default="0,1,2,3,4")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="robustness.json")
    parser.add_argument(
        "--sessions", type=int, default=None,
        help="cap the session count -- used when a variant costs money per turn",
    )
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    if args.sessions:
        samples = samples[: args.sessions]
    catalog_ids, categories, products = load_catalog(args.catalog)
    config = _config_from_env()
    agent = ShoppingCopilot(args.catalog, config)
    print(f"mining={config.use_constraint_mining} profile={config.use_profile} "
          f"questions={config.question_strategy}")

    report = {}
    print(f"{'level':>6} {'score':>8} {'hit':>7} {'mrr':>8} {'mttc':>7}   retained")
    print("-" * 56)
    baseline = None
    for level in [int(x) for x in args.levels.split(",")]:
        agent.sessions.clear()
        result = run(agent, samples, catalog_ids, categories, products, make_perturb(level))
        report[f"L{level}"] = result
        if baseline is None:
            baseline = result["technical_score"]
        retained = result["technical_score"] / baseline if baseline else 0.0
        print(
            f"{'L' + str(level):>6} {result['technical_score']:>8.4f} {result['hit_rate']:>7.3f} "
            f"{result['mrr']:>8.4f} {result['mttc']:>7.3f}   {retained:>6.1%}"
        )

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
