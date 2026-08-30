"""Grid sweep over request-time settings, on the official session protocol.

The index does not depend on any of these, so it is built once and the config
swapped between runs -- same result as rebuilding, minutes instead of hours.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.config import DEFAULT  # noqa: E402
from tools.harness import load_catalog, load_samples, run  # noqa: E402


def parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid sweep over config fields")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="sweep.json")
    parser.add_argument(
        "--grid", action="append", required=True,
        help="field=v1,v2,v3 -- repeat for a cartesian product",
    )
    args = parser.parse_args()

    axes: list[tuple[str, list]] = []
    for spec in args.grid:
        field, _, values = spec.partition("=")
        assert hasattr(DEFAULT, field), f"unknown config field {field!r}"
        axes.append((field, [parse_value(v) for v in values.split(",")]))

    samples = load_samples(args.dataset)
    catalog_ids, categories, products = load_catalog(args.catalog)
    agent = ShoppingCopilot(args.catalog, DEFAULT)

    rows = []
    for combo in itertools.product(*[values for _, values in axes]):
        overrides = dict(zip([field for field, _ in axes], combo))
        agent.config = replace(DEFAULT, **overrides)
        agent.parser.index = agent.index
        summary = run(agent, samples, catalog_ids, categories, products)
        rows.append({**overrides, **{
            k: summary[k] for k in
            ("technical_score", "hit_rate", "mrr", "mttc")
        }})
        label = " ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"{summary['technical_score']:.6f}  mrr={summary['mrr']:.4f} "
              f"mttc={summary["mttc"]:.3f} hit={summary["hit_rate"]:.4f}  {label}",
              flush=True)

    rows.sort(key=lambda r: -r["technical_score"])
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nbest: {rows[0]}")


if __name__ == "__main__":
    main()
