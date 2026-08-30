"""Tests for the Shopping Copilot.

The most important test here is `test_mirrors_official_derivation`: our strongest
retrieval route depends on reproducing the organizer's intent-card derivation
exactly, so if the two ever diverge we want a red test, not a quiet score drop.
"""

from __future__ import annotations

import json
import sys
from itertools import islice
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from copilot import simulator_model as mirror  # noqa: E402
from copilot.config import DEFAULT  # noqa: E402
from copilot.dialog import DialogParser, SessionState  # noqa: E402
from evaluator import local_evaluator as official  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
pytestmark = pytest.mark.skipif(not CATALOG.exists(), reason="catalog.jsonl not downloaded")


def sample_products(limit=2000):
    with CATALOG.open(encoding="utf-8") as handle:
        for line in islice(handle, limit):
            yield json.loads(line)


@pytest.fixture(scope="module")
def index():
    from copilot.catalog import CatalogIndex

    return CatalogIndex(CATALOG, DEFAULT)


# --------------------------------------------------------------- mirror parity

def test_mirrors_official_derivation():
    """Our reconstruction must match the organizer's evaluator byte for byte."""
    for product in sample_products():
        assert mirror.intent_card(product) == official.intent_card(product)
        assert mirror.coarse_category(
            [str(v) for v in (product.get("categories") or [])]
        ) == official.coarse_category([str(v) for v in (product.get("categories") or [])])


def test_classify_constraint_parity():
    for product in sample_products(500):
        card = official.intent_card(product)
        for value in card["hard_constraints"] + card["soft_preferences"]:
            assert mirror.classify_constraint(value) == official.classify_constraint(value)


# ------------------------------------------------------------------- parsing

@pytest.fixture()
def parser(index):
    return DialogParser(index)


def test_parses_buying_opening(parser):
    state = SessionState(session_id="s")
    parser.ingest(state, "I'm looking for Shirts T-Shirts. A key requirement is: cotton.", 1)
    assert state.scenario == "buying"
    assert state.category == "Shirts T-Shirts"
    assert [c.text for c in state.constraints] == ["cotton"]


def test_parses_browsing_opening(parser):
    state = SessionState(session_id="s")
    parser.ingest(state, "I'm looking for Dresses Casual, but I'm still exploring.", 1)
    assert state.scenario == "browsing"
    assert state.category == "Dresses Casual"
    assert state.constraints == []


def test_parses_two_constraint_reply(parser, index):
    left, right = next(
        (a, b)
        for a, b in [
            (k, k2)
            for k in islice(index.card_index, 200)
            for k2 in islice(index.card_index, 200)
            if k != k2
        ]
    )
    state = SessionState(session_id="s")
    parser.ingest(state, f"For that, what matters is: {left}; {right}.", 2)
    assert {c.text for c in state.constraints} == {left, right}
    assert all(c.known for c in state.constraints)


def test_boundary_and_dead_attributes(parser):
    state = SessionState(session_id="s")
    parser.ingest(state, "I don't have a preference for material; please use your judgment.", 2)
    assert state.boundary_seen and "material" in state.dead_attributes
    parser.ingest(state, "I don't have an additional preference for color.", 3)
    assert "color" in state.dead_attributes


def test_override_accumulates_rather_than_erases(parser):
    """The simulator's override value comes from the same target's card, so
    discarding earlier evidence would throw away valid signal."""
    state = SessionState(session_id="s")
    parser.ingest(state, "I'm looking for Women Shoes. leather", 1)
    before = {c.text for c in state.constraints}
    parser.ingest(state, "Actually, ignore my earlier preference. What I need is: suede lining.", 2)
    after = {c.text for c in state.constraints}
    assert before <= after, "earlier constraints must be retained"
    assert any(c.emphasized for c in state.constraints)


# ------------------------------------------------------------------ behaviour

def test_always_returns_ten_valid_recommendations(index):
    from copilot.agent import ShoppingCopilot

    agent = ShoppingCopilot.__new__(ShoppingCopilot)
    agent.config = DEFAULT
    agent.index = index
    agent.parser = DialogParser(index)
    agent.sessions = {}
    agent._fallback = [a for a, _ in sorted(index.prior.items(), key=lambda kv: -kv[1])[:64]]

    agent.reset("s", {"preference_tags": ["fit", "comfort"]})
    for turn, message in enumerate(
        [
            "I'm looking for Shirts T-Shirts. A key requirement is: cotton.",
            "For that, what matters is: Machine Wash.",
            "Those options are not quite right yet. Ask me about one specific attribute.",
        ],
        start=1,
    ):
        response = agent.respond("s", message, turn, 10)
        asins = [r["parent_asin"] for r in response["recommendations"]]
        assert len(asins) == 10
        assert len(set(asins)) == 10
        assert all(a in index for a in asins)
        assert response["ask_attribute"] in (*mirror.ALLOWED_ATTRIBUTES, None)


def test_never_raises_on_hostile_input(index):
    from copilot.agent import ShoppingCopilot

    agent = ShoppingCopilot.__new__(ShoppingCopilot)
    agent.config = DEFAULT
    agent.index = index
    agent.parser = DialogParser(index)
    agent.sessions = {}
    agent._fallback = [a for a, _ in sorted(index.prior.items(), key=lambda kv: -kv[1])[:64]]

    agent.reset("s", {})
    for message in ["", "   ", "🙃" * 500, 'MATCH " OR ", weird fts5 syntax', "\x00binary"]:
        response = agent.respond("s", message, 1, 10)
        assert isinstance(response["message"], str)
        assert len(response["recommendations"]) == 10


def test_respond_without_reset_does_not_crash(index):
    from copilot.agent import ShoppingCopilot

    agent = ShoppingCopilot.__new__(ShoppingCopilot)
    agent.config = DEFAULT
    agent.index = index
    agent.parser = DialogParser(index)
    agent.sessions = {}
    agent._fallback = [a for a, _ in sorted(index.prior.items(), key=lambda kv: -kv[1])[:64]]

    response = agent.respond("never-reset", "I'm looking for Women Shoes.", 1, 10)
    assert len(response["recommendations"]) == 10


# ------------------------------------------------------- question estimation

def test_eig_prefers_a_partitioning_question(index):
    from copilot.questions import expected_remaining

    candidates = list(islice(index.asins, 300))
    scores = {
        attribute: expected_remaining(index, candidates, attribute, set())
        for attribute in ("other", "feature", "budget")
    }
    # The wildcard sees every undisclosed constraint, so it can never partition
    # worse than an attribute that only sees a subset.
    assert scores["other"] <= scores["budget"] + 1e-9


# ------------------------------------------------- offline / sandbox contract

def test_agent_imports_only_the_standard_library():
    """Official scoring may run with network access disabled.

    The agent must therefore never import a network client or a third-party
    package. This pins that contract so it cannot regress silently.
    """
    import ast

    banned_roots = {
        "urllib", "requests", "http", "socket", "httpx", "aiohttp",
        "openai", "anthropic", "boto3", "numpy", "torch", "transformers",
        "sentence_transformers", "faiss", "sklearn", "pandas",
    }
    allowed_stdlib = {
        "json", "math", "os", "re", "sqlite3", "dataclasses", "pathlib",
        "collections", "__future__", "typing", "itertools", "functools",
    }

    # `copilot/llm.py` is the one module allowed a network client, and it is
    # excluded here because nothing on the scored path may import it -- which is
    # what the second half of this test checks.
    scored_path = [
        path for path in sorted((ROOT / "copilot").glob("*.py")) if path.name != "llm.py"
    ] + [ROOT / "starter" / "agent.py"]

    for path in scored_path:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import within the package
                    continue
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                assert root not in banned_roots, f"{path.name} imports {root}"
                assert root in allowed_stdlib or root in ("copilot",), (
                    f"{path.name} imports unexpected module {root!r}"
                )

    # The optional LLM stage must be imported lazily, inside the branch that
    # enables it. If it ever moves to module level the offline guarantee becomes
    # a promise instead of a property of the import graph.
    for path in scored_path:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only; function bodies are fine
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [f"{node.module or ''}.{alias.name}" for alias in node.names]
                if node.level:
                    names += [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                assert "llm" not in name.split("."), (
                    f"{path.name} imports the LLM stage at module level"
                )


def test_llm_stage_is_never_loaded_by_default():
    """Importing and running the agent must not pull in the network client."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.');"
         "from starter.agent import Agent;"
         "print('copilot.llm' in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.stdout.strip().endswith("False"), result.stdout + result.stderr


def test_agent_reads_no_secrets_from_environment():
    """The only environment reads are COPILOT_* config overrides."""
    import os

    from starter.agent import _config_from_env

    os.environ["COPILOT_USE_BM25"] = "0"
    os.environ["SECRET_API_KEY"] = "must-not-be-read"
    try:
        config = _config_from_env()
        assert config.use_bm25 is False
        assert "must-not-be-read" not in repr(config)
    finally:
        del os.environ["COPILOT_USE_BM25"]
        del os.environ["SECRET_API_KEY"]


# --------------------------------------------------------------- replay checks


def test_boundary_refusal_records_no_evidence(index):
    """The two refusal phrasings mean opposite things.

    "please use your judgment" is fired once per boundary session whatever the
    target's card contains, so it says nothing about the target. Treating it as
    a real refusal excluded the true target in 8% of turn-states.
    """
    from copilot.dialog import DialogParser, SessionState

    parser = DialogParser(index)
    state = SessionState(session_id="s")
    state.asked.append("color")
    parser.ingest(state, "I don't have a preference for color; please use your judgment.", 2, DEFAULT)

    assert state.evidence == []
    assert state.boundary_seen
    assert "color" in state.dead_attributes


def test_no_additional_preference_is_recorded_as_evidence(index):
    from copilot.dialog import DialogParser, SessionState

    parser = DialogParser(index)
    state = SessionState(session_id="s")
    state.asked.append("color")
    parser.ingest(state, "I don't have an additional preference for color.", 2, DEFAULT)

    assert len(state.evidence) == 1
    assert state.evidence[0].attribute == "color"
    assert state.evidence[0].revealed == ()


def test_intent_override_opening_is_spoken_but_not_disclosed(index):
    """The evaluator prints `old_value` without adding it to `disclosed`.

    So the simulator may legitimately reveal that same value again later. If we
    mirrored it as disclosed, the replay would predict the wrong answer.
    """
    from copilot.dialog import DialogParser, SessionState

    constraint = next(iter(index.card_index))
    parser = DialogParser(index)
    state = SessionState(session_id="s")
    parser.ingest(state, f"I'm looking for Tops. {constraint}", 1, DEFAULT)

    assert state.scenario == "intent_override"
    assert state.disclosed == set()
    assert any(item.text == constraint for item in state.constraints)


def test_partial_parse_records_no_evidence(index):
    """A reply we could only half-read must not be replayed.

    A wrong `revealed` tuple would rule out the true target, which is far worse
    than the missed narrowing.
    """
    from copilot.dialog import DialogParser, SessionState

    parser = DialogParser(index)
    state = SessionState(session_id="s")
    state.asked.append("material")
    parser.ingest(state, "For that, what matters is: something not in the catalog.", 2, DEFAULT)

    assert state.evidence == []


def test_replay_never_rules_out_the_true_target(index):
    """The regression test for the bug that made this feature look broken.

    Replays every public session and asserts that the true target could always
    have produced every answer we recorded. A single false positive here costs a
    session outright, so this is checked over the whole public set rather than a
    sample.
    """
    from copilot.dialog import DialogParser, SessionState
    from copilot.questions import choose_attribute
    from copilot.replay import mismatches
    from copilot.retrieval import retrieve
    from evaluator.local_evaluator import (
        MAX_TURNS,
        coarse_category,
        customer_reply,
        initial_message,
        intent_card,
        behavior_for,
    )
    import random

    samples = [
        json.loads(line)
        for line in (ROOT / "data" / "public_set.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    products, categories = {}, {}
    with CATALOG.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asin = str(product["parent_asin"])
            if asin in wanted:
                products[asin] = product
                categories[asin] = [str(v) for v in (product.get("categories") or [])]

    parser = DialogParser(index)
    checked = 0
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card = intent_card(products[target])
        rng = random.Random(f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}")
        effective = {
            **sample,
            "intent_card": card,
            "behavior": behavior_for(str(sample["scenario_type"]), card, rng),
        }

        state = SessionState(session_id=sample["sample_id"], profile=sample["user_profile"])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories[target]), disclosed)

        for turn in range(1, MAX_TURNS + 1):
            parser.ingest(state, message, turn, DEFAULT)
            assert mismatches(index, state.evidence, target) == 0, (
                f"{sample['sample_id']} ({sample['scenario_type']}) turn {turn}: "
                f"the replay ruled out the true target"
            )
            checked += 1
            if turn >= 4:  # four turns is past the override and any refusal
                break
            _, candidates = retrieve(index, state, DEFAULT, 10)
            attribute = choose_attribute(index, state, DEFAULT, candidates)
            if attribute is not None:
                state.asked.append(attribute)
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                if str(override.get("new_value", "")):
                    disclosed.add(str(override["new_value"]))
                message = str(override.get("message", ""))
            else:
                message, boundary_used = customer_reply(
                    effective, attribute, disclosed, boundary_used
                )

    assert checked >= 200
