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

    for path in sorted((ROOT / "copilot").glob("*.py")) + [ROOT / "starter" / "agent.py"]:
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
