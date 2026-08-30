"""Choosing which attribute to ask about, by expected information gain.

The simulator decides what to reveal with a deterministic rule: given
`ask_attribute`, it returns the first two *undisclosed* constraints of the
target whose classification matches.  Because that rule is public, the value of
a question is not something we have to guess at -- we can compute it.

For each candidate attribute we ask: "if the target were product X, what would
this question return?"  Candidates that would answer identically stay mutually
indistinguishable, so grouping the candidate set by predicted answer and taking
the size-weighted mean group size gives the expected number of candidates still
in play after the answer.  We ask the question that minimises it.

This is a real question-value estimator rather than a fixed script: it keeps
working if the attribute rules change, and it is what lets the agent converge in
two or three turns instead of exhausting the ten-turn budget.
"""

from __future__ import annotations

from collections import defaultdict

from .config import Config
from .dialog import SessionState
from .replay import predicted_answer
from .simulator_model import ALLOWED_ATTRIBUTES

QUESTION_TEMPLATES = {
    "material": "What material would you prefer?",
    "color": "Any colour you have in mind?",
    "size": "What size or fit are you looking for?",
    "style": "What style would suit you best?",
    "brand": "Is there a brand you prefer?",
    "budget": "What budget are you working with?",
    "feature": "Is there a particular feature that matters most to you?",
    "use_case": "What will you mainly be using it for?",
    "category": "Which kind of item are you after exactly?",
    "other": "Is there anything else that matters to you here?",
}

_CYCLE = ("feature", "material", "color", "style", "other")


def expected_remaining(index, candidates, attribute: str, disclosed: set[str]) -> float:
    """Size-weighted mean group size after asking `attribute`. Lower is better.

    `predicted_answer` is shared with `copilot.replay` on purpose: the estimator
    that picks a question and the check that scores the answer must model the
    simulator identically, or they can quietly disagree.
    """
    groups: dict[tuple[str, ...], int] = defaultdict(int)
    for asin in candidates:
        groups[predicted_answer(index, asin, attribute, disclosed)] += 1
    total = sum(groups.values())
    if not total:
        return 0.0
    return sum(size * size for size in groups.values()) / total


def choose_attribute(index, state: SessionState, config: Config, candidates) -> str | None:
    strategy = config.question_strategy
    if strategy == "none":
        return None
    if strategy == "other":
        return "other"
    if strategy == "cycle":
        for attribute in _CYCLE:
            if attribute not in state.asked and attribute not in state.dead_attributes:
                return attribute
        return "other"

    options = [
        attribute
        for attribute in ALLOWED_ATTRIBUTES
        if attribute not in state.dead_attributes
        and (config.allow_other_arm or attribute != "other")
    ]
    if not options:
        return "other"
    if not candidates:
        return "feature" if "feature" in options else options[0]

    sample = candidates[: config.eig_max_candidates]
    # The simulator's own `disclosed` set, not `seen`: mined guesses and the
    # intent-override opening are in `seen` but were never actually disclosed,
    # and counting them makes the estimator predict answers that cannot happen.
    disclosed = state.disclosed

    best_attribute, best_score = None, float("inf")
    for attribute in options:
        score = expected_remaining(index, sample, attribute, disclosed)
        # Prefer a question we have not already spent a turn on.
        if attribute in state.asked:
            score += 0.5
        if score < best_score:
            best_attribute, best_score = attribute, score

    return best_attribute


def phrase(attribute: str | None, candidate_count: int) -> str:
    if attribute is None:
        return "Here are the closest matches I found."
    question = QUESTION_TEMPLATES.get(attribute, QUESTION_TEMPLATES["other"])
    if candidate_count > 50:
        return f"I have a broad set of options so far. {question}"
    return f"Here are my closest matches. {question}"
