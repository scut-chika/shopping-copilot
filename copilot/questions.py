"""Choosing which attribute to ask about, by expected information gain.

The simulator decides what to reveal with a deterministic rule: given
`ask_attribute`, it returns the first two *undisclosed* constraints of the
target whose classification matches.  Because that rule is public, the value of
a question is not something we have to guess at -- we can compute it.

For each candidate attribute we ask: "if the target were product X, what would
this question return?"  Candidates that would answer identically stay mutually
indistinguishable, so the predicted answers partition the candidate set and the
group sizes are what a question is worth.

*What* to read off those sizes is a design choice, and the two options here
genuinely disagree:

* `expected_size` -- the size-weighted mean group size, i.e. how many candidates
  survive on average.  The textbook objective, and the right one for an agent
  that simply wants a smaller set.
* `convergence` -- the fraction of candidates that land in a group small enough
  to commit to.  This is the one that matches an agent with a confidence gate in
  front of its recommendations, because such an agent is not paid for a smaller
  set, it is paid for *being able to answer*.  It prefers a question that
  isolates candidates outright over one that splits the set evenly.

Either way this is a real question-value estimator rather than a fixed script:
it keeps working if the attribute rules change, and it is what lets the agent
converge in two or three turns instead of exhausting the ten-turn budget.
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


def _partition(index, candidates, attribute: str, disclosed: set[str]) -> dict:
    """Group candidates by the answer they would produce for `attribute`.

    Candidates in the same group stay mutually indistinguishable after the
    answer, so the group sizes are what any question objective is computed from.

    `predicted_answer` is shared with `copilot.replay` on purpose: the estimator
    that picks a question and the check that scores the answer must model the
    simulator identically, or they can quietly disagree.
    """
    groups: dict[tuple[str, ...], int] = defaultdict(int)
    for asin in candidates:
        groups[predicted_answer(index, asin, attribute, disclosed)] += 1
    return groups


def expected_remaining(index, candidates, attribute: str, disclosed: set[str]) -> float:
    """Size-weighted mean group size after asking `attribute`. Lower is better."""
    groups = _partition(index, candidates, attribute, disclosed)
    total = sum(groups.values())
    if not total:
        return 0.0
    return sum(size * size for size in groups.values()) / total


def convergence_odds(index, candidates, attribute: str, disclosed: set[str], target: int) -> float:
    """Chance the answer leaves few enough candidates to commit. Higher is better.

    Minimising the expected surviving count and maximising this are *different*
    objectives, and which is right depends on what the agent does with the
    answer. Expected count prefers an even split. With a confidence gate in
    front of the recommendations, what actually pays is landing under the gate's
    threshold -- which prefers a question that isolates candidates outright,
    even if the rest of the mass stays lumped together.

    Reading: the fraction of candidates that would land in a group small enough
    to commit to a full list for.
    """
    groups = _partition(index, candidates, attribute, disclosed)
    total = sum(groups.values())
    if not total:
        return 0.0
    return sum(size for size in groups.values() if size <= target) / total


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
    # `disclosed` mirrors the simulator exactly, but only while turns parse: it
    # excludes mined guesses and the intent-override opening, neither of which
    # the simulator considers revealed. Once we stop parsing it stops filling,
    # so fall back to `seen` rather than conclude nothing has been said.
    source = config.eig_disclosed_source
    if source == "seen" or (source == "adaptive" and state.parse_failures):
        disclosed = state.seen
    else:
        disclosed = state.disclosed

    converging = config.question_objective == "convergence"
    best_attribute, best_score = None, None
    for attribute in options:
        if converging:
            # Maximise the chance of committing next turn, breaking ties on the
            # expected surviving count -- which still separates two questions
            # that would converge equally often.
            odds = convergence_odds(
                index, sample, attribute, disclosed, config.gate_candidate_threshold
            )
            penalty = 0.05 if attribute in state.asked else 0.0
            score = (-(odds - penalty), expected_remaining(index, sample, attribute, disclosed))
        else:
            remaining = expected_remaining(index, sample, attribute, disclosed)
            # Prefer a question we have not already spent a turn on.
            if attribute in state.asked:
                remaining += 0.5
            score = (remaining,)
        if best_score is None or score < best_score:
            best_attribute, best_score = attribute, score

    return best_attribute


def phrase(attribute: str | None, candidate_count: int) -> str:
    if attribute is None:
        return "Here are the closest matches I found."
    question = QUESTION_TEMPLATES.get(attribute, QUESTION_TEMPLATES["other"])
    if candidate_count > 50:
        return f"I have a broad set of options so far. {question}"
    return f"Here are my closest matches. {question}"
