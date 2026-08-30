"""Replaying the dialogue against a candidate, rather than only testing membership.

The rest of retrieval asks a *set* question: does this product's reconstructed
intent card **contain** the constraint the customer disclosed?  That throws away
most of what the transcript actually tells us, because the simulator's replies
are deterministic.  Given `ask_attribute` it returns the first **two undisclosed**
card constraints whose classification matches, in card order, and says "I don't
have an additional preference" when there are none.

So for any candidate we can ask a much sharper question: *if the target were this
product, would it have said exactly what we heard?*  Two kinds of evidence fall
out of that, both of which pure membership ignores:

* **It would have said more.**  We asked for `material` and heard "cotton".  A
  product whose card is ``[cotton, silk lining, ...]`` -- both `material` -- would
  have answered "cotton; silk lining".  It cannot be the target, yet membership
  scores it a full tier.
* **It said there was nothing more.**  "I don't have an additional preference for
  color" means the target has *no* undisclosed `color` constraint, which rules
  out every candidate that still has one.

Measured over the public set this cuts the working candidate set from ~632 to
~18 while keeping the true target in every single turn-state.

The check is applied as a **penalty, not a filter** (see `demote`).  It leans
harder on the simulator being deterministic than anything else we do, and the
private simulator is not ours to inspect: if it ever differs, a penalty costs us
rank while a filter would delete the right answer outright.
"""

from __future__ import annotations

from .config import Config


def predicted_answer(index, asin: str, attribute: str, disclosed) -> tuple[str, ...]:
    """What the simulator would reveal for `attribute` if `asin` were the target.

    Mirrors `customer_reply` in the official evaluator: first two undisclosed
    constraints matching the attribute, in card order.  `card_constraints` is
    ordered hard-then-soft, which is the order the evaluator iterates, and that
    correspondence is pinned by `test_mirrors_official_derivation`.
    """
    matches: list[str] = []
    for constraint in index.cards.get(asin, ()):
        if constraint in disclosed:
            continue
        if attribute != "other" and index.attribute_of(constraint) != attribute:
            continue
        matches.append(constraint)
        if len(matches) == 2:
            break
    return tuple(matches)


def mismatches(index, evidence, asin: str) -> int:
    """How many recorded answers this candidate could not have produced."""
    total = 0
    for item in evidence:
        if predicted_answer(index, asin, item.attribute, item.disclosed_before) != item.revealed:
            total += 1
    return total


def demote(index, state, config: Config, scores: dict[str, float]) -> None:
    """Penalise replay-inconsistent candidates, in place.

    Only the top `replay_pool` scores are examined: anything below that is
    already far out of contention, and bounding the work keeps per-turn latency
    flat as the score map grows with the category posting list.
    """
    if not config.use_replay_consistency or not scores:
        return
    evidence = getattr(state, "evidence", ())
    if not evidence:
        return

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[: config.replay_pool]
    inconsistent = [(asin, n) for asin, _ in ordered if (n := mismatches(index, evidence, asin))]
    if not inconsistent:
        return

    if config.replay_hard_filter:
        # Refuse to empty the pool: if the evidence excludes everything we were
        # looking at, the evidence is what is wrong, not the whole catalog.
        if len(inconsistent) < len(ordered):
            for asin, _ in inconsistent:
                del scores[asin]
            return

    for asin, count in inconsistent:
        scores[asin] -= config.replay_penalty * count
