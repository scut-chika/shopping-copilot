"""Multi-route retrieval and tiered fusion.

Routes, in descending order of trust:

1. **Card-exact** - the constraint string is a verbatim entry of a product's
   reconstructed intent card.  Near-deterministic when it fires.
2. **Loose** - the constraint appears as a raw feature/detail string.  Catches
   products whose card ordering differs from what the customer disclosed.
3. **Category** - the coarse category peeled off the opening message.
4. **BM25** - FTS5 over the accumulated dialog.  This is the route that keeps
   working if the organizer paraphrases the customer's wording, so it is always
   scored even when the exact routes fire.

Fusion is additive with a large per-constraint weight, which makes the ordering
lexicographic in "number of constraints matched" while still letting BM25, the
popularity prior, and the profile break ties inside a tier.
"""

from __future__ import annotations

from collections import defaultdict

from .config import Config
from .dialog import SessionState
from .replay import demote


def _add(scores: dict[str, float], asins, amount: float) -> None:
    for asin in asins:
        scores[asin] += amount


def score_candidates(index, state: SessionState, config: Config) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)

    matched_constraints = 0
    if config.use_card_index or config.use_loose_index:
        for constraint in state.constraints:
            weight = config.tier_weight + (config.emphasis_bonus if constraint.emphasized else 0.0)
            if constraint.mined:
                # Mined constraints are inferred, not stated: trust them less.
                weight *= config.mined_weight_factor

            postings = index.postings(constraint.text) if config.use_card_index else None
            if postings and len(postings) <= config.max_posting_list:
                _add(scores, postings, weight)
                matched_constraints += 1
                continue

            loose = index.loose_postings(constraint.text) if config.use_loose_index else None
            if loose and len(loose) <= config.max_posting_list:
                _add(scores, loose, weight * config.loose_match_weight)
                matched_constraints += 1

    if config.use_category_filter and state.category:
        _add(scores, index.in_category(state.category), config.category_weight)

    if config.use_bm25:
        query = state.dialog_text()
        ranked = index.bm25(query, config.bm25_pool)
        span = max(len(ranked), 1)
        for position, asin in enumerate(ranked):
            scores[asin] += config.bm25_weight * (1.0 - position / span)

    if not scores:
        for asin in index.bm25(state.dialog_text(), config.candidate_pool):
            scores[asin] += config.bm25_weight

    return scores


def retrieve(index, state: SessionState, config: Config, top_k: int):
    """Score once, then derive both outputs a turn needs.

    `rank` and `candidate_set` previously each recomputed the scores, which
    doubled per-turn latency for no benefit.
    """
    scores = score_candidates(index, state, config)
    demote(index, state, config, scores)
    return (
        _top_k(index, state, config, scores, top_k),
        _consistent(config, scores, config.eig_max_candidates),
    )


def rank(index, state: SessionState, config: Config, top_k: int) -> list[str]:
    scores = score_candidates(index, state, config)
    demote(index, state, config, scores)
    return _top_k(index, state, config, scores, top_k)


def _top_k(index, state: SessionState, config: Config, scores, top_k: int) -> list[str]:
    if not scores:
        return []

    pool = sorted(scores.items(), key=lambda kv: -kv[1])[: config.candidate_pool]

    tags = state.preference_tags if config.use_profile else []
    adjusted: list[tuple[float, str]] = []
    for asin, base in pool:
        total = base
        if config.use_prior:
            total += config.prior_weight * index.prior.get(asin, 0.0)
        if tags:
            total += config.profile_weight * index.profile_overlap(asin, tags)
        adjusted.append((total, asin))

    adjusted.sort(key=lambda item: (-item[0], item[1]))
    return [asin for _, asin in adjusted[:top_k]]


def candidate_set(index, state: SessionState, config: Config, limit: int) -> list[str]:
    """The working candidate set the question estimator reasons over."""
    scores = score_candidates(index, state, config)
    demote(index, state, config, scores)
    return _consistent(config, scores, limit)


def _consistent(config: Config, scores, limit: int) -> list[str]:
    if not scores:
        return []
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    best = ordered[0][1]
    # Keep only candidates still plausibly consistent with everything disclosed.
    threshold = best - config.tier_weight * 0.5
    return [asin for asin, value in ordered[:limit] if value >= threshold]
