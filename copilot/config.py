"""Tunable configuration for the Shopping Copilot agent.

Every retrieval route and strategy is behind a flag so that `tools/ablation.py`
can turn one thing off at a time and measure its contribution.  The defaults
here are the configuration we submit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- retrieval routes -------------------------------------------------
    use_card_index: bool = True
    """Route 1: exact match against reconstructed intent-card constraints."""

    use_loose_index: bool = False
    """Route 2: constraint text matched against raw feature/detail strings.

    Off by default because it was measured to contribute *nothing*: identical
    scores to four decimal places on the clean public set and at every one of the
    five paraphrase levels, while costing ~62 MB of heap and time to build. It
    was a plausible fallback for constraints whose card ordering differs from
    what the customer disclosed; that case appears not to occur. Kept behind the
    flag rather than deleted so the finding stays reproducible.
    """

    use_category_filter: bool = True
    """Route 3: coarse-category narrowing from the opening message."""

    use_bm25: bool = True
    """Route 4: FTS5 BM25 over accumulated dialog text (paraphrase fallback)."""

    use_constraint_mining: bool = True
    """Recover constraints by token overlap when template parsing misses them.

    Template parsing is exact but brittle: the robustness harness showed that
    rewording the templates alone (constraint text untouched) cost more score
    than heavily paraphrasing the constraints themselves.  Mining removes that
    dependence on the organizer's exact phrasing.
    """

    mining_only_when_parse_fails: bool = True
    """Mine only when template parsing recovered nothing this turn.

    Mining is a salvage path, not a second opinion: on well-formed turns the
    template parse is exact, so letting mining also fire there only adds false
    positives.  Gating it keeps clean-set accuracy while retaining the
    paraphrase safety net.
    """

    mining_min_overlap: float = 0.75
    """Fraction of a constraint's content tokens that must appear in the turn."""

    mining_min_tokens: int = 2
    """Minimum constraint length to be mineable.

    Kept at the floor: raising it to 4 was measured and *cost* score (L1 0.826
    -> 0.796), because it excluded genuine short constraints.  The problem it
    was meant to solve -- short generic strings like "Machine Wash" hitting
    ratio 1.0 and crowding out the long specific constraint -- is instead fixed
    by ranking on absolute matched tokens rather than ratio.
    """

    mining_candidates: int = 60

    mining_max_results: int = 16
    """How many mined constraints to admit per turn.

    Raised from 4 after measuring what mining actually does. Its precision at
    recovering the *target's own* constraints is only ~0.27, and tripling its
    recall (candidate pool 60 -> 300) changed the end score by nothing. What it
    really provides is an aggregate boost to products whose card text overlaps
    the turn, so the useful quantity is total evidence, not the single best
    guess. Widening to 16 gained +0.017 to +0.038 at every paraphrase level;
    32 was indistinguishable from 16, so this is the knee.
    """

    mined_weight_factor: float = 0.35
    """Mined constraints are weighted below parsed ones: they are usually wrong
    individually and only informative in aggregate. Lowering this from 0.6
    removed the clean-set cost of mining entirely (L0 back to 0.9189) while
    keeping the paraphrase gains."""

    use_replay_consistency: bool = True
    """Replay the question/answer history against each candidate.

    Membership scoring only asks whether a product's card *contains* what was
    disclosed. The simulator's replies are deterministic, so we can ask the
    sharper question -- would this product have given exactly this answer? --
    which also uses what the customer did *not* say. See `copilot.replay`.
    """

    replay_penalty: float = 600.0
    """Score charged per answer a candidate could not have produced.

    Above `tier_weight * 0.5`, so one inconsistency also drops the candidate out
    of the working set `_consistent` hands to the question estimator.
    """

    replay_pool: int = 1500
    """How many top-scoring candidates to replay. >= `eig_max_candidates` so the
    estimator never reasons over candidates that were never checked."""

    replay_hard_filter: bool = False
    """Delete inconsistent candidates instead of demoting them.

    Off: this check leans harder on simulator determinism than anything else we
    do, and the private simulator is not ours to inspect. A penalty costs rank
    if it is ever wrong; a filter deletes the right answer.
    """

    use_prior: bool = True
    """Popularity/quality prior used to break ties inside a tier."""

    use_profile: bool = False
    """Anonymized user_profile preference tags as a soft ranking signal.

    Off by default because the ablation measured it as a *net negative*
    (+0.0131 TechnicalScore when disabled): the tags are coarse ("fit",
    "comfort", "durability") and match almost every clothing item, so they add
    noise to tie-breaking rather than signal.  Kept behind a flag rather than
    deleted so the finding stays reproducible.
    """

    # ---- fusion weights ---------------------------------------------------
    tier_weight: float = 1000.0
    """Weight per matched constraint. Large, so constraint count dominates."""

    loose_match_weight: float = 0.5
    """A loose (raw-text) constraint match counts less than a card match."""

    category_weight: float = 400.0
    bm25_weight: float = 30.0
    prior_weight: float = 8.0
    profile_weight: float = 4.0

    emphasis_bonus: float = 250.0
    """Extra weight for a constraint the customer explicitly re-stated."""

    retain_products: bool = False
    """Keep the raw 50k product dicts in memory after indexing.

    Off by default. Nothing in the agent's request path reads them -- they are
    needed only while building the indexes -- and retaining them cost 173 MB,
    about 55% of the process footprint. Tools that need raw catalog fields load
    the catalog themselves; set this to True only for interactive debugging.
    """

    # ---- candidate generation --------------------------------------------
    max_posting_list: int = 20_000
    """Constraint postings larger than this are too common to intersect on."""

    candidate_pool: int = 400
    """How many scored candidates to carry into reranking."""

    bm25_pool: int = 300

    # ---- question strategy ------------------------------------------------
    question_strategy: str = "eig"
    """One of: "eig" (expected information gain), "other", "cycle", "none"."""

    allow_other_arm: bool = True
    """Whether the wildcard `other` attribute may be chosen by the estimator."""

    eig_max_candidates: int = 1200
    """Above this candidate count, EIG is estimated on a sample this size."""

    # ---- LLM (optional, off by default) -----------------------------------
    use_llm_rerank: bool = False
    """Off by default: the system is designed to score well at zero API cost."""

    llm_rerank_depth: int = 50


DEFAULT = Config()
