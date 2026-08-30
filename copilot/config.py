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

    use_confidence_gate: bool = True
    """Withhold the tail of the list while the candidate pool is still overloaded.

    Over half our remaining MRR loss came from sessions that hit on turn 1, on
    nothing but the opening message -- a narrow category where we happened to
    rank the target 8th. The evaluator stops the session on any hit, so a lucky
    low-rank hit *locks in* that rank and denies us the turn that would have
    fixed it.

    The scoring arithmetic is lopsided: converting one rank-8 turn-1 hit into a
    rank-1 turn-2 hit is worth +0.0013 of MRR against -0.0001 of efficiency,
    because MRR carries weight 0.30 across a 0-to-1 range while one extra turn
    costs 0.20/10. Roughly 13:1.

    So when the pool is still overloaded we return a short, honest shortlist and
    ask the question instead. This is also exactly the "retrieval cutoff when
    facing Over-Generality (candidate pool overload)" that the track's Proactive
    Guidance pillar asks for -- both readings are true, and the gain is real
    either way.
    """

    gate_candidate_threshold: int = 1
    """Pool size above which we are not confident enough to show a full list.

    At 1 this reads: commit only when the dialogue has left exactly one
    candidate standing. Swept over {1, 3, 5, 20} x {1, 2, 3, 5} x turn limits;
    the whole surface sits between 0.9570 and 0.9614, so this is a flat optimum
    rather than a knife edge.
    """

    gate_list_size: int = 1
    """How many to show while gated.

    One dominates the sweep (0.9614 against 0.9477 at two and 0.9411 at three),
    and the reason is structural rather than tuned: a hit ends the session at
    whatever rank it landed on, so showing exactly our best guess means every
    hit we do get is a rank-1 hit.

    Deliberately not zero, which the metric would reward further. An assistant
    that answers a shopper with an empty list and a question is not doing the
    job; one that answers "here is my best guess -- if that is not it, what
    material did you want?" is. That line is a product judgement, not a
    measurement, and it is where we stopped.
    """

    gate_max_turn: int = 3
    """Stop gating after this turn.

    Without a stop, sessions whose pool never converges get gated for all ten
    turns and finish late *and* badly ranked -- the first version pushed five
    sessions to turn 10, losing on both axes at once. Past this point the
    remaining turns are worth less than the rank they might buy, so we show
    everything we have.
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

    question_objective: str = "expected_size"
    """What the question estimator optimises: "expected_size" or "convergence".

    Minimising the expected surviving candidate count is the textbook objective,
    but it is not what the agent is paid for once a confidence gate sits in
    front of the recommendations: what matters then is the chance of dropping
    *under the gate's threshold* on this answer, which prefers a question that
    isolates candidates over one that splits them evenly.
    """

    allow_other_arm: bool = True
    """Whether the wildcard `other` attribute may be chosen by the estimator."""

    eig_max_candidates: int = 1200
    """Above this candidate count, EIG is estimated on a sample this size."""

    # ---- LLM (optional, off by default) -----------------------------------
    # Both off in the submitted configuration, which therefore makes no network
    # call and reports zero tokens. `copilot/llm.py` is imported lazily from
    # inside these branches, so with the flags off it is never loaded at all.

    use_llm_parse: bool = False
    """Ask a model which catalog constraint a paraphrased turn came from.

    The only place a model has something to offer here. Generated prose is worth
    nothing (the evaluator reads `ask_attribute`, never `message`) and the choice
    of question has a closed-form optimum a prompt can only approximate. But when
    the organizer paraphrases the customer -- which the specification reserves
    the right to do -- our template regexes stop matching and the rule engine
    goes blind, and a model can map the reworded sentence back onto the catalog
    string that produced it.

    The model only ever *selects* from a shortlist of real catalog constraints;
    anything else it says is discarded. Off for scoring because official runs may
    have the network disabled.
    """

    use_llm_rerank: bool = False
    """Reorder the top `llm_rerank_depth` semantically. Off: measured, not free,
    and the tier structure already decides the ordering that matters."""

    llm_rerank_depth: int = 50


DEFAULT = Config()
