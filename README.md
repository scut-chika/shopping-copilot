# Shopping Copilot — TikTok TechJam 2026, Track 4

A multi-turn shopping agent for the Conversational E-Commerce Search challenge.
It finds a customer's hidden target product in a 50,000-item Amazon catalog in
about two turns, and two ideas carry almost all of that:

1. **Ask the question most likely to leave exactly one candidate.** The
   simulator's reply rule is public and deterministic, so the value of a question
   is computable rather than guessable. Removing this costs **−0.3318**.
2. **Refuse to answer until it has one.** A hit ends the session at whatever rank
   it landed on, so converting while still guessing books a bad rank forever.
   Until the dialogue leaves a single candidate the agent returns its one best
   guess plus the question, not ten it knows are wrong. Removing this costs
   **−0.0578** — more than every retrieval route put together.

**TechnicalScore 0.9717** on the official evaluator, against 0.1067 for the
organizer's baseline. Our own estimate for the private set is **0.930**, and the
[held-out section](#does-it-generalize-or-did-we-fit-200-sessions) explains why
we quote that number rather than the public one.

**The scored configuration uses no LLM, makes no network call, has no
third-party dependencies, and runs on CPU for $0.00.** An LLM stage exists and is
switched off; see [The optional LLM stage](#the-optional-llm-stage-and-why-it-is-off).

Repository: <https://github.com/scut-chika/shopping-copilot>

The submission rules note that *"for official final scoring, organizer policy may
disable network access"*. This agent is built so that restriction changes nothing
about how it runs — a property pinned by a test, not just claimed.

## Results

Official evaluator, run unmodified (`python -m evaluator.local_evaluator`), 200 public sessions:

| | Hit Rate@10 | MRR | MTTC | Efficiency | **TechnicalScore** |
|---|---|---|---|---|---|
| Organizer BM25 baseline | 0.125 | 0.0680 | 9.81 | 0.119 | **0.1067** |
| **Shopping Copilot** | **1.000** | **0.9790** | **2.10** | **0.8900** | **0.9717** |

The target is ranked **first in 194 of 200 sessions**, and never below tenth.

By scenario — token usage **0 prompt, 0 completion**:

| Scenario | n | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 1.000 | 0.969 | 1.85 |
| browsing | 80 | 1.000 | 0.954 | 2.21 |
| intent_override | 30 | 1.000 | 0.949 | 3.63 |
| boundary | 10 | 1.000 | 1.000 | 2.70 |

`intent_override` cannot convert before turn 3 by construction, and `boundary`
spends a turn on a question the customer declines, so those MTTC floors are
structural rather than modelling failures.

> The organizer clarified on 27 Aug that *"TechnicalScore is an objective input to
> the Technical Execution assessment. It is not a separate judging criterion and
> does not represent the entire Technical Execution score."* We read that as: past
> a point, pushing this number is not where the value is. The work after we
> reached 0.916 went into evidence that it means something — generalization,
> ablation, robustness — rather than into the number itself.

## Does it generalize, or did we fit 200 sessions?

A 1.000 hit rate on the public set is a warning sign: at the ceiling, the public
set can no longer tell us anything, and any further tuning fits those 200
sessions. The private set is 800 sessions built from **different users and
different target products**, and we cannot see it.

So we built a stand-in. The organizer's session generator is deterministic and
public, so `tools/generalize.py` samples target products that appear **nowhere in
the public set**, synthesizes user profiles by resampling profile fields
independently, and replays the official protocol at the official 40/40/15/5
scenario mix.

| Set | n | Score | Hit Rate@10 | MRR | MTTC | Retained |
|---|---|---|---|---|---|---|
| public | 200 | 0.9717 | 1.000 | 0.979 | 2.10 | — |
| held-out, seed 20260830 | 800 | **0.9307** | 0.979 | 0.916 | 2.67 | **95.8%** |
| held-out, seed 7 | 800 | **0.9298** | 0.975 | 0.922 | 2.71 | **95.7%** |

Two independent draws of 800 unseen targets. Reproduce with
`python tools/generalize.py --sessions 800 [--seed N]`.

**If you want one number for how this is likely to do on the private set, it is
0.930, not 0.972.** We would rather say that than quote the public figure and let
it be read as a forecast.

This is the check that mattered most, because the mechanisms below read the
simulator's behaviour closely enough that overfitting was the obvious risk. The
held-out score moved **0.8815 → 0.9303** (mean of the two seeds) as they were
added, so roughly 92% of the public-set gain is real.

The exception is worth naming. Re-tuning `bm25_weight` was worth +0.0094 on the
public set and **exactly nothing** on held-out targets — held-out sits at 0.930
whether the weight is 5, 10, or 30. That change is kept because it costs nothing
held-out and gains 0.017–0.062 at every paraphrase level, but the public-set
portion of its gain is fitted to those 200 sessions and we do not count it.

This isolates one axis — unseen targets. It deliberately holds the *generator*
fixed, so it says nothing about organizer paraphrasing; that is measured
separately below.

## The insight this is built on

The obvious framing is semantic search: embed the query, embed the catalog, rank
by similarity. Reading the organizer's evaluator shows the task is shaped
differently.

`evaluator/local_evaluator.py` builds every customer utterance deterministically
from the **target product's own catalog record**. `intent_card()` flattens the
product's `features` and `details`, prepends a regex-matched material and colour,
appends a price line, and keeps the first four strings. Those four strings are
the entire vocabulary of the conversation.

Three consequences drive the design.

### 1. The customer speaks in verbatim catalog text

So the primary retrieval route is not a vector index — it is an inverted index
from constraint string to the products that could have produced it. Reproducing
the same derivation across all 50,000 products (`copilot/simulator_model.py`)
inverts the generator. Measured on the public set:

| Constraints disclosed | Median candidates remaining | Sessions with ≤10 |
|---|---|---|
| 0 (category only) | 184 | 3.0% |
| 1 | 29 | 29.0% |
| 2 | 1 | 76.0% |
| 3 | 1 | 94.5% |
| 4 | 1 | 99.0% |

Two answered questions is usually enough to guarantee a top-10 hit. That is why
the agent converges in ~2 turns instead of spending its budget of 10.
(`tools/explore.py`, `tools/explore2.py` reproduce this table.)

### 2. Question value is computable, not guessable

The simulator picks what to reveal with a public rule (`classify_constraint`), so
for any candidate set we can ask, for each attribute, *"if the target were product
X, what would this question return?"*, group candidates by predicted answer, and
choose the question that minimises the expected surviving group size.
`copilot/questions.py` implements this as genuine expected-information-gain
estimation rather than a fixed script.

**This is where most of the score comes from.** Disabling questions costs
**−0.4797** — more than every retrieval route combined. The headline result of
this project is not the exact-matching trick; it is that *asking the right
question is worth more than retrieving better*.

### 3. "Intent Override" does not change the target

The problem statement asks for *slot erasure and rewriting*. The evaluator tells
a different story:

```python
old_value = soft[-1]    # from the target's intent card
new_value = hard[0]     # from the same target's intent card
```

Both come from the same product, and `parent_asin` never changes. Implementing
literal slot erasure would discard valid evidence and lose score. The agent
**accumulates constraints and re-weights** the restated one instead
(`emphasis_bonus`). This is deliberate, and pinned by a test.

A second consequence: `override_applied` gates hit detection, so an override
session cannot convert before turn 3 no matter what. The agent still returns a
full list on turns 1–2 because it costs nothing; it simply cannot score there.

### 4. Membership is the weak question; replay is the strong one

Every retrieval route above asks a *set* question: does this product's intent
card **contain** what the customer disclosed? That discards most of what the
transcript says, because the simulator's answers are deterministic. Given
`ask_attribute` it returns the first **two undisclosed** matching constraints, in
card order — so for any candidate we can ask the far sharper question: *if the
target were this product, would it have said exactly what we heard?*

Two kinds of evidence fall out, and membership sees neither:

- **It would have said more.** We asked for `material` and heard `"cotton"`. A
  product whose card is `[cotton, silk lining, …]` — both `material` — would have
  answered `"cotton; silk lining"`. It cannot be the target, yet membership gives
  it a full tier.
- **It said there was nothing more.** `"I don't have an additional preference for
  color"` means the target has *no* undisclosed `color` constraint, ruling out
  every candidate that still has one. We previously used this only to stop asking
  about `color`.

The working candidate set falls from **~632 to ~18 (−97.1%)**, with the true
target surviving **149/149** turn-states. `tests/test_copilot.py` replays the
whole public set and fails if the target is ever ruled out.

**The trap.** The simulator has two refusal phrasings that mean opposite things:

```
boundary : "I don't have a preference for {a}; please use your judgment."
           -> fired once per boundary session whatever the card holds. No evidence.
no-extra : "I don't have an additional preference for {a}."
           -> the target really has no undisclosed constraint of that class.
```

Conflating them ruled out the true target in 8% of turn-states, and looked like
the whole idea was broken. They are now handled separately, and `state.disclosed`
is an explicit mirror of the simulator's own set rather than `seen` — which also
holds mined guesses and the intent-override opening value, neither of which was
ever actually disclosed.

Applied as a **penalty, not a filter**. This leans on simulator determinism
harder than anything else here and the private simulator is not ours to inspect;
if it differs, a penalty costs us rank while a filter would delete the right
answer. `replay_hard_filter` keeps the alternative measurable.

### 5. A hit locks in whatever rank it landed on

The evaluator ends the session the moment the target appears in the list. So a
*lucky* hit — turn 1, off nothing but the opening message, target eighth in a
narrow category — is not a win. It permanently books rank 8 and denies us the
turn that would have made it rank 1. Over half our remaining MRR loss was exactly
this.

The arithmetic is lopsided:

```
hit at turn 1, rank 8  ->  0.30 x 0.125  =  0.0375   (per-session, before /N)
hit at turn 2, rank 1  ->  0.30 x 1.000  =  0.3000
extra turn costs       ->  0.20 x 1/10   =  0.0200
```

Deferring wins whenever the rank we would have booked is 2 or worse. So while the
dialogue has not yet left exactly one candidate standing, the agent returns **its
single best guess plus the question**, rather than padding out ten it knows are
wrong. Swept over thresholds x list sizes x turn limits, the whole surface lies
between 0.9570 and 0.9614 — a flat optimum, which matters more than the peak.

Read the other way, this is the *"retrieval cutoff when facing Over-Generality
(candidate pool overload)"* that the track's Proactive Guidance pillar asks for.
Both readings are true and the gain is measured either way; we would rather state
the scoring arithmetic than dress it up.

**Where we stopped.** Returning *zero* items would score higher still. We did not
ship that: an assistant that answers a shopper with an empty list is not doing
the job, and "here is my best guess — if that is not it, what material did you
want?" is. That line is a product judgement, not a measurement, and it is the
only place in this project where we left score on the table on purpose.

### 6. The objective the estimator optimises is not the obvious one

Insight 2 minimises the expected surviving candidate count. Once insight 5 sits
in front of the recommendations, that is no longer what the agent is paid for:
the gate cannot commit until one candidate is left, so what matters is the chance
of *reaching* that state, not the average size of what remains. The two genuinely
disagree — expected size prefers an even split, convergence prefers isolating
candidates outright even if the rest stays lumped together. Switching objectives
was worth +0.0009, and `question_objective` keeps both measurable.

## Architecture

```
reset(session_id, profile)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Offline, once at startup  (copilot/catalog.py, ~17s)        │
│   card index      constraint string → products              │
│   loose index     raw feature/detail string → products      │
│   category index  coarse category → products                │
│   product FTS5    BM25 over title/features/description      │
│   constraint FTS5 BM25 over the 60,670 constraint strings   │
│   priors          rating × log(rating_count)                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼   each turn: respond(message, turn, top_k)
┌─────────────────────────────────────────────────────────────┐
│ 1. Parse            copilot/dialog.py                       │
│    Template match → constraints. If that recovers nothing,  │
│    mine constraints by token overlap instead, so reworded   │
│    phrasing still yields structure.                         │
├─────────────────────────────────────────────────────────────┤
│ 2. Accumulate       constraints never erased, only weighted │
├─────────────────────────────────────────────────────────────┤
│ 3. Retrieve         copilot/retrieval.py                    │
│    R1 card-exact  ─┐                                        │
│    R2 loose match  ├─ additive fusion with a large          │
│    R3 category     │  per-constraint weight, so ordering is │
│    R4 BM25        ─┘  lexicographic in "constraints matched"│
│                       and prior/BM25 break ties inside it   │
├─────────────────────────────────────────────────────────────┤
│ 4. Replay           copilot/replay.py                       │
│    Demote candidates that would have answered our earlier   │
│    questions differently -- including "it would have said   │
│    more" and "it said there was nothing more".              │
├─────────────────────────────────────────────────────────────┤
│ 5. Ask              copilot/questions.py                    │
│    argmax over attributes of P(one candidate left)          │
├─────────────────────────────────────────────────────────────┤
│ 6. Commit or defer  copilot/agent.py                        │
│    One candidate left  -> full ranked ten.                  │
│    Still ambiguous     -> best guess only, plus the         │
│                           question. A hit books its rank    │
│                           permanently, so converting while  │
│                           guessing is a loss, not a win.    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
{message, ask_attribute, recommendations[1 or 10]}
```

### Two invariants worth calling out

- **Always return a full ranked list.** The evaluator checks for a hit *before*
  generating the customer's reply, and recommending costs nothing, so there is
  never a reason to answer with a question alone. The organizer's baseline leaves
  this on the table — it never sets `ask_attribute`, and its MTTC is 9.81.
- **Never raise.** An exception becomes an empty turn, wasting one of ten. Every
  turn is wrapped, falling back to the previous answer, then to a popularity list.
  `test_never_raises_on_hostile_input` covers empty, oversized, binary, and
  FTS5-injection inputs.

## Against the four pillars

The problem statement asks for four things by name. Two we do well, one we do
differently on purpose, and one we largely do not do. Setting that out ourselves
seems better than leaving a reader to find the gaps.

### I. Intent routing and a hybrid pipeline

**Dual-track routing — done, but on the evidence rather than the label.** The
scenario is detected on turn 1 (`dialog.py`), and for a long time nothing read it
again: a buying session discloses a hard constraint immediately and a browsing
one does not, and both retrieval and the confidence gate already respond to that
difference. The obvious objection is that the label must carry *something*
extra, so we tested the strongest case for it. An intent-override session cannot
convert until the override lands on turn 3 or 4, so its efficiency floor is
already paid and holding back longer ought to be nearly free — and it is our
weakest track by MRR. Extending the gate for that track alone, across horizons 3
to 8, left **MRR identical to four decimal places** while MTTC crept upward.
Those sessions have already collapsed to one candidate by the time the override
lands. The label carries no information the evidence had not already delivered,
and `use_scenario_routing` keeps that measurable.

**Multi-route retrieval — three of the four named routes.** Card-exact, coarse
category, and BM25 over the accumulated dialog, fused additively (a fourth,
loose text matching, is implemented and measured to contribute nothing).

**Vector similarity — not implemented.** The honest gap. Two reasons, one good
and one merely convenient. The good one: the ablation says retrieval is not the
bottleneck — reducing it to plain BM25 costs **0.0076**, against **0.3318** for
removing the question policy. Adding a dense route would buy from the part of the
system that is already not the constraint. The convenient one: it would cost the
zero-dependency, offline-by-construction property that the network-restriction
rule makes valuable. We would test it first if we had another day, and we would
expect it to matter under paraphrase rather than on clean text.

**LLM semantic ranking — implemented, measured, switched off.** See
[the LLM section](#the-optional-llm-stage-and-why-it-is-off) for what it does,
what it measured, and why it is not in the scored path.

### II. Dialog strategy

**Information accumulation — done.** Constraints accumulate across turns, and
every answer is also replayed against each candidate, so what the customer
*declined* to say narrows the field too.

**Intent override — deliberately not slot erasure.** The statement asks for
erasure and rewriting. The evaluator takes both the old and the new value from
the *same* product's intent card and never changes `parent_asin`, so erasing
would discard valid evidence about the target. We accumulate and re-weight, and
pin the decision with a test. If a private simulator genuinely switches targets
mid-session this is wrong, and the [limitations](#limitations-and-what-we-would-do-next)
say what we would build to detect it.

**Proactive guidance and retrieval cutoff — this is our largest component.**
"Trigger an immediate retrieval cutoff when facing Over-Generality (candidate
pool overload)" is exactly what the confidence gate does, and removing it costs
**−0.0578**, more than every retrieval route combined.

### III. Self-evolution and dynamic context programming

**This is our weakest pillar, and the weakness is deliberate in one place and
real in the other.**

*Deliberate:* personalized context distillation over the long-term profile is
implemented and off. The tags carry genuine signal — a target's overlap averages
0.371 against 0.237 for its category peers — but the target is already ranked
first in 194 of 200 sessions, so the signal has nowhere to go and costs 0.0052.

*Real:* adaptive orchestration exists but is modest. The runtime does re-route
itself — mining fires only when template parsing recovered nothing, the question
estimator switches which "already disclosed" set it reasons from when parses
start failing, the gate releases when the candidate set collapses or the turn
budget runs short — but these are conditional strategies, not the runtime
workflow re-orchestration the statement envisions.

### IV. Evaluation matrix

Coverage, precision, and efficiency are the three metrics, and the
[results](#results) and [held-out](#does-it-generalize-or-did-we-fit-200-sessions)
sections report all of them, by scenario, with the raw output committed under
`results/`.

## Ablation

`python tools/ablation.py` → `results/ablation.json`. One setting changed at a
time, official protocol, 200 sessions:

| Variant | Score | Δ | Hit | MRR | MTTC |
|---|---|---|---|---|---|
| **full system** | **0.9717** | — | 1.000 | 0.9790 | 2.10 |
| questions: none | 0.6399 | **−0.3318** | 0.710 | 0.4997 | 4.25 |
| − confidence gate | 0.9139 | **−0.0578** | 1.000 | 0.7594 | 1.70 |
| gate: show 3, not 1 | 0.9458 | −0.0259 | 1.000 | 0.8782 | 1.89 |
| − popularity prior | 0.9494 | −0.0223 | 1.000 | 0.9391 | 2.62 |
| retrieval: BM25 only | 0.9641 | −0.0076 | 0.995 | 0.9749 | 2.30 |
| − category filter | 0.9654 | −0.0064 | 0.995 | 0.9748 | 2.23 |
| + user profile *(default off)* | 0.9665 | −0.0052 | 1.000 | 0.9670 | 2.18 |
| questions: fixed cycle | 0.9669 | −0.0048 | 1.000 | 0.9687 | 2.19 |
| questions: EIG without `other` | 0.9676 | −0.0041 | 1.000 | 0.9705 | 2.18 |
| − card-exact index | 0.9711 | −0.0006 | 1.000 | 0.9790 | 2.13 |
| − replay consistency | 0.9712 | −0.0005 | 1.000 | 0.9790 | 2.13 |
| questions: always `other` | 0.9714 | −0.0003 | 1.000 | 0.9782 | 2.11 |
| mining ungated | 0.9716 | −0.0001 | 1.000 | 0.9790 | 2.11 |
| replay: hard filter *(default off)* | 0.9717 | ±0.0000 | 1.000 | 0.9790 | 2.10 |
| gate: stop after parse failure *(default off)* | 0.9717 | ±0.0000 | 1.000 | 0.9790 | 2.10 |
| estimator sees `seen` / `disclosed` | 0.9717 | ±0.0000 | 1.000 | 0.9790 | 2.10 |
| + loose index / + retain raw products *(default off)* | 0.9717 | ±0.0000 | 1.000 | 0.9790 | 2.10 |
| − constraint mining | 0.9718 | +0.0001 | 1.000 | 0.9790 | 2.10 |
| objective: expected size | 0.9718 | +0.0001 | 1.000 | 0.9790 | 2.10 |
| − BM25 route | 0.9736 | +0.0019 | 1.000 | 0.9800 | 2.02 |

**Asking is still the whole story, but the gate is now the second pillar.**
Removing the question policy costs −0.3318; removing the confidence gate costs
−0.0578; every retrieval route put together costs less than either.

**Three rows are uncomfortable and none of them are hidden.**

**`− card-exact index` costs 0.0006.** The exact inverted index was the original
insight of this project and, in the finished system, removing it is almost free.
The questions and the gate together do the work; the index only makes them
converge slightly sooner. It stays because it is what makes the candidate set
small enough for the question estimator to be cheap, and because it is the route
that does not depend on the gate.

**`− replay consistency` costs 0.0005, after being worth +0.0031 on its own.**
Replaying the dialogue against candidates — asking "would this product have
*said* that?" instead of "does it contain that?" — collapses the working
candidate set from ~632 to ~18 and was measured as a real gain before the
confidence gate existed. After the gate, it is inside the noise: the two overlap,
because the gate declines to convert until the set has collapsed and replay makes
it collapse sooner. That was the larger piece of work and it ended up nearly
redundant. It stays because it is what the generalization and robustness runs
were validated with, and because it is what would carry the system if the gate
ever had to come off.

**`objective: expected size` costs −0.0001, i.e. the convergence objective now
gains nothing.** It was worth +0.0009 when it was measured, and re-tuning
`bm25_weight` afterwards absorbed it. The earlier figure is not restated as if it
still held.

**`− BM25 route` still *gains* 0.0019**, down from +0.0113 before the weight was
re-tuned. Deleting the route would cost 0.065 at L3 and 0.068 at L4, so this is a
trade quantified in [Robustness](#robustness-to-paraphrasing) rather than settled
here.

Four flags change how the *index* is built rather than how a request is served,
so those variants rebuild the index instead of reusing a shared one. An earlier
version of the script did not, and silently reported 0.0000 for the user-profile
row; `tools/ablation.py` now marks each result with `rebuilt_index`.

Things we changed because of this table, not because they sounded good:

- **`user_profile` is off by default — and our published reason for that was
  wrong.** We claimed the anonymized tags "match nearly every clothing item, so
  they add noise rather than signal". Measured: a target's tag overlap averages
  **0.371 against 0.237 for its own category peers**, and the target beats its
  peers in **135 of 200** sessions. The tags carry real signal. It is off because
  the signal has nowhere to go — the target is already ranked first in 194 of 200
  sessions, so a cue that is right two times in three disturbs more correct
  rankings than it repairs. Restricting it to undecided tiers only recovers
  0.0009 of the 0.0052. The flag stays; the explanation is corrected.
- **Constraint mining is free here** (+0.0001, i.e. nothing) and worth **+0.14**
  under paraphrase — by far the largest gap between what a component looks like on
  clean data and what it is actually holding up. See
  [What constraint mining actually does](#what-constraint-mining-actually-does).
- **The loose index is off by default: it contributes literally nothing.**
  Identical scores to four decimal places on the clean set *and* at all five
  paraphrase levels, for ~62 MB of heap. It was a plausible fallback for
  constraints whose card ordering differs from what the customer disclosed; that
  case appears never to occur. Kept behind a flag so the finding is reproducible.
- **EIG barely beats always asking `other`** (+0.0013), because `other` is a
  wildcard matching any undisclosed constraint. We kept the estimator anyway: it
  is the mechanism that generalises, and with the wildcard removed entirely it
  still scores 0.9046, where a hardcoded `other` would have nothing to fall back on.

## Robustness to paraphrasing

The specification notes the organizer may add natural-language paraphrasing
(*"If natural-language paraphrasing is added by the organizer, it cannot decide
correctness"*). That is the most likely way a verbatim-matching route fails to
transfer, so we measured the exposure. `tools/robustness.py` replays the official
protocol while rewriting every customer utterance:

| Level | Perturbation | Shipped | Mining off | Previous version |
|---|---|---|---|---|
| L0 | official wording | **0.9717** | 0.9718 | 0.9189 |
| L1 | every template reworded, constraints verbatim | **0.9052** | 0.7657 | 0.8435 |
| L2 | L1 + surface edits (case, punctuation) | **0.9023** | 0.7655 | 0.8418 |
| L3 | L1 + 25% of constraint words dropped | **0.8755** | 0.7403 | 0.8172 |
| L4 | L1 + 40% dropped and word order shuffled | **0.7678** | 0.7151 | 0.7514 |

**The parser was the weak link, not the matching.** An early version lost more
score from rewording the templates alone (L0→L1, −0.166) than from heavily
paraphrasing the constraints inside them. That was not the failure mode we
expected. Constraint mining — recovering constraints by token overlap rather than
template match — is what closes it.

**Mining became far more load-bearing than it used to be, and we only found out
by re-running the A/B.** It is worth **+0.14** at L1 through L3 now, against
+0.017 to +0.039 when it was last measured, and it no longer loses at L4 — it
wins there by 0.053. The cause is a change made for an unrelated reason: cutting
`bm25_weight` from 30 to 5 removed the lexical route that had been quietly
covering for mining's absence. Two components that each looked marginal were
partly substituting for each other, and the ablation only shows that if you
re-measure the old A/B after changing the other one.

**L4 was a regression for most of a day, and the fix came from the ablation.**
The confidence gate costs 0.005 at L4 and the convergence objective about 0.004,
which for a while left L4 the one level below where it started. The row that
resolved it looked unrelated: at the old `bm25_weight` of 30, *removing* BM25
outright gained 0.0113 on clean text. The gate had changed what a fuzzy signal is
worth — when you show ten items a lexical nudge is nearly free, when you commit
to one a nudge toward the wrong product costs the session. Re-tuning the weight
to 5 rather than deleting the route gained at **every** level at once (see
[Ablation](#ablation)), and carried L4 past its old figure as well.

**A hypothesis this table killed.** The confidence gate looked dangerous under
paraphrase: withholding the list bets on one more question settling the session,
and a question whose answer we cannot read settles nothing. So we made the gate
self-limiting on parse health. Measurement said the opposite — switching it off
under paraphrase cost score at *every* level it was meant to help (L1 0.8482 →
0.8147, L3 0.8110 → 0.7875) and recovered only 0.005 at L4. Even on a half-read
transcript, committing to one guess beats padding out ten. The flag survives as
`gate_needs_clean_parse`, defaulted off, with the numbers in its docstring.

**A regression this table located.** L3/L4 fell unexpectedly when the new
mechanisms landed. Holding everything else fixed and swapping one thing at a time
found it: feeding the question estimator the simulator's exact `disclosed` set is
right while turns parse and wrong the moment they stop, because that set is only
filled by a successful parse. Under paraphrase it stays empty, the estimator
concludes nothing has been revealed, and it re-asks what the customer already
answered. Choosing per session on `parse_failures` recovered L3 0.8110 → 0.8522
and L4 0.7146 → 0.7392. The same sweep confirmed replay consistency is completely
*inert* under paraphrase — identical scores with it disabled — which is the
intended behaviour, since it only records evidence from turns it could read.

**Even the pessimistic case holds.** At L4 the system stays ~6.9× the organizer
baseline. The score does not rest on the exact-matching insight alone.

## What constraint mining actually does

Mining was our weakest component: it cost score on clean text and regressed badly
under heavy paraphrase. Fixing it took five hypotheses, four of which the data
killed. The reason it is written up in full is that the one that survived came
from a *null* result, and it changed what the component is for.

**H1 — the overlap threshold is too strict.** Measured: when the true constraint
is missed, its token overlap with the turn is **1.00** at the median *and* the
90th percentile. It was clearing the threshold comfortably. Rejected.

**H2 — the true constraint is being outranked.** Measured at the shipped
candidate pool of 60: **zero** cases lost to ranking, and when present the true
constraint ranks **median 1**. Rejected.

**H3 — the bottleneck is candidate recall.** Measured: **72%** of the time the
true constraint was not in the top-60 FTS candidates at all. BM25 length-
normalises, so an OR-query over 60,670 short constraint documents surfaces short
generic strings and buries the long specific one. Correct diagnosis — widening
the pool to 300 lifted recall from **0.28 to 0.88**.

**And it changed the end-to-end score by nothing** (≤0.002 at every level). That
null result is the interesting one: if tripling the rate at which we find the
*right* constraint does not move the score, then finding the right constraint is
not what mining is contributing.

**H4 — mining is just BM25 with extra steps**, since a mined constraint routes
message-similar products through the high-weight card index. Measured: raising
`bm25_weight` from 30 to 300 and 800 with mining off is **worse at every level**
(L1 0.7546 and 0.7261 against 0.8265). Rejected — mining reaches something BM25
does not.

**H5 — then index the card text directly**, one hop instead of two: BM25 over
each product's concatenated card strings, the only text a customer can quote.
Measured: **much worse** (top-10 0.107 against 0.233 for product BM25). Four card
strings per product is too little text for BM25 recall. Rejected.

**What survived.** Mining is not constraint *recovery*; its precision at finding
the target's own constraints is only ~0.27. It is an aggregate boost to products
whose card text overlaps the turn, and the two-stage path through the constraint
index reaches them where a direct card-text index cannot. If the signal is
aggregate, the useful quantity is total evidence rather than the single best
guess — so admit many weak constraints instead of few confident ones.

That prediction held, and has held up better since. Widening
`mining_max_results` from 4 to 16 and cutting `mined_weight_factor` from 0.6 to
0.35 gained at every level and removed the clean-set cost entirely. Measured at
the time:

| | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| before (4 results, weight 0.6) | 0.9160 | 0.8265 | 0.8205 | 0.7788 | 0.7078 |
| **after (16 results, weight 0.35)** | **0.9189** | **0.8435** | **0.8418** | **0.8172** | **0.7514** |
| gain | +0.0029 | +0.0170 | +0.0213 | +0.0384 | +0.0436 |

(Measured at the configuration of the time. The absolute figures are lower than
the current table above because the confidence gate did not exist yet; the
comparison between the two rows is the point, and it was taken in one sitting.)

32 results was indistinguishable from 16, so 16 is the knee. Per-turn latency was
unaffected, because mining only runs on turns where template parsing recovered
nothing.

Three earlier tuning attempts also failed and are kept in the git history rather
than quietly dropped: raising the overlap threshold to 0.9 made L3/L4 *worse*
(0.7268/0.6831), requiring 4+ tokens cost L1 0.03, and the candidate-pool
widening from H3 is not in the shipped config because it buys nothing.

## Cost, latency, and the offline guarantee

`python tools/profile_cost.py` and `python tools/memcheck.py` →
`results/cost_profile.json`, `results/memory_profile.json`

| | Shipped | Pre-optimization |
|---|---|---|
| Model | **none** | none |
| Network access required | **no** | no |
| API cost | **$0.00** | $0.00 |
| Token usage | 0 prompt / 0 completion | 0 / 0 |
| Index build (`memcheck.py`) | **16.7 s** | 26.5 s |
| Python heap (tracemalloc peak) | **50.4 MB** | 306.6 MB |
| Process RSS after build | **206 MB** | — |
| Per-turn latency, mean (`profile_cost.py`) | 66 ms | 66 ms |
| Per-turn latency, p95 / p99 | 117 / 149 ms | 127 / 154 ms |

Both columns were measured back to back in isolated processes on the same
machine, because our first attempt at this table was wrong: earlier readings of
"32.1 s startup" and "82 ms mean latency" came from runs taken hours apart under
different machine load, and comparing them overstated the build-time improvement
as 3.2x and invented a latency improvement that does not exist. Absolute
millisecond figures here will vary with the host; the *ratios within a column
pair* are what we stand behind.

The rules reserve the right to score "under CPU, memory, timeout, and network
restrictions", so we treated the resource envelope as a requirement rather than a
footnote. Three structures were costing 84% of the Python heap for no measured
benefit, and the ablation confirms removing each is worth exactly +-0.0000:

- the **raw 50k product dicts** (173 MB) - read only while building the indexes,
  never in the request path, so the catalog is now streamed and discarded;
- **`_profile_text`** (26 MB) - only read when `use_profile` is on, which the
  ablation had already shown to be a net negative;
- the **loose index** (62 MB) - measured to contribute nothing at any
  perturbation level.

Score at the time: **0.918872**, unchanged by all three. **Heap fell 6.1x and
build time 1.5x; per-turn latency did not move**, which in hindsight is the
expected result: the removed structures were never read during a turn, so
dropping them frees memory without shortening the request path.

The shipped column was re-measured after the confidence gate, replay consistency,
and the optional LLM module landed. None of them moved it: latency is flat within
noise and the heap is unchanged, because all three work on a candidate list that
was already in memory.

The gap between the 206 MB RSS and the 50.4 MB Python heap is the two in-memory
SQLite FTS5 indexes, which SQLite allocates in C where `tracemalloc` cannot see
them. RSS is the number that matters against a memory cap, and it is the one we
report.

The agent imports only `json`, `math`, `os`, `re`, `sqlite3`, `dataclasses`,
`pathlib`, and `collections`. `test_agent_imports_only_the_standard_library`
parses the AST of every shipped module and fails if a network client or
third-party package ever appears. The single environment read is the `COPILOT_*`
config-override mechanism; `test_agent_reads_no_secrets_from_environment` pins
that it touches nothing else.

## The optional LLM stage, and why it is off

The problem statement names *"Multi-Route Retrieval → **LLM Semantic Ranking**"*
as the pipeline base, so this section answers that directly rather than leaving
it to be inferred from a config flag.

`copilot/llm.py` is real, works, was measured against a live model, and is
disabled in the submitted configuration. Two capabilities, both behind flags:

- **`use_llm_parse`** — when template parsing fails, ask a model which catalog
  constraint the reworded sentence came from.
- **`use_llm_rerank`** — semantic reorder of the top 50.

It matters *which direction* the model is pointed. Generating the customer-facing
prose is worth exactly nothing: the specification states that `ask_attribute` is
what the simulator reads, *"instead of guessing from prose"*, so `message` never
touches the score. Choosing the question is worse than nothing: that has a
closed-form optimum (insight 2) which a prompt can only approximate. The one
place a model has something to offer is the **inbound** direction — mapping a
paraphrased utterance back onto the catalog string that produced it — which is
precisely the failure the robustness table above measures.

**Containment.** The model is a *selector*, never a source. It is handed a
shortlist drawn from the real catalog and its answer is discarded unless it is
one of them verbatim, so a hallucinated constraint cannot reach retrieval: a
string that is not already in `card_index` has no posting list to contribute.
Every entry point returns its input unchanged on any failure and never raises.

**Why it is off for scoring.** The rules say official scoring *"may disable
network access"* and prohibit *"code that depends on undeclared external services
for official final scoring"*. The submitted configuration therefore makes no
network call and reports zero tokens — and the guarantee is structural, not a
promise: `copilot/llm.py` is imported lazily from inside the enabling branch, and
`test_llm_stage_is_never_loaded_by_default` starts a fresh interpreter and fails
if `copilot.llm` appears in `sys.modules` after the agent is built.

**Measured, on 40 sessions per cell against DeepSeek `deepseek-v4-flash`:**

| | L3 | L4 |
|---|---|---|
| LLM off | 0.8600 | **0.8314** |
| LLM on, answer injected at full weight | 0.7823 | 0.7803 |
| LLM on, answer weighted as a guess | **0.8782** | 0.8206 |

The first version was a clean failure with a clean cause. Hit rate *rose* at both
levels (+0.025 at L3, +0.050 at L4) — the model really is recovering constraints
the parser lost — but MRR fell 0.17 and the net score went down. The mistake was
ours: we injected the model's answer at full weight, as though the customer had
said it. Under the confidence gate one wrong pick both tops a one-item list and
collapses the candidate set below the commit threshold, so the agent converges
early on the wrong product. Weighting it like a mined guess flips L3 to +0.018;
L4 still slips 0.011.

It stays off in the shipped configuration for three reasons, in order: official
scoring may disable the network, the official wording parses without it so it
would never fire, and a call costs 1–8 s against a 66 ms turn.

One implementation note worth recording: it is a reasoning model, and `max_tokens`
bounds the reasoning trace *plus* the answer. At 512 roughly a third of calls
returned an empty string with no error at all — not a truncated answer, no
answer. The budget is 2048.

`retrieve()` scores candidates once per turn instead of once for ranking and
again for question estimation; that change did reduce latency, measured before
the machine-load problem above was understood.

## Layout

```
copilot/
  simulator_model.py   reconstruction of the organizer's utterance derivation
  catalog.py           catalog load, inverted indexes, both FTS5 indexes
  dialog.py            utterance parsing, constraint mining, session state
  replay.py            replaying the dialogue against a candidate
  retrieval.py         multi-route retrieval and fusion
  questions.py         question selection, by information gain or convergence
  agent.py             per-turn orchestration, including the confidence gate
  llm.py               optional inbound paraphrase parsing; off, lazily imported
  config.py            every route and strategy behind a flag, for ablation
starter/
  agent.py             adapter exposing the required `Agent` interface
  baseline_agent.py    organizer's original weak BM25 agent, kept for reference
tools/
  harness.py           shared replay harness with a perturbation hook
  memcheck.py          index memory footprint and build time
  generalize.py        held-out unseen-target evaluation
  ablation.py          component ablation
  robustness.py        paraphrase stress test
  profile_cost.py      startup, latency, memory, token accounting
  demo.py              one verbose multi-turn session
  explore.py           the analysis that motivated the design
  explore2.py          candidate-narrowing measurement
tests/                 22 tests: derivation parity, parsing, invariants,
                       question estimation, replay evidence, offline contract
results/               raw output backing every table above
RUN.md                 one-page reproduction guide
```

`evaluator/`, `data/public_set.jsonl`, and the organizer's `docs/` are unmodified
— verified by SHA256 against the upstream repository. The organizer's own tests
pass.

## Setup

See **[RUN.md](RUN.md)** for the one-page version. Short form:

```bash
curl -L -o catalog.jsonl.gz \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
sha256sum -c --ignore-missing SHA256SUMS   # SHA256SUMS also lists the kit zip, which we do not need
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl

python -m evaluator.local_evaluator     # -> results.json, TechnicalScore 0.971714
```

Python 3.10+, no third-party runtime dependencies.

## Limitations and what we would do next

- **Held-out testing holds the generator fixed.** Our 800-session stand-in varies
  the targets and the users, which is the property we most needed to test, but it
  reuses the organizer's own `intent_card` derivation. If the private set's
  generator differs in some way we cannot see, that difference is invisible to us.
- **The held-out harness cannot adjudicate profile-dependent features.** Its
  `preference_tags` were originally drawn at random, which zeroed the signal by
  construction; they now correlate with the target, but only because we *ground
  them in the target's own text*. That is a leak, not an observation. It reproduces
  the aggregate statistics of the real tags and still understates their
  discriminative power (target beats peers 44.6% of the time, against 67.5% on the
  public set), so any profile result it produces is directional at best. This is
  why the `use_profile` decision is made on public-set evidence even though
  held-out disagrees.
- **We never tested against a real LLM paraphraser**, only a scripted one. The
  robustness levels are our best construction of what paraphrasing does, not an
  observation of it. This is the first thing we would add with more time. (We did
  measure a real model on the *reading* side — see
  [the optional LLM stage](#the-optional-llm-stage-and-why-it-is-off) — but that
  tests recovery from our paraphrase, not whether our paraphrase is realistic.)
- **Intent Override is handled as the evaluator behaves, not as the prompt
  describes.** If the private simulator genuinely switches targets mid-session,
  accumulate-don't-erase becomes wrong. The fix would be to detect a sustained
  drop in agreement between old and new evidence and decay the pre-override
  constraints; we scoped it out because the public evaluator gives no signal to
  tune such a detector on, and guessing would have been worse than documenting
  the assumption.
- **Mining is an aggregate signal we do not fully understand.** We know what it
  is *not* (constraint recovery: precision ~0.27, and tripling recall changes
  nothing) and we tuned it accordingly, but "boost products whose card text
  overlaps the turn" is a description of its effect rather than a principled
  mechanism. A properly calibrated soft-match score would likely beat it; we ran
  out of time to build one that keeps the zero-dependency property.
- **The confidence gate is the largest single component and the least
  conventional.** It is worth −0.0578 in the ablation, and it works by declining
  to answer. If the private evaluator scored partial answers differently — for
  instance by penalising short lists — that entire gain would invert. Nothing in
  the specification suggests it does (`up to 10` is explicit, and only the first
  ten valid unique IDs are scored), but it is a single assumption carrying a lot
  of weight, and `COPILOT_USE_CONFIDENCE_GATE=0` reverts it.

- **The public score is not a forecast.** Held-out is 0.930 against 0.972 public,
  and one of our late changes gained on the public set and nothing held-out. We
  quote 0.930 as the expectation and treat the difference as fitted.
- **`boundary` is our weakest scenario on held-out data** (hit 0.950, MRR 0.65–0.68
  against 0.85 on the public set). It is only 5% of sessions and n=40 per draw, so
  the estimate is noisy, but it is the one place the held-out gap is consistent
  across both seeds and is where we would look next. Part of it is structural: the
  simulator spends the first question of a boundary session refusing to answer, so
  a turn is lost no matter what the agent asks.

## Author

**LONG HONGYU** — Nanyang Technological University
<hongyu021@e.ntu.edu.sg> · [github.com/scut-chika](https://github.com/scut-chika)

A solo submission. Every part of this project — reading the evaluator, the
retrieval design, the question policy, the confidence gate, the generalization
and robustness harnesses, and the write-up — is the work of one person.

Shopping Copilot: AI Conversational Search and Recommendations.
TikTok TechJam 2026.

## Attribution

Catalog and sessions derive from Amazon Reviews 2023 (McAuley Lab, UCSD) — see
`DATA_ATTRIBUTION.md`. No secrets are committed; the default configuration reads
no credentials and makes no network calls.
