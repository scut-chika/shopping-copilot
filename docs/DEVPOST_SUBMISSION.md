# Devpost submission text

Paste the sections below into the corresponding Devpost fields. Placeholders in
Everything below is filled in; nothing is left to substitute.

---

## Project name

**Shopping Copilot — asking the right question beats retrieving better**

## Elevator pitch (200 chars max)

> A shopping agent that finds a hidden product in 2 turns, by computing which
> question reveals the most — and refusing to answer until it knows. $0, offline.

---

## Problem statement

Track 4 — **Shopping Copilot: AI Conversational Search and Recommendations**

---

## About the project

### What it does

Given a customer who starts vague ("I'm looking for tunics, but I'm still
exploring") and a 50,000-product Amazon catalog, the agent has ten turns to get
the customer's hidden target product into its top 10. It does it in **2.1 turns
on average**, hitting on **100% of the 200 public sessions**, and ranking the
target **first in 194 of them**.

| | Hit Rate@10 | MRR | MTTC | **TechnicalScore** |
|---|---|---|---|---|
| Organizer BM25 baseline | 0.125 | 0.068 | 9.81 | **0.1067** |
| **Shopping Copilot** | **1.000** | **0.979** | **2.10** | **0.9717** |

Measured by the organizer's evaluator, run unmodified.

### The insight

We started where everyone starts — treat this as semantic search, embed the query,
embed the catalog, rank by similarity — and then read the organizer's evaluator
before writing any code. That changed the whole design.

The evaluator builds every customer utterance deterministically from the **target
product's own catalog record**: it flattens the product's `features` and
`details`, prepends a regex-matched material and colour, and keeps the first four
strings. Those four strings are the entire vocabulary of the conversation.

Three things follow, and each one changed what we built:

**1. The customer speaks in verbatim catalog text.** So the right primary index
is not a vector store — it is an inverted index from constraint string to the
products that could have produced it. We reconstruct the organizer's derivation
across all 50,000 products and invert it. With the category plus two disclosed
constraints, 76% of sessions collapse to ten or fewer candidates; with three, 94.5%.

**2. Question value is computable, not guessable.** What a question reveals
follows a public rule, so for each candidate attribute we ask "if the target were
product X, what would this question return?", group the candidates by predicted
answer, and pick the question that minimises the expected surviving group. This is
real expected-information-gain estimation, not a scripted question order.

**3. "Intent Override" never actually changes the target.** The problem statement
asks for *slot erasure and rewriting*. The evaluator's override takes both its old
and new value from the *same* product's intent card, and the target `parent_asin`
never changes. Implementing literal erasure would throw away valid evidence and
lose score. We accumulate and re-weight instead, and pinned that decision with a
test, which the ablation still credits at every configuration since.

### What we found that surprised us

**Asking beats retrieving, by a lot.** We expected the exact-matching index to be
the story. The ablation says otherwise: removing the question policy costs
**−0.3318**, while removing the exact-match index — the original insight of the
whole project — costs **−0.0006**. With retrieval reduced to plain BM25 the
system still scores 0.9641. The result is about the questions, and about knowing
when not to answer.

**A hit locks in whatever rank it landed on.** The evaluator ends the session the
moment the target appears, so a *lucky* turn-1 hit at rank 8 is not a win — it
books rank 8 permanently and denies us the turn that would have made it rank 1.
Over half our remaining MRR loss was exactly that. The arithmetic is lopsided:
deferring costs 0.20/10 of efficiency and buys up to 0.30 of MRR, about 13:1. So
while the dialogue has not left exactly one candidate standing, the agent returns
**its single best guess plus the question** instead of padding out ten it knows
are wrong. That single change is worth **+0.0578** — more than every retrieval
route combined.

Read the other way, it is the *"retrieval cutoff when facing Over-Generality"*
the track's Proactive Guidance pillar asks for. Both readings are true and we
would rather state the scoring arithmetic than dress it up. Returning *zero*
items would score higher still; we did not ship that, because an assistant that
answers a shopper with an empty list is not doing the job. That is the one place
we left score on the table on purpose.

**The parser was the fragile part, not the matching.** We built a paraphrase
stress harness expecting exact matching to break first. Rewording only the
*templates*, leaving constraint text untouched, cost more score than heavily
paraphrasing the constraints themselves. Template-independent constraint mining
closes most of that gap.

**Our own component did not do what we thought.** Constraint mining recovers the
customer's stated constraint with precision ~0.27, and tripling its recall
changed the end score by *nothing*. That null result told us it is an aggregate
signal, not a retrieval step -- so we widened it to admit 16 weak matches instead
of 4 confident ones, which gained at every paraphrase level and removed its
cost on clean data. Four other hypotheses about it were falsified along the way
and are written up in the README.

**We published a wrong number and caught it.** An early version of our
resource table claimed a 3.2x build-time speedup and a 2.6x latency
improvement. Re-measuring old and new configurations back to back on the same
machine showed the real figures: 1.5x on build time, and *no* latency change at
all. The memory result (6.1x) held. The README carries the correction.

**One feature was actively harmful.** The anonymized `user_profile` tags — "fit",
"comfort", "durability" — match nearly every clothing item, so they added noise to
tie-breaking. Disabling it *gains* 0.0052. It is off by default, kept behind a
flag so the finding stays reproducible.

**Our biggest piece of work ended up nearly redundant.** We found that the
simulator answers deterministically, so instead of asking "does this product
*contain* what the customer said" we could ask "would this product have *said*
it" — which also uses what the customer did not say. It collapses the working
candidate set from ~632 to ~18 and was worth +0.0031 on its own. Then we built
the confidence gate, and it fell to −0.0005: the two overlap, because the gate
declines to convert until the set has collapsed and replay only makes it collapse
sooner. We are reporting that rather than restating the earlier number.

### Does it generalize?

A 1.000 hit rate on 200 sessions is a warning sign, not a victory lap — at the
ceiling the public set cannot tell you anything more, and further tuning just
fits those sessions. The private set has 800 sessions with different users and
different targets.

So we built a stand-in: 800 sessions over target products that appear **nowhere in
the public set**, with independently resampled user profiles, at the official
40/40/15/5 scenario mix.

| Set | n | Score | Retained |
|---|---|---|---|
| public | 200 | 0.9717 | — |
| held-out, seed A | 800 | 0.9307 | **95.8%** |
| held-out, seed B | 800 | 0.9298 | **95.7%** |

**If you want one number for how this is likely to do on the private set, it is
0.930, not 0.972.** We would rather say that than quote the public figure and let
it be read as a forecast. As the new mechanisms landed, held-out moved 0.8815 →
0.9303, so about 92% of the public gain is real — and we can name the part that
is not: re-tuning one weight was worth +0.0094 on the public set and *exactly
nothing* held-out. We kept it for a robustness gain and do not count the rest.

### It runs with the network switched off

The submission rules note that *"for official final scoring, organizer policy may
disable network access"*. Shopping Copilot makes **no network calls, uses no LLM,
reads no credentials, and has no third-party dependencies** — only the Python
standard library. A test parses the AST of every shipped module and fails if a
network client or third-party package ever appears.

| | |
|---|---|
| Model | none |
| API cost | **$0.00** |
| Token usage | 0 prompt / 0 completion |
| Index build | 16.3 s once, for 50,000 products |
| Per-turn latency | mean 66 ms, p99 149 ms |
| Process RSS | 205 MB, in-process |

**There is a real LLM stage, and it is off.** `copilot/llm.py` maps a
*paraphrased customer utterance* back onto the catalog constraint that produced
it — the one direction where a model has something to offer here. Generating the
prose is worth nothing (the evaluator reads `ask_attribute`, never `message`) and
choosing the question is worse than nothing (that has a closed-form optimum). The
model can only *select* from real catalog strings; anything else it returns is
discarded, so a hallucination cannot reach retrieval.

Measured against DeepSeek `deepseek-v4-flash` under heavy paraphrase, the first
version did exactly what it was built to do and still lost: **hit rate rose**
(+0.025 at L3, +0.050 at L4 — it really is recovering constraints) while **MRR
fell 0.17**. The cause was ours, not the model's: we were injecting its answer at
full weight, as though the customer had said it. One wrong pick then both tops a
one-item list and collapses the candidate set below the gate's threshold,
committing early to the wrong answer.

Weighting it as a *guess* instead flipped the result — L3 0.8600 → **0.8782**,
though L4 still slips 0.8314 → 0.8206 (40 sessions per cell; small). So the stage
works, and it stays off anyway: official scoring may disable the network, the
official wording never triggers it, and each call costs 1–8 s against a 66 ms
turn. That is a measurement, not a preference.

The offline guarantee is structural, not a promise: `copilot/llm.py` is imported
lazily from inside the branch that enables it, and a test starts a fresh
interpreter and fails if it appears in `sys.modules` after the agent is built.

### How we built it

Per turn: parse the utterance into constraints → accumulate them (never erase) →
replay the dialogue against each candidate to drop the ones that would have
answered differently → score through fused routes (card-exact, category, BM25) →
estimate which question is most likely to leave exactly one candidate → then
either commit to a full ranked ten, or return a single best guess and the
question, depending on whether the dialogue has settled.

Two invariants that mattered more than any tuning:

- **Always return ten recommendations.** The evaluator checks for a hit before
  generating the customer's reply, and recommending costs nothing — so answering
  with a question alone is pure waste. The organizer's baseline never sets
  `ask_attribute` at all, which is why its MTTC is 9.81.
- **Never raise.** An exception becomes an empty turn, wasting one of only ten.

### Challenges

Deciding what to do when no configuration dominates. Constraint mining wins under
light paraphrase, loses under extreme paraphrase, and costs 0.0029 on clean text.
We shipped it enabled, documented the trade-off in full, and left it behind one
environment variable — rather than picking whichever number looked best. Two
tuning attempts failed outright and we kept them in the git history and the README
instead of quietly dropping them.

### What we learned

That reading the evaluator was worth more than any model choice. And that the
honest version of "we score 1.000" is "we score 1.000, here is why that number is
suspicious, and here is the held-out experiment we ran to find out whether it
means anything."

### What's next

Feed the LLM stage a *calibrated* confidence rather than a hard pick, so its
recall gain stops costing precision — the measurement above says the recovery
works and only the weighting is wrong. A real LLM paraphraser instead of our
scripted one, to check whether L1–L4 resemble the paraphrase the organizer would
actually apply. And an evidence-disagreement detector for the case where a
private simulator genuinely does switch targets mid-session.

---

## Built with

`python`, `sqlite3` (FTS5), `pytest`

**Development tools:** VS Code, Git, Claude Code
**APIs used:** none in the scored configuration, which reports 0 tokens. DeepSeek
`deepseek-v4-flash` (OpenAI-compatible) was used only to measure the optional
paraphrase-parsing stage, which ships disabled. Measured cost per call: ~220
prompt + ~220 completion tokens; we did not instrument the total for the A/B run
and so do not quote a dollar figure.
**Libraries and frameworks:** none beyond the Python standard library (`json`,
`re`, `os`, `sqlite3`, `math`, `collections`, `dataclasses`, `pathlib`, and
`urllib` in the optional, disabled LLM module). `pytest` for tests only.
**Datasets and assets:** the organizer's frozen 50,000-product catalog and 200
public sessions, derived from Amazon Reviews 2023 (McAuley Lab, UCSD). No external
data was added.

---

## Links

- **Repository:** https://github.com/scut-chika/shopping-copilot
- **Demo video:** https://youtu.be/sgenegmZxXo

## Team

**LONG HONGYU** — Nanyang Technological University
hongyu021@e.ntu.edu.sg · github.com/scut-chika

Solo submission; all work by one person. (The deliverables list asks for team
member contributions "if applicable, i.e. team participants, non-solo
participants".)
