# Devpost submission text

Paste the sections below into the corresponding Devpost fields. Placeholders in
`{{ }}` need filling before submitting.

---

## Project name

**Shopping Copilot — asking the right question beats retrieving better**

## Elevator pitch (200 chars max)

> A conversational shopping agent that finds a hidden product in 2 turns instead
> of 10, by computing which question reveals the most. No LLM, no network, $0.

---

## Problem statement

Track 4 — **Shopping Copilot: AI Conversational Search and Recommendations**

---

## About the project

### What it does

Given a customer who starts vague ("I'm looking for tunics, but I'm still
exploring") and a 50,000-product Amazon catalog, the agent has ten turns to get
the customer's hidden target product into its top 10. It does it in **2.05 turns
on average**, hitting on **100% of the 200 public sessions**.

| | Hit Rate@10 | MRR | MTTC | **TechnicalScore** |
|---|---|---|---|---|
| Organizer BM25 baseline | 0.125 | 0.068 | 9.81 | **0.1067** |
| **Shopping Copilot** | **1.000** | **0.790** | **2.05** | **0.9160** |

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
test. It is our highest-MRR scenario (0.941).

### What we found that surprised us

**Asking beats retrieving, by a lot.** We expected the exact-matching index to be
the story. The ablation says otherwise: removing the question policy costs
**−0.4797**, while removing the exact-match index costs **−0.0951**. With
retrieval reduced to plain BM25 the system still scores 0.8457. The result is
about the questions.

**The parser was the fragile part, not the matching.** We built a paraphrase
stress harness expecting exact matching to break first. Rewording only the
*templates*, leaving constraint text untouched, cost more score than heavily
paraphrasing the constraints themselves. We added template-independent constraint
mining to close about half that gap.

**One feature was actively harmful.** The anonymized `user_profile` tags — "fit",
"comfort", "durability" — match nearly every clothing item, so they added noise to
tie-breaking. Disabling it *gained* 0.0142. It is off by default, kept behind a
flag so the finding stays reproducible.

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
| public | 200 | 0.9160 | — |
| held-out, seed A | 800 | 0.8861 | **96.7%** |
| held-out, seed B | 800 | 0.8749 | **95.5%** |

Two independent draws, both retaining ~96%. We fit the task, not the sessions.

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
| Startup | 32.1 s once, to index 50,000 products |
| Per-turn latency | mean 82 ms, p99 196 ms |
| Peak memory | 304 MB, in-process |

We shipped an optional LLM reranking hook and left it **off**: it would forfeit
the offline guarantee for no measured gain.

### How we built it

Per turn: parse the utterance into constraints → accumulate them (never erase) →
score candidates through four fused routes (card-exact, loose text, category,
BM25) → estimate which question minimises the surviving candidate set → return a
full ranked ten *and* a question in the same response.

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

A real LLM paraphraser instead of our scripted one; character-level or embedding
similarity for constraint mining to fix the extreme-paraphrase regression; and an
evidence-disagreement detector for the case where a private simulator genuinely
does switch targets mid-session.

---

## Built with

`python`, `sqlite3` (FTS5), `pytest`

**Development tools:** VS Code, Git, Claude Code
**APIs used:** none
**Libraries and frameworks:** none beyond the Python standard library (`json`,
`re`, `sqlite3`, `math`, `collections`, `dataclasses`, `pathlib`). `pytest` for
tests only.
**Datasets and assets:** the organizer's frozen 50,000-product catalog and 200
public sessions, derived from Amazon Reviews 2023 (McAuley Lab, UCSD). No external
data was added.

---

## Links

- **Repository:** {{ GITHUB_URL }}
- **Demo video:** {{ YOUTUBE_URL }}

## Team

{{ TEAM_MEMBERS_AND_CONTRIBUTIONS }}
