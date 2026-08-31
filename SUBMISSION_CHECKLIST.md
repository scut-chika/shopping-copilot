# Submission checklist

Checked against the Track 4 **Deliverables** section of the official problem
statement, not from memory. Three deliverables, all three complete.

## 1. Written Project Description (via Devpost) — ready to paste

Text: [`docs/DEVPOST_SUBMISSION.md`](docs/DEVPOST_SUBMISSION.md).

| Required by the deliverables list | Where |
|---|---|
| How the solution addresses the problem statement | "About the project", "The insight" |
| Development tools used | "Built with" — VS Code, Git, Claude Code |
| APIs used | "Built with" — none in the scored configuration; DeepSeek used only to measure the optional, disabled stage |
| Libraries and frameworks used | "Built with" — Python standard library only; `pytest` for tests |
| Datasets and assets used | "Built with" — the organizer's frozen catalog and public sessions |

Complete. Demo video linked: <https://youtu.be/sgenegmZxXo>

## 2. Public GitHub repository — done

<https://github.com/scut-chika/shopping-copilot>, public.

| Required | Where |
|---|---|
| Well-structured, commented code covering all components | `copilot/`, `starter/`, `tools/`, `tests/` |
| README: project overview | top of `README.md` |
| README: setup and installation | `README.md` → Setup, and [`RUN.md`](RUN.md) |
| README: steps to reproduce results | [`RUN.md`](RUN.md); verified from a clean clone |
| README: limitations and what we would improve | `README.md` → Limitations |
| README: team member contributions | Not applicable — the list says *"if applicable, i.e. team participants, non-solo participants"*. Authorship is recorded anyway under Author. |

## 3. Demo video — done

<https://youtu.be/sgenegmZxXo>

- [x] demonstrates the solution working end to end
- [x] uploaded to YouTube, set to public visibility
- [x] linked in the Devpost description
- [x] contains no third-party trademarks or copyrighted content

A terminal walkthrough is explicitly acceptable here: *"Note for backend/NLP
tracks: if a front-end interface is not applicable to your solution, a
walkthrough video showing API usage, inference examples, or result analysis is
accepted."* UI/UX is out of scope for this track, so no interface is needed.

Shot list and narration: [`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md).
Every figure in it has been checked against the current measurements, and the one
line that had become false was rewritten.

## Compliance notes

- **Devpost "New & Existing" rule** — the project must be significantly updated
  after the submission period opened (29 Aug 12:00 SGT). Satisfied: 20+ commits
  on 30–31 Aug, including the two mechanisms that produce most of the score.
- **No secrets committed.** The default configuration reads no credentials and
  makes no network call; `test_agent_reads_no_secrets_from_environment` pins it.
- **Evaluator untouched.** `evaluator/`, `data/public_set.jsonl` and the
  organizer's `docs/` are byte-identical to upstream.
- **Runs offline.** Official scoring may disable network access; the scored
  configuration needs none and reports zero tokens.

## Headline figures, for filling in forms

| | |
|---|---|
| Official TechnicalScore (200 public sessions) | **0.971714** |
| Hit Rate@10 / MRR / MTTC | 1.000 / 0.9790 / 2.10 |
| Held-out estimate for the private set (800 unseen targets x2) | **0.930** |
| Organizer BM25 baseline | 0.1067 |
| Model / API cost / tokens | none / $0.00 / 0 |
| Index build / per-turn latency | 16.7 s once / 66 ms mean, 149 ms p99 |
