# Reproduction — one page

## Requirements

- **Python 3.10 or newer** (developed and measured on CPython 3.14.7, Windows 11).
- **No third-party runtime dependencies.** `pip install -r requirements.txt`
  installs nothing; the agent uses only the standard library.
  `sqlite3` must be built with the **FTS5** extension, which is the default in
  official CPython builds. Verify with:

  ```bash
  python -c "import sqlite3;sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)');print('FTS5 OK')"
  ```

- **Network access: not required.** The agent makes no network calls, reads no
  credentials, and calls no model API. It runs unchanged with networking
  disabled.

## Get the catalog

The 50k-product catalog is a release asset and is not committed (58 MB raw).

```bash
curl -L -o catalog.jsonl.gz \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
sha256sum -c --ignore-missing SHA256SUMS   # SHA256SUMS also lists the kit zip, which we do not need
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
```

## One command to score the agent in the official harness

```bash
python -m evaluator.local_evaluator
```

Writes `results.json`. Expected: `recommended_technical_score` **0.971714**,
Hit Rate@10 1.0, MRR 0.979048, MTTC 2.100, and `total_tokens` 0.

## Everything else

```bash
python tools/generalize.py --sessions 800   # held-out targets  -> generalization.json
python tools/ablation.py                    # component ablation -> ablation_final.json
python tools/robustness.py                  # paraphrase stress  -> robustness.json
python tools/profile_cost.py --memory       # latency and memory -> cost_profile.json
python tools/demo.py --scenario browsing --index 1   # one narrated session
python -m pytest tests/ -q                  # test suite (needs pytest)
```

## Environment variables

None are required. Any `Config` field in `copilot/config.py` can be overridden
for one run with `COPILOT_<FIELD_NAME>`, which is how the ablation and
robustness sweeps vary a single setting without editing code. Examples:

```bash
COPILOT_USE_CONSTRAINT_MINING=0 python -m evaluator.local_evaluator
COPILOT_QUESTION_STRATEGY=other  python -m evaluator.local_evaluator
```

No environment variable is needed for the submitted configuration, and none
carries a secret.

## Resource envelope

The rules reserve the right to score "under CPU, memory, timeout, and network
restrictions", so these are measured, not estimated. Over 410 turns on the public
set (`results/cost_profile.json`, `results/memory_profile.json`). The two build
figures are the same operation timed by two different tools, quoted separately
rather than averaged into a number neither of them produced:

| | |
|---|---|
| Index build (once per process) | **16.3 s** (`profile_cost.py`) / **16.7 s** (`memcheck.py`) |
| Per-turn latency | mean 66 ms, p50 64 ms, p95 117 ms, p99 149 ms |
| Process RSS after index build | **206 MB** |
| Python heap (tracemalloc peak) | **50.4 MB** |
| LLM calls / tokens / cost | 0 / 0 / **$0.00** |
| Network | none required |

Absolute timings vary with host load; measure on your own machine with the two
commands below rather than trusting these figures.

The gap between RSS and the Python heap is the two in-memory SQLite FTS5
indexes, which SQLite allocates in C and `tracemalloc` therefore does not see.
RSS is the number that matters for a memory cap.

The index build happens once in `Agent.__init__`, not per session or per turn.

Reproduce with `python tools/memcheck.py` and `python tools/profile_cost.py`.

## Verified from a clean clone

This exact sequence was run against a fresh `git clone` of the public repository
on 31 Aug 2026 and reproduced `recommended_technical_score` **0.971714**, with
all 22 tests passing. If it does not reproduce for you, that is a bug and we want
to know.

## Mapping to the recommended submission layout

`docs/submission_rules.md` suggests `submission/{agent.py, requirements.txt,
README.md, src/}`. The official harness imports `starter.agent`, so we keep that
path rather than renaming it and breaking the one command that scores us:

| Recommended | Here |
|---|---|
| `agent.py` (exports `Agent`) | `starter/agent.py` |
| `src/` (helper modules) | `copilot/` |
| `requirements.txt` | `requirements.txt` (installs nothing) |
| `README.md` | `README.md`, plus this file |
