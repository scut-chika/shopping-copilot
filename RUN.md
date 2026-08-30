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
sha256sum -c SHA256SUMS
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
```

## One command to score the agent in the official harness

```bash
python -m evaluator.local_evaluator
```

Writes `results.json`. Expected: `recommended_technical_score` **0.916014**,
Hit Rate@10 1.0, MRR 0.790046, MTTC 2.05, and `total_tokens` 0.

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

Measured over 410 turns on the public set (`cost_profile.json`):

| | |
|---|---|
| Startup (build all indexes over 50,000 products) | 32.1 s, once per process |
| Per-turn latency | mean 82 ms, p50 78 ms, p95 155 ms, p99 196 ms, max 231 ms |
| Peak index memory | 304 MB, in-process |
| LLM calls / tokens / cost | 0 / 0 / $0.00 |

The index build happens once in `Agent.__init__`, not per session or per turn.
