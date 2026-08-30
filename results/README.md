# Raw evidence

Every table in the top-level `README.md` is generated from a file here. These are
the unedited outputs of the tools, committed so the numbers can be checked rather
than taken on trust.

| File | Produced by | Backs |
|---|---|---|
| `public_set.json` | `python -m evaluator.local_evaluator` | the headline score |
| `generalization_seed20260830.json` | `python tools/generalize.py` | held-out targets, seed 20260830 |
| `generalization_seed7.json` | `python tools/generalize.py --seed 7` | held-out targets, seed 7 |
| `ablation.json` | `python tools/ablation.py` | the ablation table |
| `robustness.json` | `python tools/robustness.py` | the paraphrase table (shipped config) |
| `robustness_mining_on.json` | same, mining enabled | the mining A/B |
| `robustness_mining_off.json` | `COPILOT_USE_CONSTRAINT_MINING=0 python tools/robustness.py` | the mining A/B |
| `cost_profile.json` | `python tools/profile_cost.py --memory` | latency, memory, token accounting |

`public_set.json` is the output of the organizer's evaluator run unmodified, and
is the only number that is an official score. Everything else is our own
measurement.
