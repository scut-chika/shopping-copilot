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
| `robustness_mining_off.json` | `COPILOT_USE_CONSTRAINT_MINING=0 python tools/robustness.py` | the mining A/B |
| `robustness_loose_index_off.json` | `COPILOT_USE_LOOSE_INDEX=0 python tools/robustness.py` | evidence the loose index does nothing |
| `cost_profile.json` | `python tools/profile_cost.py` | latency and token accounting |
| `memory_profile.json` | `python tools/memcheck.py` | RSS, Python heap, build time |

Sweeps and A/Bs behind specific claims:

| File | Produced by | Backs |
|---|---|---|
| `sweep_gate_threshold_size.json` | `tools/sweep.py --grid gate_candidate_threshold=… --grid gate_list_size=…` | the confidence gate is a flat optimum |
| `sweep_gate_maxturn.json` | `tools/sweep.py --grid gate_max_turn=…` | where gating stops paying |
| `sweep_question_objective.json` | `tools/sweep.py --grid question_objective=…` | convergence vs expected-size |
| `sweep_bm25_weight.json` | `tools/sweep.py --grid bm25_weight=…` | the weight the gate invalidated |
| `robustness_bm25_{0,2,3,5,7,10,20}.json`, `robustness_no_bm25.json` | `COPILOT_BM25_WEIGHT=N tools/robustness.py` | why the peak is 5, not 0 |
| `robustness_no_gate.json` | `COPILOT_USE_CONFIDENCE_GATE=0 tools/robustness.py` | the gate isolated under paraphrase |
| `robustness_gate_parsehealth.json` | `COPILOT_GATE_NEEDS_CLEAN_PARSE=1 tools/robustness.py` | the falsified parse-health hypothesis |
| `diag_rb_{a,b,c,d}.json` | one flag each at L3/L4 | locating the `disclosed`-vs-`seen` regression |
| `robustness_llm_off_matched.json` | `tools/robustness.py --levels 3,4 --sessions 40` | LLM A/B baseline |
| `robustness_llm_on_fullweight.json` | same, LLM on, answer at full weight | the version that lost |
| `robustness_llm_on_downweighted.json` | same, LLM on, answer down-weighted | the version that works |

`robustness_llm_off.json` and `robustness_current.json` / `robustness_adaptive.json`
/ `robustness_gate_v1.json` / `official_replay_consistency.json` /
`official_confidence_gate.json` are intermediate states kept so the sequence of
measurements can be walked, not just its endpoint. Where an intermediate baseline
was taken at a different configuration and is *not* comparable to a later cell,
the matched re-run is the one named above.

`public_set.json` is the output of the organizer's evaluator run unmodified, and
is the only number that is an official score. Everything else is our own
measurement.
