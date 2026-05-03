# Pre-computed Results

This directory contains the raw CSV outputs used to generate the tables and statistical claims in the paper. Each subfolder maps to one or more specific tables/sections.

## Directory map

| Folder | Source script | Paper artifact |
|---|---|---|
| `causaltime_10seed/` | `experiments/exp7_real_benchmarks_trust.py --bench causaltime --seeds 0..9` | App. table "CausalTime: 10-seed Trust Validation"; trust vs. per-group $p$-values cited in Sec. 4.3 |
| `nonlinear/` | `experiments/exp1_synthetic_benchmark.py --sub nonlinear --seeds 0 1 2 --dims 10 20` | App. "Limitations on Nonlinear Data" (per-acc Table); Sec. 4 nonlinear PCMCI+ comparison |
| `scale/` | `experiments/exp1_synthetic_benchmark.py --sub scale --seeds 0 1 2 --dims 20 50 100` | App. "Main-text Scalability with Baselines" (Table $d{\in}\{20,50,100\}$) |
| `cross_sectional/` | `experiments/exp5_cross_sectional.py` | App. "Cross-Sectional Structure Learning" (NOTEARS / DAGMA / NOTEARS+mask vs PRCD-MAP) |
| `ablation/` | `experiments/exp3_ablation.py --sub {synthetic,lorenz,real,hard_mask}` | Sec. 4.5 ablation table; App. ablation breakdowns |
| `community_mixing/` | `experiments/exp10_community_mixing.py --seeds 0..4` | Sec. 4.4 Community Mixing table; App. extended settings |
| `significance/` | `experiments/exp11_significance_test.py --seeds 0..9` | Sec. 4.4 sign-test $p{=}0.002$ claim |

## How to reproduce

```bash
# From repo root
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
cd experiments
python exp7_real_benchmarks_trust.py --bench causaltime --seeds 0 1 2 3 4 5 6 7 8 9
# ... (see scripts/run_all.sh)
```

Required environment: see `requirements.txt`. CausalTime data must be placed under `data/causaltime/{AQI,Traffic,Medical}/`.

## Schema notes

All CSVs share a common metric schema:
- `auroc`, `auprc`, `f1_opt` — combined-graph metrics
- `w0_*` — instantaneous-graph-only metrics
- `comb_*` — alias for combined metrics (some scripts)
- Identifier columns: `method`, `seed`, `setting` / `benchmark`, `prior_acc` / `acc`, `d`, `T`

For grouping/aggregation, see the scripts themselves; representative analysis snippets are in `experiments/exp11_significance_test.py` (paired $t$-test) and `experiments/exp1_synthetic_benchmark.py` (multi-method aggregation).
