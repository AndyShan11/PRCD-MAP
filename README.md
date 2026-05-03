# PRCD-MAP

**Learning How Much to Trust Domain Priors for Causal Structure Discovery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

NeurIPS 2026 submission (anonymous).

## Key Idea

Existing causal discovery methods either ignore priors or impose them globally---but real priors have **spatially varying reliability** (physical laws give high-confidence edges, LLM suggestions are speculative). PRCD-MAP is the first framework with **structure-aware trust calibration**:

- **Reliable prior** → $+0.158$ AUROC over best baseline in low-data regimes
- **Mediocre prior** → graceful fallback to no-prior ($\leq -0.038$)
- **Fixed-trust alternatives** → collapse ($-0.156$)

The core mechanism is **structure-aware trust propagation** (per-edge $\tau$ learned by aggregating neighborhood consistency), which strictly improves over per-group temperature under heterogeneous priors ($\Omega(1/G)$ gap, Theorem 6).

<p align="center">
  <img src="assets/prior_sweep.png" width="75%" alt="Asymmetric robustness curve">
</p>

## Repository Structure

```
PRCD-MAP/
├── src/                              # Core model implementations
│   ├── model_linear.py                   # Linear SVAR + per-group τ (baseline)
│   ├── model_nam.py                      # Neural Additive Model variant
│   ├── trust_propagation.py              # Structure-aware trust (GAT + Lite)
│   ├── model_linear_trust.py             # Linear + trust propagation
│   ├── model_nam_trust.py                # NAM + trust propagation
│   ├── utils.py                          # Data gen, baselines, metrics
│   └── utils_trust.py                    # Trust-propagation wrappers
│
├── experiments/                      # 12 experiment scripts
│   ├── exp1_synthetic_benchmark.py           # Synthetic SVAR (Table 1)
│   ├── exp2_real_benchmarks_original.py      # CausalTime + electricity
│   ├── exp3_ablation.py                      # Ablation (Table 3)
│   ├── exp4_scalability.py                   # Scalability
│   ├── exp5_cross_sectional.py               # Cross-sectional SEM (App K)
│   ├── exp6_trust_validation.py              # Trust vs per-group
│   ├── exp7_real_benchmarks_trust.py         # Trust on real data
│   ├── exp8_scalability_trust.py             # Trust scalability
│   ├── exp9_llm_prior_pipeline.py            # LLM prior end-to-end (App B)
│   ├── exp10_community_mixing.py             # Designed validation (Table 7)
│   ├── exp11_significance_test.py            # 10-seed paired test (App L)
│   └── exp12_theory_verification.py          # Numerical theorem check
│
├── data_loaders/                     # Data prep + baseline runners
│   ├── generate_llm_priors.py
│   ├── baseline_dycast.py
│   └── baseline_rhino.py
│
├── scripts/run_all.sh                # One-click reproduction
├── results/                          # Pre-computed result CSVs (see results/README.md)
│   ├── causaltime_10seed/                # 10-seed CausalTime trust validation
│   ├── nonlinear/                        # Nonlinear regime characterization
│   ├── scale/                            # d=20/50/100 with baselines
│   ├── cross_sectional/                  # NOTEARS/DAGMA comparison
│   ├── ablation/, community_mixing/, significance/
├── assets/                           # Figures for README
├── data/                             # Dataset directory (README.md inside)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Python 3.10+ with PyTorch:

```bash
pip install -r requirements.txt
# Optional baselines:
pip install tigramite   # PCMCI+
pip install lingam      # VARLiNGAM
pip install anthropic   # For exp9 LLM pipeline (live API calls; not required if using cached priors)
```

## Quick Start

```python
import sys, numpy as np
sys.path.insert(0, "src")
from model_linear_trust import PRCD_MAP_Trust, train_prcd_trust_alm
from utils_trust import run_prcd_trust

# Your time series: (T, d) standardized; prior matrix P_prior in [0,1]^{d×d}
X = np.random.randn(500, 20)
P_prior = np.random.uniform(0, 1, (20, 20))
np.fill_diagonal(P_prior, 0.0)

W0, Wk, tau = run_prcd_trust(
    X, P_prior, d=20, K=1,
    lambda1=0.001, lambda2=0.01,
    max_iter=35, inner_iter=400, lr=8e-3, seed=0)
# W0: (d, d) instantaneous graph; Wk: list of lag matrices; tau: mean learned trust
```

## Reproducing Paper Results

### Core results (Tables 1–3)
```bash
cd experiments/
python exp1_synthetic_benchmark.py --sub sample_size --seeds 0 1 2     # Table 1
python exp7_real_benchmarks_trust.py --bench causaltime --seeds 0 1 2  # Table 2
python exp3_ablation.py --seeds 0 1 2                                  # Table 3
```

### Trust propagation validation (Table 7, new in NeurIPS version)
```bash
python exp10_community_mixing.py --variant v1 --seeds 0 1 2   # BA d=20, main designed validation
python exp10_community_mixing.py --variant v2 --seeds 0 1 2   # BA d=30, scale
python exp10_community_mixing.py --variant v3 --seeds 0 1 2   # ER negative control
python exp10_community_mixing.py --variant v4 --seeds 0 1 2   # Extreme heterogeneity
```

### Appendix experiments
```bash
python exp6_trust_validation.py --sub prior --seeds 0 1 2      # Table 8
python exp6_trust_validation.py --sub nonlinear --seeds 0 1 2  # Nonlinear validation
python exp8_scalability_trust.py --sub scale --seeds 0 1 2     # Scalability (App G)
python exp5_cross_sectional.py                                 # Cross-sectional (App K)
python exp11_significance_test.py --seeds 0 1 2 3 4 5 6 7 8 9  # 10-seed paired test
python exp12_theory_verification.py                            # Numerical theorem check
```

### LLM prior pipeline (App B)
```bash
# 1. Generate cached priors (uses domain templates; no API key required)
python ../data_loaders/generate_llm_priors.py
# 2. Run end-to-end pipeline
python exp9_llm_prior_pipeline.py --dataset AQI --seeds 0 1 2
```

## Data

- **Synthetic data**: Generated on the fly (ER/BA graphs, SVAR simulation).
- **CausalTime**: Download from the public CausalTime benchmark (MIT license); place at `data/causaltime/{AQI,Traffic,Medical}/`.
- **Electricity**: Sector-level monthly consumption data from a national electricity council's statistical yearbook; subject to data-sharing policy, available upon request for review purposes.

## Paper-to-Code Map

| Paper | Code | Pre-computed CSV |
|---|---|---|
| §3.2 Eq. (1)–(7) MAP objective | `src/model_linear.py` | — |
| §3.2 Eq. (8) Trust propagation | `src/trust_propagation.py`, `src/model_linear_trust.py` | — |
| §3.1 NAM extension (App F) | `src/model_nam.py`, `src/model_nam_trust.py` | — |
| §3.3 Empirical Bayes | `train_prcd_alm` (linear) / `train_prcd_trust_alm` (trust) | — |
| §4.2 Asymmetric robustness (Table 1) | `experiments/exp1_synthetic_benchmark.py` | — |
| §4.3 CausalTime (Table 2) | `experiments/exp7_real_benchmarks_trust.py` | `results/causaltime_10seed/` |
| §4.4 Ablation (Table 4) | `experiments/exp3_ablation.py` | `results/ablation/` |
| §4.4 Community Mixing (Table 3) | `experiments/exp10_community_mixing.py` | `results/community_mixing/` |
| Sec. 4 nonlinear PCMCI+ trade-off | `experiments/exp1_synthetic_benchmark.py --sub nonlinear` | `results/nonlinear/` |
| App "Main-text Scalability" ($d{\in}\{20,50,100\}$) | `experiments/exp1_synthetic_benchmark.py --sub scale` | `results/scale/` |
| App "Cross-Sectional Structure Learning" | `experiments/exp5_cross_sectional.py` | `results/cross_sectional/` |
| App "10-seed Trust Validation" on CausalTime | `experiments/exp7_real_benchmarks_trust.py` | `results/causaltime_10seed/` |
| App G Scalability | `experiments/exp8_scalability_trust.py` | — |
| App L Significance test | `experiments/exp11_significance_test.py` | `results/significance/` |
| Numerical theorem verification | `experiments/exp12_theory_verification.py` | — |

## Hardware

Tested on NVIDIA RTX 2080 Ti (11 GB). PRCD-MAP with trust propagation completes $d{=}100$ in $1.7$ s ($\sim\!6000{\times}$ faster than PCMCI+). NAM variant requires $d \leq 10$.

## Citation

```bibtex
@inproceedings{anon2026prcd,
  title={Learning How Much to Trust Domain Priors for Causal Structure Discovery},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
