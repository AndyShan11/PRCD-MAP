#!/bin/bash
# =============================================================================
# PRCD-MAP: Reproduce paper experiments.
#
# Usage:
#   bash scripts/run_all.sh           # all experiments sequentially
#   bash scripts/run_all.sh core      # Tables 1-3 only
#   bash scripts/run_all.sh trust     # Table 7 + App L (trust propagation)
#   bash scripts/run_all.sh llm       # App B LLM pipeline
#   bash scripts/run_all.sh appendix  # All appendix experiments
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/experiments"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

run_core() {
    echo "==== Table 1: Synthetic SVAR ===="
    python exp1_synthetic_benchmark.py --sub sample_size --seeds 0 1 2
    echo "==== Table 2: CausalTime ===="
    python exp7_real_benchmarks_trust.py --bench causaltime --seeds 0 1 2
    echo "==== Table 3: Ablation ===="
    python exp3_ablation.py --seeds 0 1 2
}

run_trust() {
    echo "==== Table 7: Community Mixing (main designed validation) ===="
    python exp10_community_mixing.py --variant v1 --seeds 0 1 2
    python exp10_community_mixing.py --variant v2 --seeds 0 1 2
    echo "==== Negative controls (Appendix) ===="
    python exp10_community_mixing.py --variant v3 --seeds 0 1 2
    python exp10_community_mixing.py --variant v4 --seeds 0 1 2
    echo "==== Table 8 + App L: Trust validation + significance ===="
    python exp6_trust_validation.py --sub prior --seeds 0 1 2
    python exp11_significance_test.py --seeds 0 1 2 3 4 5 6 7 8 9 --accs 0.8 1.0
}

run_llm() {
    echo "==== Generate cached LLM priors ===="
    python ../data_loaders/generate_llm_priors.py
    echo "==== App B: LLM pipeline ===="
    for ds in AQI Traffic Medical Electricity; do
        python exp9_llm_prior_pipeline.py --dataset $ds --seeds 0 1 2
    done
}

run_appendix() {
    echo "==== App F: Nonlinear validation ===="
    python exp6_trust_validation.py --sub nonlinear --seeds 0 1 2
    echo "==== App G: Scalability ===="
    python exp4_scalability.py
    python exp8_scalability_trust.py --sub scale --seeds 0 1 2
    echo "==== App K: Cross-sectional ===="
    python exp5_cross_sectional.py
    echo "==== Numerical theorem verification ===="
    python exp12_theory_verification.py
}

case "${1:-all}" in
    core) run_core ;;
    trust) run_trust ;;
    llm) run_llm ;;
    appendix) run_appendix ;;
    all) run_core; run_trust; run_llm; run_appendix ;;
    *) echo "Usage: bash $0 {core|trust|llm|appendix|all}"; exit 1 ;;
esac

echo ""
echo "Done. Results in experiments/*_results/ and logs in logs/."
