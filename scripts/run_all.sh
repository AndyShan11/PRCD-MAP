#!/bin/bash
# =============================================================================
# Run all PRCD-MAP experiments for paper reproduction.
#
# Usage:
#   bash scripts/run_all.sh          # run all experiments sequentially
#   bash scripts/run_all.sh exp1     # run only Experiment 1
#   bash scripts/run_all.sh exp2     # run only Experiment 2
#   ...
#
# Prerequisites:
#   pip install -r requirements.txt
#   (Optional) pip install tigramite lingam   # for baselines
#
# Hardware: Tested on NVIDIA RTX 2080 Ti (11 GB). CPU-only mode is supported
#           but significantly slower for d >= 50.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

run_exp1() {
    echo "============================================="
    echo " Experiment 1: Synthetic Benchmark"
    echo "============================================="
    python experiments/exp1_synthetic.py
}

run_exp2() {
    echo "============================================="
    echo " Experiment 2: Real-World Benchmarks"
    echo "============================================="
    python experiments/exp2_real.py
}

run_exp3() {
    echo "============================================="
    echo " Experiment 3: Ablation Study"
    echo "============================================="
    python experiments/exp3_ablation.py
}

run_exp4() {
    echo "============================================="
    echo " Experiment 4: Scalability & Sensitivity"
    echo "============================================="
    python experiments/exp4_scalability.py
}

run_figures() {
    echo "============================================="
    echo " Generating paper figures & tables"
    echo "============================================="
    python experiments/generate_figures.py
}

# Dispatch
if [ $# -eq 0 ]; then
    run_exp1
    run_exp2
    run_exp3
    run_exp4
    run_figures
    echo ""
    echo "All experiments completed. Results saved to results/"
else
    case "$1" in
        exp1) run_exp1 ;;
        exp2) run_exp2 ;;
        exp3) run_exp3 ;;
        exp4) run_exp4 ;;
        figures) run_figures ;;
        *) echo "Unknown experiment: $1. Use: exp1, exp2, exp3, exp4, figures" ;;
    esac
fi
