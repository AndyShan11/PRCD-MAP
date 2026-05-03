"""
E-4 verification: M2 (EB calibration) ablation directly on CausalTime.

Paper Table 3 estimates M2 = +0.020 from a synthetic learned-vs-fixed gap.
PAT flagged this as methodologically invalid: M2 should be measured directly
on the target CausalTime datasets.

This script does that. For each (dataset, prior_idx, seed):
    - PRCD-MAP(trust, learned tau) — current paper main
    - PRCD-MAP(trust, fixed tau=1) — same MLP architecture but no EB calibration
The per-cell M2 is the AUROC difference; aggregated over priors and seeds.

Usage:
    python verify_e4_m2_causaltime.py --dataset all --seeds 0 1 2 3 4
"""
import os, sys, time, argparse, warnings
from typing import List
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils import (set_seed, standardize, compute_dual_metrics)
from utils_trust import run_prcd_trust
# Use the same cached LLM priors as exp9
from exp9_llm_prior_pipeline import (DATASET_DESCRIPTIONS,
                                      generate_llm_priors)


def load_causaltime(causaltime_dir: str, dataset_name: str, n_samples: int = 10):
    """Load a CausalTime dataset (X, B_true)."""
    from utils_trust import load_causaltime as _loader
    return _loader(causaltime_dir, dataset_name, n_samples=n_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["AQI", "Traffic", "Medical", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--causaltime_dir", type=str, default="./data/causaltime")
    parser.add_argument("--cache_dir", type=str, default="llm_prior_cache")
    parser.add_argument("--n_priors", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--out", type=str, default="results/e4_m2_causaltime.csv")
    args = parser.parse_args()

    datasets = (["AQI", "Traffic", "Medical"]
                if args.dataset == "all" else [args.dataset])

    rows = []
    for ds_name in datasets:
        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}", flush=True)
        X, B_true = load_causaltime(args.causaltime_dir, ds_name, n_samples=10)
        if X is None:
            print(f"  [SKIP] {ds_name} not found")
            continue
        d = X.shape[1]
        var_names = [f"var_{i}" for i in range(d)]
        desc = DATASET_DESCRIPTIONS.get(ds_name, {})
        if desc.get("variables"):
            var_names = list(desc["variables"].keys())[:d]

        # Use cached LLM priors (no API calls; raises if cache missing)
        priors = generate_llm_priors(ds_name, var_names, d,
                                     n_priors=args.n_priors,
                                     api_key=None, cache_dir=args.cache_dir)
        print(f"  Loaded {len(priors)} cached LLM priors for {ds_name}", flush=True)

        W0_true = B_true.astype(float)
        Wk_true = [np.zeros_like(W0_true)]

        for prior_idx, P_prior in enumerate(priors):
            for seed in args.seeds:
                # Variant A: learned tau (M2 + M3 active) — current paper main
                try:
                    t0 = time.time()
                    W0_a, Wk_a, _ = run_prcd_trust(
                        X, P_prior, d, K=1,
                        lambda1=args.lambda1, lambda2=args.lambda2,
                        max_iter=args.max_iter, inner_iter=args.inner_iter,
                        lr=args.lr, seed=seed, learn_tau=True)
                    dt_a = time.time() - t0
                    met_a = compute_dual_metrics(W0_true, Wk_true, W0_a, Wk_a)
                    rows.append({
                        "dataset": ds_name, "prior_idx": prior_idx, "seed": seed,
                        "variant": "learned_tau", "runtime": dt_a, **met_a,
                    })
                    print(f"  [{ds_name} p{prior_idx} s{seed}] learned: AUROC={met_a['auroc']:.4f} ({dt_a:.1f}s)",
                          flush=True)
                except Exception as e:
                    warnings.warn(f"learned failed [{ds_name} p{prior_idx} s{seed}]: {e}")

                # Variant B: fixed tau=1 (M3 active, M2 disabled) — direct M2 ablation
                try:
                    t0 = time.time()
                    W0_b, Wk_b, _ = run_prcd_trust(
                        X, P_prior, d, K=1,
                        lambda1=args.lambda1, lambda2=args.lambda2,
                        max_iter=args.max_iter, inner_iter=args.inner_iter,
                        lr=args.lr, seed=seed, learn_tau=False, tau0=1.0)
                    dt_b = time.time() - t0
                    met_b = compute_dual_metrics(W0_true, Wk_true, W0_b, Wk_b)
                    rows.append({
                        "dataset": ds_name, "prior_idx": prior_idx, "seed": seed,
                        "variant": "fixed_tau1", "runtime": dt_b, **met_b,
                    })
                    print(f"  [{ds_name} p{prior_idx} s{seed}] fixed:   AUROC={met_b['auroc']:.4f} ({dt_b:.1f}s)",
                          flush=True)
                except Exception as e:
                    warnings.warn(f"fixed failed [{ds_name} p{prior_idx} s{seed}]: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {args.out}")

    # Per-cell M2 = learned − fixed
    if len(df):
        print("\n=== Per-dataset M2 (learned − fixed_tau=1) ===")
        pivot = df.pivot_table(values="auroc",
                               index=["dataset", "prior_idx", "seed"],
                               columns="variant").reset_index()
        pivot["M2"] = pivot["learned_tau"] - pivot["fixed_tau1"]
        for ds_name in pivot["dataset"].unique():
            sub = pivot[pivot["dataset"] == ds_name]["M2"].dropna()
            print(f"  {ds_name:10s}: M2 = {sub.mean():+.4f} ± {sub.std():.4f}  (n={len(sub)})")
        overall = pivot["M2"].dropna()
        print(f"  {'OVERALL':10s}: M2 = {overall.mean():+.4f} ± {overall.std():.4f}  (n={len(overall)})")
        print(f"\n  (Paper Table 3 reports M2=+0.020 from synthetic transfer.)")


if __name__ == "__main__":
    main()
