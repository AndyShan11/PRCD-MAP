"""
W6a verification: PRCD-MAP at d=100 with FULL inner loop (no ALM early termination).

Goal: determine whether the Table 23 entry "1.7s, AUROC=0.617" reflects (a) a
broken d=100 run that hit ALM early-termination at iter 1-2, or (b) a genuine
end-to-end fast convergence whose AUROC is the real number.

This script forces tol=0 so the ALM outer loop always runs to max_iter=35.
Compare wall-clock and AUROC to the reported (1.7s, 0.617).

Usage:
    python verify_w6a_d100.py --seeds 0 1 2 3 4
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
sys.path.insert(0, SRC_DIR)

from utils import (set_seed, make_er_dag, make_lag_matrices, simulate_svar_linear,
                   standardize, gen_prior, compute_dual_metrics)
from utils_trust import make_lag_tensors_with_mask
from model_linear_trust import PRCD_MAP_Trust, train_prcd_trust_alm


def run_full_iter(X, P_prior, d, K, seed, trust_lite=True,
                  max_iter=35, inner_iter=400, lr=1e-2,
                  lambda1=0.001, lambda2=0.01, tol=0.0):
    """Same as run_prcd_trust but with tol explicitly exposed (default 0 = no early stop)."""
    set_seed(seed)
    X_t, X_lags, obs_mask = make_lag_tensors_with_mask(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]
    if obs_mask is not None:
        obs_mask = obs_mask.to(dev)

    model = PRCD_MAP_Trust(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=True, tau0=1.0,
        tau_min=0.05, tau_max=3.0,
        loss_type="huber", prior_l1_weight=True,
        n_tau_groups=4,
        trust_feat_dim=16, trust_n_layers=2,
        trust_lite=trust_lite,
    ).to(dev)

    W0, Wk, tau = train_prcd_trust_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=tol,
        verbose=True, postprocess=False,
        obs_mask=obs_mask,
    )
    return W0, Wk, tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--acc", type=float, default=0.6)
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--out", type=str, default="results/w6a_d100_verify.csv")
    args = parser.parse_args()

    rows = []
    for tol_setting in [0.0, 1e-6]:  # 0.0 = forced full; 1e-6 = current default (early-term enabled)
        for seed in args.seeds:
            print(f"\n=== tol={tol_setting}, seed={seed}, d={args.d}, T={args.T}, acc={args.acc} ===",
                  flush=True)
            set_seed(seed)
            W0_true = make_er_dag(args.d, edge_prob=0.15, seed=seed)
            Wk_true = make_lag_matrices(args.d, args.K, edge_prob=0.10, seed=seed)
            X = simulate_svar_linear(args.T, W0_true, Wk_true, seed=seed)
            if X is None:
                print("  [SKIP] sim failed", flush=True)
                continue
            X = standardize(X)
            P_prior = gen_prior(W0_true, Wk_true, acc=args.acc, seed=seed)

            t0 = time.time()
            W0, Wk, tau = run_full_iter(
                X, P_prior, args.d, args.K, seed,
                trust_lite=(args.d > 50),
                max_iter=args.max_iter, inner_iter=args.inner_iter,
                tol=tol_setting,
            )
            dt = time.time() - t0
            met = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
            row = {"seed": seed, "tol": tol_setting, "d": args.d, "T": args.T,
                   "max_iter": args.max_iter, "inner_iter": args.inner_iter,
                   "wall_clock_s": dt, **met}
            rows.append(row)
            print(f"  -> {dt:.2f}s, AUROC={met['auroc']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved -> {args.out}")
    print("\n=== Summary ===")
    for tol_setting in df["tol"].unique():
        sub = df[df["tol"] == tol_setting]
        print(f"tol={tol_setting}: wall-clock={sub['wall_clock_s'].mean():.1f}±{sub['wall_clock_s'].std():.1f}s, "
              f"AUROC={sub['auroc'].mean():.4f}±{sub['auroc'].std():.4f} (n={len(sub)})")


if __name__ == "__main__":
    main()
