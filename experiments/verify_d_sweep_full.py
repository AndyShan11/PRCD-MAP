"""
Full d-sweep verification: combines W6a (d=100 tol=0) + Tab 9 PRCD-MAP claims.

Sweeps d in {20, 50, 100, 150, 200} (skip 300 to fit 11.3GB GPU) and tol in
{0.0, 1e-6} for 3 seeds each. For each (d, tol, seed) records:
  - Wall-clock
  - Final |h| (was early-termination triggered?)
  - AUROC

This produces a definitive empirical answer to:
  (1) Is the d=100 1.7s entry in Table 23 an early-termination artifact?
  (2) Does the AUROC at d=100 (reported 0.617) recover when forced to full
      iterations?
  (3) How does the d-sweep AUROC compare to the values in tab:app_scale_main?

Usage:
    python verify_d_sweep_full.py --seeds 0 1 2
"""
import os, sys, time, argparse, warnings
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


def run_with_tol(X, P_prior, d, K, seed, tol, trust_lite,
                 max_iter=35, inner_iter=400, lr=1e-2,
                 lambda1=0.001, lambda2=0.01):
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
        n_tau_groups=4, trust_feat_dim=16, trust_n_layers=2,
        trust_lite=trust_lite,
    ).to(dev)

    W0, Wk, tau = train_prcd_trust_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=tol,
        verbose=False, postprocess=False,
        obs_mask=obs_mask,
    )
    return W0, Wk, tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ds", type=int, nargs="+", default=[20, 50, 100, 150, 200])
    parser.add_argument("--tols", type=float, nargs="+", default=[0.0, 1e-6])
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--acc", type=float, default=0.6)
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--out", type=str, default="results/d_sweep_full.csv")
    args = parser.parse_args()

    rows = []
    for d in args.ds:
        for tol in args.tols:
            for seed in args.seeds:
                print(f"\n=== d={d} tol={tol} seed={seed} ===", flush=True)
                set_seed(seed)
                W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
                Wk_true = make_lag_matrices(d, args.K, edge_prob=0.10, seed=seed)
                X = simulate_svar_linear(args.T, W0_true, Wk_true, seed=seed)
                if X is None:
                    continue
                X = standardize(X)
                P_prior = gen_prior(W0_true, Wk_true, acc=args.acc, seed=seed)

                try:
                    t0 = time.time()
                    W0, Wk, tau = run_with_tol(
                        X, P_prior, d, args.K, seed, tol=tol,
                        trust_lite=(d > 50),
                        max_iter=args.max_iter, inner_iter=args.inner_iter)
                    dt = time.time() - t0
                    met = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
                    rows.append({"d": d, "tol": tol, "seed": seed,
                                 "T": args.T, "acc": args.acc,
                                 "runtime": dt, **met})
                    print(f"  -> {dt:.1f}s, AUROC={met['auroc']:.4f}", flush=True)
                except Exception as e:
                    warnings.warn(f"d={d} tol={tol} s{seed} failed: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {args.out}")

    if len(df):
        print("\n=== Wall-clock by (d, tol) ===")
        piv = df.pivot_table(values="runtime", index="d", columns="tol", aggfunc="mean")
        print(piv.to_string(float_format="%.1f"))
        print("\n=== AUROC by (d, tol) ===")
        piv = df.pivot_table(values="auroc", index="d", columns="tol", aggfunc="mean")
        print(piv.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
