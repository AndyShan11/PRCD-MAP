"""
W3 verification: PRCD-MAP behavior in the weak-data regime T<<d.

PAT noted that the Laplace log-det term in the EB objective mathematically
rewards inflating prior variance when T<<d, potentially causing the EB
mechanism to incorrectly attenuate accurate priors.

This script characterizes the actual behavior across:
  T in {20, 50, 100, 200, 500}, d=20, acc in {0.4, 0.6, 0.9}, 5 seeds.

For each cell, we record:
  - Final tau* (groups + per-edge MLP output mean)
  - AUROC of learned-tau vs fixed-tau=1 vs no-prior
  - Whether tau collapses to tau_min when prior IS accurate (failure mode)

Usage:
    python verify_w3_weak_data.py --seeds 0 1 2 3 4
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
from utils_trust import run_prcd_trust


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--Ts", type=int, nargs="+",
                        default=[20, 50, 100, 200, 500])
    parser.add_argument("--accs", type=float, nargs="+",
                        default=[0.4, 0.6, 0.9])
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--out", type=str, default="results/w3_weak_data_regime.csv")
    args = parser.parse_args()

    rows = []
    for T in args.Ts:
        for acc in args.accs:
            for seed in args.seeds:
                print(f"\n=== T={T} acc={acc} seed={seed} ===", flush=True)
                set_seed(seed)
                W0_true = make_er_dag(args.d, edge_prob=0.15, seed=seed)
                Wk_true = make_lag_matrices(args.d, args.K, edge_prob=0.10, seed=seed)
                X = simulate_svar_linear(T, W0_true, Wk_true, seed=seed)
                if X is None:
                    print("  [SKIP] sim failed", flush=True)
                    continue
                X = standardize(X)
                P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)

                for variant_name, kwargs in [
                    ("learned_tau", dict(learn_tau=True)),
                    ("fixed_tau1",  dict(learn_tau=False, tau0=1.0)),
                    ("no_prior",    dict(learn_tau=False, tau0=0.05)),
                ]:
                    try:
                        t0 = time.time()
                        W0, Wk, tau = run_prcd_trust(
                            X, P_prior, args.d, args.K,
                            max_iter=args.max_iter, inner_iter=args.inner_iter,
                            seed=seed, **kwargs)
                        dt = time.time() - t0
                        met = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
                        try:
                            tau_arr = np.asarray(tau).ravel()
                            tau_mean = float(tau_arr.mean()) if tau_arr.size else float(tau) if np.isscalar(tau) else float('nan')
                            tau_min  = float(tau_arr.min())  if tau_arr.size else float('nan')
                            tau_max  = float(tau_arr.max())  if tau_arr.size else float('nan')
                        except Exception:
                            tau_mean = tau_min = tau_max = float('nan')
                        rows.append({
                            "variant": variant_name, "T": T, "acc": acc, "seed": seed,
                            "d": args.d, "runtime": dt,
                            "tau_mean": tau_mean, "tau_min": tau_min, "tau_max": tau_max,
                            **met,
                        })
                        print(f"  {variant_name:12s}: AUROC={met['auroc']:.4f}, "
                              f"tau_mean={tau_mean:.3f}  ({dt:.1f}s)", flush=True)
                    except Exception as e:
                        warnings.warn(f"{variant_name} failed: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {args.out}")

    if len(df):
        print("\n=== Summary: AUROC by (variant, T, acc) ===")
        for variant in df["variant"].unique():
            print(f"\n  {variant}:")
            sub = df[df["variant"] == variant]
            piv = sub.pivot_table(values="auroc", index="T", columns="acc", aggfunc="mean")
            print(piv.to_string(float_format="%.4f"))

        print("\n=== tau_mean by (variant, T, acc) — diagnostic for W3 collapse ===")
        for variant in df["variant"].unique():
            sub = df[df["variant"] == variant]
            piv = sub.pivot_table(values="tau_mean", index="T", columns="acc", aggfunc="mean")
            print(f"\n  {variant}:")
            print(piv.to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()
