"""
E-5 verification: unified "No Prior" baseline across all settings used in
Tables 5/9/12/15.

PAT flagged that the prior-free PRCD-MAP variant gave 4 different AUROC
numbers across 4 tables at the same nominal setting (d=20, T=500):
  Table 5 ("NoPrior"):  0.820
  Table 9 ("uniform"):  0.859
  Table 12 ("no prior"):0.826
  Table 15 ("no prior"):0.884 (this one is averaged over accs)

The variation comes from naming/pipeline differences:
  - "NoPrior" / "no prior" = tau=tau_min (effectively, learn_tau=False, tau0=tau_min)
  - "uniform"              = P_prior=0.5 fed to live EB pipeline, learned tau

This script re-runs both interpretations on a unified seed list (10 seeds)
at d=20, T=500, ER graph, Gaussian noise, acc=0.6 (the canonical setting
that all four tables share). It then re-averages over accs to match Table 15.

Usage:
    python verify_e5_noprior_canonical.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
sys.path.insert(0, SRC_DIR)

from utils import (set_seed, make_er_dag, make_lag_matrices, simulate_svar_linear,
                   standardize, gen_prior, compute_dual_metrics)
from utils_trust import run_prcd_trust


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--accs", type=float, nargs="+",
                        default=[0.4, 0.6, 0.9])  # for Table 15 averaging
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--out", type=str, default="results/e5_noprior_canonical.csv")
    args = parser.parse_args()

    rows = []
    for acc in args.accs:
        for seed in args.seeds:
            set_seed(seed)
            W0_true = make_er_dag(args.d, edge_prob=0.15, seed=seed)
            Wk_true = make_lag_matrices(args.d, args.K, edge_prob=0.10, seed=seed)
            X = simulate_svar_linear(args.T, W0_true, Wk_true, seed=seed)
            if X is None:
                continue
            X = standardize(X)
            P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)
            P_uniform = np.full((args.d, args.d), 0.5)
            np.fill_diagonal(P_uniform, 0.0)

            # Variant A: tau=tau_min ("NoPrior" in Tables 5, 12)
            try:
                t0 = time.time()
                W0_a, Wk_a, _ = run_prcd_trust(
                    X, P_prior, args.d, args.K,
                    lambda1=args.lambda1, lambda2=args.lambda2,
                    max_iter=args.max_iter, inner_iter=args.inner_iter,
                    lr=args.lr, seed=seed, learn_tau=False, tau0=0.05)
                dt_a = time.time() - t0
                met_a = compute_dual_metrics(W0_true, Wk_true, W0_a, Wk_a)
                rows.append({"variant": "tau_min", "acc": acc, "seed": seed,
                             "runtime": dt_a, **met_a})
                print(f"[acc={acc} s{seed}] tau_min:   AUROC={met_a['auroc']:.4f} ({dt_a:.1f}s)", flush=True)
            except Exception as e:
                warnings.warn(f"tau_min failed: {e}")

            # Variant B: uniform prior P=0.5 fed to learned-tau pipeline ("uniform" in Table 9)
            try:
                t0 = time.time()
                W0_b, Wk_b, _ = run_prcd_trust(
                    X, P_uniform, args.d, args.K,
                    lambda1=args.lambda1, lambda2=args.lambda2,
                    max_iter=args.max_iter, inner_iter=args.inner_iter,
                    lr=args.lr, seed=seed, learn_tau=True)
                dt_b = time.time() - t0
                met_b = compute_dual_metrics(W0_true, Wk_true, W0_b, Wk_b)
                rows.append({"variant": "uniform_learned", "acc": acc, "seed": seed,
                             "runtime": dt_b, **met_b})
                print(f"[acc={acc} s{seed}] uniform:   AUROC={met_b['auroc']:.4f} ({dt_b:.1f}s)", flush=True)
            except Exception as e:
                warnings.warn(f"uniform failed: {e}")

            # Variant C: tau=1 fixed, with noise prior — "FixedTau" in Table 5
            try:
                t0 = time.time()
                W0_c, Wk_c, _ = run_prcd_trust(
                    X, P_prior, args.d, args.K,
                    lambda1=args.lambda1, lambda2=args.lambda2,
                    max_iter=args.max_iter, inner_iter=args.inner_iter,
                    lr=args.lr, seed=seed, learn_tau=False, tau0=1.0)
                dt_c = time.time() - t0
                met_c = compute_dual_metrics(W0_true, Wk_true, W0_c, Wk_c)
                rows.append({"variant": "tau_1_fixed", "acc": acc, "seed": seed,
                             "runtime": dt_c, **met_c})
                print(f"[acc={acc} s{seed}] tau=1:     AUROC={met_c['auroc']:.4f} ({dt_c:.1f}s)", flush=True)
            except Exception as e:
                warnings.warn(f"tau1 failed: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {args.out}")

    if len(df):
        print("\n=== Summary at d={}, T={} ===".format(args.d, args.T))
        for variant in df["variant"].unique():
            for acc in df["acc"].unique():
                sub = df[(df["variant"] == variant) & (df["acc"] == acc)]["auroc"].dropna()
                if len(sub):
                    print(f"  {variant:20s} acc={acc:.1f}: AUROC = {sub.mean():.4f} ± {sub.std():.4f}  (n={len(sub)})")
        print("\n=== Table-15-style (averaged over accs) ===")
        for variant in df["variant"].unique():
            sub = df[df["variant"] == variant]["auroc"].dropna()
            print(f"  {variant:20s}: AUROC = {sub.mean():.4f} ± {sub.std():.4f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
