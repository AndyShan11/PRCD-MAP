"""
=============================================================================
Experiment 15 — Table 1 extended: 10-seed re-run for T={100, 200, 500}
=============================================================================
Reviewer concern (W6): Table 1 cells at T=100/200/500 use only 3 seeds, while
some reported gaps (e.g. acc=0.6, T=500: 0.870 vs 0.872) are smaller than
joint seed std. This script adds 7 new seeds (3..9) per cell, complementing
the 10 seeds at T=50 already in exp13. The output csv is post-processed to
report mean±std and paired-t p-values for learned-τ vs fixed-τ.

Settings:
  - d=20, T ∈ {100, 200, 500}, K=1, ER graph, Gaussian noise
  - acc ∈ {0.4, 0.6, 0.9}
  - seeds 3..9 (default)  -- combine with old 0..2 to get 10 seeds total
  - methods: PRCD-MAP (learned-τ), PRCD-MAP (τ=1), PCMCI+, DYNOTEARS, VARLiNGAM

Usage:
  python exp15_table1_extended_seeds.py --Ts 100 200 500 \
      --seeds 3 4 5 6 7 8 9
=============================================================================
"""
import os, sys, time, argparse, traceback
import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from utils import (set_seed, make_er_dag, make_lag_matrices,
                   simulate_svar_linear, standardize, gen_prior,
                   compute_dual_metrics, run_prcd_map, run_dynotears,
                   run_pcmci_plus, run_varlingam, ensure_dir, fmt_time)


def _eval_one(X, P_prior, d, K, seed, W0_true, Wk_true, results, acc, T):
    def _save(name, W0, Wk):
        if W0 is None:
            return
        m = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
        m.update({"method": name, "seed": seed, "acc": acc, "T": T})
        results.append(m)

    # PRCD-MAP learned-τ
    try:
        W0, Wk, _ = run_prcd_map(
            X, P_prior, d, K,
            lambda1=0.001, lambda2=0.01,
            learn_tau=True,
            max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
        _save("PRCD-MAP(learned)", W0, Wk)
    except Exception:
        traceback.print_exc()

    # PRCD-MAP fixed τ=1
    try:
        W0, Wk, _ = run_prcd_map(
            X, P_prior, d, K,
            lambda1=0.001, lambda2=0.01,
            learn_tau=False, tau0=1.0,
            max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
        _save("PRCD-MAP(tau=1)", W0, Wk)
    except Exception:
        traceback.print_exc()

    # Baselines (prior-independent)
    for name, fn in [("DYNOTEARS", run_dynotears),
                     ("PCMCI+", run_pcmci_plus),
                     ("VARLiNGAM", run_varlingam)]:
        try:
            W0, Wk = fn(X, d, K, seed=seed)
            _save(name, W0, Wk)
        except Exception:
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--accs", type=float, nargs="+",
                        default=[0.4, 0.6, 0.9])
    parser.add_argument("--Ts", type=int, nargs="+",
                        default=[100, 200, 500])
    args = parser.parse_args()

    output_dir = "exp15_table1_extended"
    ensure_dir(output_dir)

    d, K = 20, 1
    results = []
    t_global = time.time()

    for T in args.Ts:
        for acc in args.accs:
            for seed in args.seeds:
                print(f"\n--- T={T}, acc={acc}, seed={seed} ---")
                t0 = time.time()
                try:
                    set_seed(seed)
                    W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
                    Wk_true = make_lag_matrices(d, K, edge_prob=0.10, seed=seed)
                    X = simulate_svar_linear(T, W0_true, Wk_true,
                                              noise_type="gaussian", seed=seed)
                    if X is None:
                        continue
                    X = standardize(X)
                    P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)
                    _eval_one(X, P_prior, d, K, seed,
                              W0_true, Wk_true, results, acc, T)
                    print(f"  [{fmt_time(time.time()-t0)}] done")
                except Exception:
                    traceback.print_exc()

                # Incremental save
                if results:
                    pd.DataFrame(results).to_csv(
                        os.path.join(output_dir, "_intermediate.csv"),
                        index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "exp15_results.csv"), index=False)

    # Summary: mean±std per (T, acc, method)
    print("\n" + "=" * 60)
    print("Summary (AUROC mean±std)")
    print("=" * 60)
    for T in sorted(df["T"].unique()):
        for acc in sorted(df["acc"].unique()):
            sub = df[(df["T"] == T) & (df["acc"] == acc)]
            print(f"\nT={T}, acc={acc}:")
            for method in sorted(sub["method"].unique()):
                row = sub[sub["method"] == method]
                if "auroc_combined" in row.columns:
                    vals = row["auroc_combined"].dropna().values
                else:
                    vals = row.get("auroc", row.get("AUROC", pd.Series([]))).dropna().values
                if len(vals) > 0:
                    print(f"  {method:24s}: {np.mean(vals):.3f}±{np.std(vals):.3f} "
                          f"(n={len(vals)})")

    # Paired-t learned vs fixed per cell
    print("\n" + "=" * 60)
    print("Paired-t: learned-τ vs τ=1 (n_seeds × acc per T)")
    print("=" * 60)
    try:
        from scipy import stats
        for T in sorted(df["T"].unique()):
            for acc in sorted(df["acc"].unique()):
                sub = df[(df["T"] == T) & (df["acc"] == acc)]
                lr = sub[sub["method"] == "PRCD-MAP(learned)"].sort_values("seed")
                fx = sub[sub["method"] == "PRCD-MAP(tau=1)"].sort_values("seed")
                col = "auroc_combined" if "auroc_combined" in lr.columns else "auroc"
                if col not in lr.columns or col not in fx.columns:
                    continue
                lv = lr[col].dropna().values
                fv = fx[col].dropna().values
                n = min(len(lv), len(fv))
                if n < 2:
                    continue
                t, p = stats.ttest_rel(lv[:n], fv[:n])
                d_mean = float(np.mean(lv[:n] - fv[:n]))
                print(f"  T={T}, acc={acc}: Δ={d_mean:+.3f}, "
                      f"t={t:.2f}, p={p:.3f} (n={n})")
    except Exception:
        traceback.print_exc()

    print(f"\n>>> Total time: {fmt_time(time.time() - t_global)}")
    print(f">>> Results: {output_dir}/exp15_results.csv")


if __name__ == "__main__":
    main()
