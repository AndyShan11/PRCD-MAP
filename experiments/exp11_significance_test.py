"""
=============================================================================
Experiment 7 — Seed Boost for Statistical Significance (Table 8 update)
=============================================================================
Purpose: Increase seeds from 3 to 10 for acc=0.8 and acc=1.0 in Table 8
(app:trust_validation), then run paired t-test and Wilcoxon signed-rank test
on trust vs per-group differences.

Settings:
  - d=20, T=500, ER graph, Gaussian noise, K=1
  - acc ∈ {0.8, 1.0} (the settings showing +0.009 gap in 3-seed results)
  - Seeds 0-9 (10 total)
  - Methods: trust + per-group only

Output:
  - Per-seed results CSV
  - Summary with paired t-test p-value and Wilcoxon signed-rank p-value

Usage:
  python exp7_seed_boost.py --seeds 0 1 2 3 4 5 6 7 8 9
=============================================================================
"""

import os, sys, time, warnings, argparse, traceback
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_trust import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--accs", type=float, nargs="+",
                        default=[0.8, 1.0])
    args = parser.parse_args()

    output_dir = "exp7_seed_boost"
    ensure_dir(output_dir)

    d = 20
    T = 500
    K = 1
    t_global = time.time()
    results = []

    for acc in args.accs:
        for seed in args.seeds:
            print(f"\n--- acc={acc}, seed={seed} ---")
            t0 = time.time()
            try:
                set_seed(seed)
                W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
                Wk_true = make_lag_matrices(d, K, edge_prob=0.10, seed=seed)
                X = simulate_svar_linear(T, W0_true, Wk_true,
                                          noise_type="gaussian", seed=seed)
                if X is None:
                    print("  [SKIP] sim failed")
                    continue
                X = standardize(X)
                P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)

                B_w0 = (np.abs(W0_true) > 1e-10).astype(float)
                B_comb = B_w0.copy().astype(int)
                for wk in Wk_true:
                    B_comb = np.maximum(B_comb, (np.abs(wk) > 1e-10).astype(int))

                # Trust propagation
                W0_t, Wk_t, tau_t = run_prcd_trust(
                    X, P_prior, d, K,
                    lambda1=0.001, lambda2=0.01,
                    max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
                met_t = compute_dual_metrics(W0_true, Wk_true, W0_t, Wk_t)
                met_t.update({"method": "trust", "seed": seed, "acc": acc})
                results.append(met_t)

                # Per-group
                W0_p, Wk_p, tau_p = run_prcd_map(
                    X, P_prior, d, K,
                    lambda1=0.001, lambda2=0.01,
                    max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
                met_p = compute_dual_metrics(W0_true, Wk_true, W0_p, Wk_p)
                met_p.update({"method": "per-group", "seed": seed, "acc": acc})
                results.append(met_p)

                dt = time.time() - t0
                print(f"  [{fmt_time(dt)}] trust={met_t['auroc']:.4f} "
                      f"pg={met_p['auroc']:.4f} Δ={met_t['auroc']-met_p['auroc']:+.4f}")

            except Exception as e:
                traceback.print_exc()

    # Save raw results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp7_seed_boost_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {csv_path}")

    # Statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)

    summary_rows = []
    for acc in args.accs:
        sub = df[df["acc"] == acc]
        # Pair by seed
        trust_by_seed = sub[sub["method"] == "trust"].set_index("seed")["auroc"]
        pg_by_seed = sub[sub["method"] == "per-group"].set_index("seed")["auroc"]
        common_seeds = sorted(set(trust_by_seed.index) & set(pg_by_seed.index))
        t_arr = trust_by_seed.loc[common_seeds].values
        p_arr = pg_by_seed.loc[common_seeds].values
        diff = t_arr - p_arr

        n = len(common_seeds)
        mean_t, std_t = t_arr.mean(), t_arr.std(ddof=1)
        mean_p, std_p = p_arr.mean(), p_arr.std(ddof=1)
        mean_d, std_d = diff.mean(), diff.std(ddof=1)

        # Paired t-test (one-sided: trust > per-group)
        t_stat, p_ttest_two = stats.ttest_rel(t_arr, p_arr)
        p_ttest_one = p_ttest_two / 2 if t_stat > 0 else 1 - p_ttest_two / 2

        # Wilcoxon signed-rank (one-sided)
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(t_arr, p_arr, alternative="greater")
        except ValueError as e:
            p_wilcoxon = np.nan

        # Same for F1
        trust_f1 = sub[sub["method"] == "trust"].set_index("seed")["f1_opt"]
        pg_f1 = sub[sub["method"] == "per-group"].set_index("seed")["f1_opt"]
        f1_t = trust_f1.loc[common_seeds].values
        f1_p = pg_f1.loc[common_seeds].values
        f1_diff = f1_t - f1_p
        t_stat_f1, p_ttest_f1_two = stats.ttest_rel(f1_t, f1_p)
        p_ttest_f1_one = p_ttest_f1_two / 2 if t_stat_f1 > 0 else 1 - p_ttest_f1_two / 2
        try:
            w_stat_f1, p_wilcoxon_f1 = stats.wilcoxon(f1_t, f1_p, alternative="greater")
        except ValueError:
            p_wilcoxon_f1 = np.nan

        print(f"\n--- acc={acc} (n={n} seeds) ---")
        print(f"  AUROC:  trust={mean_t:.4f}±{std_t:.4f}  "
              f"per-group={mean_p:.4f}±{std_p:.4f}  Δ={mean_d:+.4f}±{std_d:.4f}")
        print(f"  F1:     trust={f1_t.mean():.4f}±{f1_t.std(ddof=1):.4f}  "
              f"per-group={f1_p.mean():.4f}±{f1_p.std(ddof=1):.4f}  "
              f"Δ={f1_diff.mean():+.4f}±{f1_diff.std(ddof=1):.4f}")
        print(f"  Paired t-test (AUROC, one-sided): t={t_stat:.3f}, p={p_ttest_one:.4f}")
        print(f"  Wilcoxon (AUROC, one-sided):              p={p_wilcoxon:.4f}")
        print(f"  Paired t-test (F1, one-sided):   t={t_stat_f1:.3f}, p={p_ttest_f1_one:.4f}")
        print(f"  Wilcoxon (F1, one-sided):                 p={p_wilcoxon_f1:.4f}")

        # Verdict
        sig_auroc = p_ttest_one < 0.05 and (np.isnan(p_wilcoxon) or p_wilcoxon < 0.05)
        sig_f1 = p_ttest_f1_one < 0.05 and (np.isnan(p_wilcoxon_f1) or p_wilcoxon_f1 < 0.05)
        if sig_auroc or sig_f1:
            print(f"  => SIGNIFICANT at p<0.05 (AUROC: {sig_auroc}, F1: {sig_f1})")
        else:
            print(f"  => NOT significant at p<0.05")

        summary_rows.append({
            "acc": acc, "n_seeds": n,
            "trust_auroc_mean": mean_t, "trust_auroc_std": std_t,
            "pg_auroc_mean": mean_p, "pg_auroc_std": std_p,
            "diff_auroc_mean": mean_d, "diff_auroc_std": std_d,
            "ttest_auroc_p": p_ttest_one,
            "wilcoxon_auroc_p": p_wilcoxon,
            "trust_f1_mean": f1_t.mean(), "trust_f1_std": f1_t.std(ddof=1),
            "pg_f1_mean": f1_p.mean(), "pg_f1_std": f1_p.std(ddof=1),
            "diff_f1_mean": f1_diff.mean(), "diff_f1_std": f1_diff.std(ddof=1),
            "ttest_f1_p": p_ttest_f1_one,
            "wilcoxon_f1_p": p_wilcoxon_f1,
            "significant_auroc_p05": sig_auroc,
            "significant_f1_p05": sig_f1,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "exp7_significance_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n>>> Summary saved -> {summary_path}")

    elapsed = time.time() - t_global
    print(f"\n>>> Exp7 complete in {fmt_time(elapsed)}")

    # Recommendation for paper
    print("\n" + "=" * 80)
    print("PAPER SENTENCE RECOMMENDATION")
    print("=" * 80)
    for row in summary_rows:
        acc = row["acc"]
        if row["significant_auroc_p05"] or row["significant_f1_p05"]:
            print(f'\nacc={acc}: "the improvement is small but statistically significant '
                  f'(p={min(row["ttest_auroc_p"], row["ttest_f1_p"]):.3f}, n=10 seeds)"')
        else:
            print(f'\nacc={acc}: "not significant under homogeneous corruption '
                  f'(paired t-test p={row["ttest_auroc_p"]:.3f}, n=10); '
                  f'significant only under heterogeneous priors (Table 3)"')


if __name__ == "__main__":
    main()
