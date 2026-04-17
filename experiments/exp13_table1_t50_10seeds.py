"""
=============================================================================
Experiment 13 — Table 1 Row T=50: 10-seed strengthening for headline claim
=============================================================================
The paper's headline claim "+0.158 AUROC over best baseline at T=50, acc=0.9"
was computed with 3 seeds. This script re-runs ONLY that row (T=50) with 10
seeds to tighten confidence intervals.

Settings:
  - d=20, T=50, K=1, ER graph, Gaussian noise
  - acc ∈ {0.4, 0.6, 0.9}
  - seeds 0..9
  - methods: PRCD-MAP (learned-τ), PRCD-MAP (τ=1), PCMCI+, DYNOTEARS, VARLiNGAM

Usage:
  python exp13_table1_t50_10seeds.py --seeds 0 1 2 3 4 5 6 7 8 9
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(10)))
    parser.add_argument("--accs", type=float, nargs="+",
                        default=[0.4, 0.6, 0.9])
    args = parser.parse_args()

    output_dir = "exp13_table1_10seeds"
    ensure_dir(output_dir)

    d, T, K = 20, 50, 1
    results = []
    t_global = time.time()

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
                    continue
                X = standardize(X)
                P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)

                def _eval(name, W0, Wk):
                    if W0 is None: return
                    m = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
                    m.update({"method": name, "seed": seed, "acc": acc})
                    results.append(m)

                # PRCD-MAP learned-τ
                W0, Wk, tau = run_prcd_map(
                    X, P_prior, d, K,
                    lambda1=0.001, lambda2=0.01,
                    learn_tau=True,
                    max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
                _eval("PRCD-MAP(learned)", W0, Wk)

                # PRCD-MAP fixed τ=1
                W0, Wk, tau = run_prcd_map(
                    X, P_prior, d, K,
                    lambda1=0.001, lambda2=0.01,
                    learn_tau=False, tau0=1.0,
                    max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
                _eval("PRCD-MAP(tau=1)", W0, Wk)

                # Baselines (prior-independent)
                try:
                    W0, Wk = run_dynotears(X, d, K, seed=seed)
                    _eval("DYNOTEARS", W0, Wk)
                except Exception: pass
                try:
                    W0, Wk = run_pcmci_plus(X, d, K, seed=seed)
                    _eval("PCMCI+", W0, Wk)
                except Exception: pass
                try:
                    W0, Wk = run_varlingam(X, d, K, seed=seed)
                    _eval("VARLiNGAM", W0, Wk)
                except Exception: pass

                dt = time.time() - t0
                print(f"  [{fmt_time(dt)}] done")
            except Exception as e:
                traceback.print_exc()

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp13_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {csv_path}")

    print("\n" + "=" * 70)
    print("TABLE 1 ROW T=50 — 10-SEED REFRESH")
    print("=" * 70)
    for acc in args.accs:
        sub = df[df["acc"] == acc]
        print(f"\nacc={acc}:")
        for m in ["PRCD-MAP(learned)", "PRCD-MAP(tau=1)",
                  "PCMCI+", "DYNOTEARS", "VARLiNGAM"]:
            x = sub[sub["method"] == m]
            if len(x) > 0:
                print(f"  {m:20s}  AUROC={x['auroc'].mean():.3f}±{x['auroc'].std():.3f}  "
                      f"F1={x['f1_opt'].mean():.3f}±{x['f1_opt'].std():.3f}")

    # Headline gap: PRCD-MAP(learned) - best baseline
    print("\n>>> Headline gap check: PRCD-MAP(learned) vs best baseline")
    for acc in args.accs:
        sub = df[df["acc"] == acc]
        prcd = sub[sub["method"] == "PRCD-MAP(learned)"]["auroc"]
        best = -np.inf
        best_name = ""
        for m in ["PCMCI+", "DYNOTEARS", "VARLiNGAM"]:
            x = sub[sub["method"] == m]["auroc"]
            if len(x) > 0 and x.mean() > best:
                best = x.mean()
                best_name = m
        if len(prcd) > 0:
            print(f"  acc={acc}: PRCD={prcd.mean():.3f}, best={best:.3f} ({best_name}), gap={prcd.mean()-best:+.3f}")

    elapsed = time.time() - t_global
    print(f"\n>>> Exp13 complete in {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()
