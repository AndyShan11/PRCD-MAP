"""
=============================================================================
Experiment 17 — Contemporaneous-dominant SVAR: instantaneous-DAG necessity
=============================================================================
Reviewer concern (W8): in the main ablation (Table 4), LagsOnly slightly
outperforms FullModel because lagged dependencies dominate the identifiable
signal on the default benchmark (ER 0.15, K=1). The reviewer asks for a
benchmark where contemporaneous edges dominate, so the instantaneous-DAG
component is clearly necessary.

This experiment generates an SVAR with a dense contemporaneous DAG (edge
prob 0.30) and a sparse lag-1 matrix (edge prob 0.05) so the W0 component
carries the bulk of identifiable signal. We then run the same ablation:
FullModel vs LagsOnly. We expect FullModel to strictly dominate LagsOnly here.

Settings:
  - d=20, T=500, K=1, ER, Gaussian
  - W0 edge prob = 0.30 (dense), Wk edge prob = 0.05 (sparse)
  - acc=0.6 (moderate prior; the default)
  - seeds 0..4

Usage:
  python exp17_contemporaneous_dominant.py --seeds 0 1 2 3 4
=============================================================================
"""
import os, sys, time, argparse, traceback
import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
from utils import (set_seed, make_er_dag, make_lag_matrices,
                   simulate_svar_linear, standardize, gen_prior,
                   compute_dual_metrics, ensure_dir, fmt_time,
                   make_lag_tensors_with_mask)
from model_linear import PRCD_MAP_Model, train_prcd_alm


def run_full_model(X, P_prior, d, K, seed):
    set_seed(seed)
    X_t, X_lags, obs_mask = make_lag_tensors_with_mask(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]
    if obs_mask is not None:
        obs_mask = obs_mask.to(dev)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=0.001, lambda2=0.01,
        learn_tau=True, tau0=1.0,
        tau_min=0.05, tau_max=3.0,
    ).to(dev)
    return train_prcd_alm(
        model, X_t, X_lags,
        max_iter=35, inner_iter=400, lr=8e-3,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=False, postprocess=False, obs_mask=obs_mask)


def run_lags_only(X, P_prior, d, K, seed):
    """Apply prior only to lag matrices (W0 not regularised by prior)."""
    set_seed(seed)
    X_t, X_lags, obs_mask = make_lag_tensors_with_mask(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]
    if obs_mask is not None:
        obs_mask = obs_mask.to(dev)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=0.001, lambda2=0.01,
        learn_tau=True, tau0=1.0,
        tau_min=0.05, tau_max=3.0,
        apply_prior_to_w0=False,  # <-- the LagsOnly knob
    ).to(dev)
    return train_prcd_alm(
        model, X_t, X_lags,
        max_iter=35, inner_iter=400, lr=8e-3,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=False, postprocess=False, obs_mask=obs_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--accs", type=float, nargs="+",
                        default=[0.6])
    parser.add_argument("--w0_prob", type=float, default=0.30,
                        help="W0 (instantaneous) edge probability")
    parser.add_argument("--wk_prob", type=float, default=0.05,
                        help="Wk (lag) edge probability")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--d", type=int, default=20)
    args = parser.parse_args()

    output_dir = "exp17_contemporaneous_dominant"
    ensure_dir(output_dir)

    d, K, T = args.d, 1, args.T
    results = []
    t_global = time.time()

    for acc in args.accs:
        for seed in args.seeds:
            set_seed(seed)
            # Override edge probs to make W0 dominant
            W0_true = make_er_dag(d, edge_prob=args.w0_prob, seed=seed)
            Wk_true = make_lag_matrices(d, K, edge_prob=args.wk_prob, seed=seed)
            X = simulate_svar_linear(T, W0_true, Wk_true,
                                      noise_type="gaussian", seed=seed)
            if X is None:
                continue
            X = standardize(X)
            P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)

            n_w0 = int((np.abs(W0_true) > 1e-8).sum())
            n_wk = int(sum((np.abs(W) > 1e-8).sum() for W in Wk_true))
            print(f"\n--- acc={acc}, seed={seed} (|W0|={n_w0}, |Wk|={n_wk}) ---")

            t0 = time.time()
            try:
                W0, Wk, tau = run_full_model(X, P_prior, d, K, seed)
                m = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
                m.update({"variant": "FullModel", "seed": seed, "acc": acc,
                          "n_w0_true": n_w0, "n_wk_true": n_wk})
                results.append(m)
                print(f"  FullModel: AUROC={m.get('auroc_combined', np.nan):.3f}, "
                      f"AUROC_w0={m.get('auroc_w0', np.nan):.3f}, "
                      f"AUROC_wk={m.get('auroc_wk', np.nan):.3f} "
                      f"[{fmt_time(time.time()-t0)}]")
            except Exception:
                traceback.print_exc()

            t0 = time.time()
            try:
                W0, Wk, tau = run_lags_only(X, P_prior, d, K, seed)
                m = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
                m.update({"variant": "LagsOnly", "seed": seed, "acc": acc,
                          "n_w0_true": n_w0, "n_wk_true": n_wk})
                results.append(m)
                print(f"  LagsOnly:  AUROC={m.get('auroc_combined', np.nan):.3f}, "
                      f"AUROC_w0={m.get('auroc_w0', np.nan):.3f}, "
                      f"AUROC_wk={m.get('auroc_wk', np.nan):.3f} "
                      f"[{fmt_time(time.time()-t0)}]")
            except Exception:
                traceback.print_exc()

            if results:
                pd.DataFrame(results).to_csv(
                    os.path.join(output_dir, "_intermediate.csv"), index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "exp17_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("Summary: FullModel vs LagsOnly on contemporaneous-dominant SVAR")
    print(f"  W0 edge prob = {args.w0_prob}, Wk edge prob = {args.wk_prob}")
    print("=" * 60)
    for acc in sorted(df["acc"].unique()):
        print(f"\nacc={acc}:")
        sub = df[df["acc"] == acc]
        for var in ["FullModel", "LagsOnly"]:
            r = sub[sub["variant"] == var]
            for col_label in ["auroc_combined", "auroc_w0", "auroc_wk"]:
                if col_label in r.columns:
                    vals = r[col_label].dropna().values
                    if len(vals) > 0:
                        print(f"  {var:10s} {col_label:16s}: "
                              f"{np.mean(vals):.3f}±{np.std(vals):.3f}")

        # Paired test: FullModel vs LagsOnly on auroc_combined
        try:
            from scipy import stats
            full = sub[sub["variant"] == "FullModel"].sort_values("seed")
            lago = sub[sub["variant"] == "LagsOnly"].sort_values("seed")
            col = "auroc_combined"
            if col in full.columns and col in lago.columns:
                fv = full[col].dropna().values
                lv = lago[col].dropna().values
                n = min(len(fv), len(lv))
                if n >= 2:
                    t, p = stats.ttest_rel(fv[:n], lv[:n])
                    d_mean = float(np.mean(fv[:n] - lv[:n]))
                    print(f"  Paired: FullModel - LagsOnly = {d_mean:+.3f}, "
                          f"t={t:.2f}, p={p:.3f} (n={n})")
        except Exception:
            traceback.print_exc()

    print(f"\n>>> Total time: {fmt_time(time.time() - t_global)}")
    print(f">>> Results: {output_dir}/exp17_results.csv")


if __name__ == "__main__":
    main()
