"""
=============================================================================
Experiment 16 — λ1 warmup-factor sensitivity
=============================================================================
Reviewer concern (W7): the algorithm inflates λ1 by 5× during the first I/3
outer iterations. The paper claims this is an optimization warm-start
orthogonal to EB calibration of τ. This script verifies that claim by
sweeping the warmup factor ∈ {1×, 2×, 5×, 10×} (and a no-schedule baseline)
and reporting AUROC / F1 / learned τ̄ across factors.

If the claim is true, AUROC should be a smooth, shallow function of the
factor with the optimum at our default 5×, and τ̄ should be largely
invariant (the schedule acts on W, not on τ).

Mechanism:
  src/model_linear.py reads warmup factor from env var PRCD_LAM1_WARMUP_FACTOR
  (default 5.0). For factor=0, we instead pass lambda_schedule=False.

Settings:
  - d=20, T=500, K=1, ER, Gaussian, acc ∈ {0.4, 0.6, 0.9}
  - factors ∈ {1, 2, 5, 10} + no-schedule baseline
  - seeds 0..4
  - learned-τ only (FullModel variant)

Usage:
  python exp16_lambda_sensitivity.py --seeds 0 1 2 3 4
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


def run_with_factor(X, P_prior, d, K, factor, schedule, seed):
    """factor: warmup multiplier; schedule: True/False (False = no schedule)."""
    # Set env var BEFORE training (read inside train_prcd_alm)
    os.environ["PRCD_LAM1_WARMUP_FACTOR"] = str(float(factor))

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

    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=35, inner_iter=400, lr=8e-3,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=False, postprocess=False,
        obs_mask=obs_mask,
        lambda_schedule=schedule,
    )
    return W0, Wk, tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--accs", type=float, nargs="+",
                        default=[0.4, 0.6, 0.9])
    parser.add_argument("--factors", type=float, nargs="+",
                        default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--d", type=int, default=20)
    args = parser.parse_args()

    output_dir = "exp16_lambda_sensitivity"
    ensure_dir(output_dir)

    d, K, T = args.d, 1, args.T
    results = []
    t_global = time.time()

    # (label, factor, schedule_on)
    configs = [("no_schedule", 1.0, False)]
    for f in args.factors:
        configs.append((f"x{int(f) if f.is_integer() else f}", f, True))

    for acc in args.accs:
        for seed in args.seeds:
            set_seed(seed)
            W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
            Wk_true = make_lag_matrices(d, K, edge_prob=0.10, seed=seed)
            X = simulate_svar_linear(T, W0_true, Wk_true,
                                      noise_type="gaussian", seed=seed)
            if X is None:
                continue
            X = standardize(X)
            P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)

            for label, factor, schedule in configs:
                t0 = time.time()
                try:
                    W0, Wk, tau = run_with_factor(
                        X, P_prior, d, K, factor, schedule, seed)
                    m = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
                    m.update({"config": label, "factor": factor,
                              "schedule": schedule,
                              "tau_mean": float(tau) if tau is not None else np.nan,
                              "seed": seed, "acc": acc, "T": T})
                    results.append(m)
                    auroc = m.get("auroc_combined", m.get("auroc", np.nan))
                    print(f"  acc={acc} seed={seed} {label:14s}: "
                          f"AUROC={auroc:.3f} τ̄={m['tau_mean']:.3f} "
                          f"[{fmt_time(time.time()-t0)}]")
                except Exception:
                    traceback.print_exc()

            if results:
                pd.DataFrame(results).to_csv(
                    os.path.join(output_dir, "_intermediate.csv"),
                    index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "exp16_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("Summary: AUROC mean±std and τ̄ per (config, acc)")
    print("=" * 60)
    col = "auroc_combined" if "auroc_combined" in df.columns else "auroc"
    for acc in sorted(df["acc"].unique()):
        print(f"\nacc={acc}:")
        sub = df[df["acc"] == acc]
        for label, _, _ in configs:
            r = sub[sub["config"] == label]
            if col not in r.columns:
                continue
            vals = r[col].dropna().values
            taus = r["tau_mean"].dropna().values
            if len(vals) > 0:
                print(f"  {label:14s}: AUROC={np.mean(vals):.3f}±{np.std(vals):.3f}, "
                      f"τ̄={np.mean(taus):.3f}±{np.std(taus):.3f} (n={len(vals)})")

    print(f"\n>>> Total time: {fmt_time(time.time() - t_global)}")
    print(f">>> Results: {output_dir}/exp16_results.csv")
    # Reset env
    os.environ.pop("PRCD_LAM1_WARMUP_FACTOR", None)


if __name__ == "__main__":
    main()
