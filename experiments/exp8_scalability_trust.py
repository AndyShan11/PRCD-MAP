"""
=============================================================================
Experiment 3 — Scalability & NAM Nonlinear (d=50, d=100)
=============================================================================
Part A: Scalability of trust propagation (d=10,20,50,100)
Part B: NAM nonlinear at d=50,100 (using TrustPropagationLite for large d)

Usage:
  python exp3_scalability.py --sub scale --seeds 0 1 2
  python exp3_scalability.py --sub nonlinear_large --seeds 0 1 2
  python exp3_scalability.py --sub all --seeds 0 1 2
=============================================================================
"""

import os, sys, time, warnings, argparse, traceback
from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_trust import *


@dataclass
class Cfg:
    seeds:          List[int]   = field(default_factory=lambda: [0, 1, 2])
    K:              int   = 1
    lambda1:        float = 0.001
    lambda2:        float = 0.01
    max_iter:       int   = 35
    inner_iter:     int   = 400
    lr:             float = 1e-2
    prior_acc:      float = 0.6
    do_baselines:   bool = True
    sub:            str = "all"
    output_dir:     str = "exp3_trust_results"


def run_scalability(cfg):
    """Part A: Runtime & accuracy vs d with trust propagation."""
    print("\n>>> Part A: Scalability")
    all_results = []
    dims = [10, 20, 50, 100]
    T = 500

    for d in dims:
        for seed in cfg.seeds:
            setting = f"scale_d{d}"
            print(f"\n  {setting} seed={seed}")

            try:
                set_seed(seed)
                W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
                Wk_true = make_lag_matrices(d, cfg.K, edge_prob=0.10, seed=seed)
                X = simulate_svar_linear(T, W0_true, Wk_true, seed=seed)
                if X is None:
                    print("  [SKIP] sim failed")
                    continue
                X = standardize(X)
                P_prior = gen_prior(W0_true, Wk_true, acc=cfg.prior_acc, seed=seed)

                B_w0 = (np.abs(W0_true) > 1e-10).astype(float)
                B_comb = B_w0.copy().astype(int)
                for wk in Wk_true:
                    B_comb = np.maximum(B_comb, (np.abs(wk) > 1e-10).astype(int))

                # PRCD-MAP (trust) — auto-switches to Lite for d>50
                t0 = time.time()
                W0_t, Wk_t, tau_t = run_prcd_trust(
                    X, P_prior, d, cfg.K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, seed=seed, trust_lite=(d > 50))
                dt_trust = time.time() - t0
                met_t = compute_dual_metrics(W0_true, Wk_true, W0_t, Wk_t)
                met_t.update({"method": "PRCD-MAP(trust)", "d": d, "seed": seed,
                              "runtime": dt_trust, "setting": setting})
                all_results.append(met_t)
                print(f"    trust: {dt_trust:.1f}s, AUROC={met_t['auroc']:.3f}")

                # PRCD-MAP (per-group)
                t0 = time.time()
                W0_p, Wk_p, tau_p = run_prcd_map(
                    X, P_prior, d, cfg.K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, seed=seed)
                dt_pg = time.time() - t0
                met_p = compute_dual_metrics(W0_true, Wk_true, W0_p, Wk_p)
                met_p.update({"method": "PRCD-MAP(per-group)", "d": d, "seed": seed,
                              "runtime": dt_pg, "setting": setting})
                all_results.append(met_p)
                print(f"    per-group: {dt_pg:.1f}s, AUROC={met_p['auroc']:.3f}")

                # Baselines
                if cfg.do_baselines:
                    for name, func in [("DYNOTEARS", run_dynotears),
                                       ("PCMCI+", run_pcmci_plus),
                                       ("VARLiNGAM", run_varlingam)]:
                        try:
                            t0 = time.time()
                            W0_b, Wk_b = func(X, d, cfg.K, seed=seed)
                            dt_b = time.time() - t0
                            if W0_b is not None:
                                met_b = compute_dual_metrics(W0_true, Wk_true, W0_b, Wk_b)
                                met_b.update({"method": name, "d": d, "seed": seed,
                                              "runtime": dt_b, "setting": setting})
                                all_results.append(met_b)
                                print(f"    {name}: {dt_b:.1f}s, AUROC={met_b['auroc']:.3f}")
                        except Exception as e:
                            warnings.warn(f"{name} failed: {e}")

            except Exception as e:
                traceback.print_exc()

    return all_results


def run_nonlinear_large(cfg):
    """Part B: Nonlinear experiments across scales.
    d=20,30: NAM+Trust (full GAT) vs NAM(per-group) vs linear trust
    d=50,100: linear trust-lite vs linear per-group (NAM too expensive)
    """
    print("\n>>> Part B: Nonlinear Multi-Scale")
    all_results = []
    dims = [20, 30, 50, 100]
    T = 500

    for d in dims:
        for seed in cfg.seeds:
            setting = f"NL_d{d}"
            print(f"\n  {setting} seed={seed}")
            try:
                set_seed(seed)
                W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
                Wk_true = make_lag_matrices(d, cfg.K, edge_prob=0.10, seed=seed)
                X = simulate_svar_nonlinear(T, W0_true, Wk_true, seed=seed)
                if X is None:
                    print("  [SKIP] nonlinear sim failed, falling back to linear")
                    X = simulate_svar_linear(T, W0_true, Wk_true, seed=seed)
                    if X is None:
                        continue
                X = standardize(X)
                P_prior = gen_prior(W0_true, Wk_true, acc=cfg.prior_acc, seed=seed)

                # --- NAM + Trust (d<=30 only) ---
                if d <= 10:  # NAM仅d=10, d≥20太慢(380+MLPs)
                    t0 = time.time()
                    W0_nt, Wk_nt, tau_nt = run_prcd_nam_trust(
                        X, P_prior, d, cfg.K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=5e-4, seed=seed)
                    dt = time.time() - t0
                    met = compute_dual_metrics(W0_true, Wk_true, W0_nt, Wk_nt)
                    met.update({"method": "PRCD-MAP(trust+NAM)", "d": d, "seed": seed,
                                "runtime": dt, "setting": setting})
                    all_results.append(met)
                    print(f"    trust+NAM: {dt:.1f}s, AUROC={met['auroc']:.3f}")

                    # NAM per-group baseline (原始NAM, 无trust)
                    t0 = time.time()
                    W0_np, Wk_np, tau_np = run_prcd_map_nam(
                        X, P_prior, d, cfg.K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=5e-4, seed=seed)
                    dt = time.time() - t0
                    if W0_np is not None:
                        met = compute_dual_metrics(W0_true, Wk_true, W0_np, Wk_np)
                        met.update({"method": "PRCD-MAP(NAM)", "d": d, "seed": seed,
                                    "runtime": dt, "setting": setting})
                        all_results.append(met)
                        print(f"    NAM(per-group): {dt:.1f}s, AUROC={met['auroc']:.3f}")

                # --- Linear Trust (all d) ---
                t0 = time.time()
                W0_t, Wk_t, tau_t = run_prcd_trust(
                    X, P_prior, d, cfg.K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, seed=seed, trust_lite=(d > 50))
                dt = time.time() - t0
                met = compute_dual_metrics(W0_true, Wk_true, W0_t, Wk_t)
                label = "PRCD-MAP(trust-lite)" if d > 50 else "PRCD-MAP(trust)"
                met.update({"method": label, "d": d, "seed": seed,
                            "runtime": dt, "setting": setting})
                all_results.append(met)
                print(f"    {label}: {dt:.1f}s, AUROC={met['auroc']:.3f}")

                # --- Linear Per-group baseline ---
                t0 = time.time()
                W0_p, Wk_p, tau_p = run_prcd_map(
                    X, P_prior, d, cfg.K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, seed=seed)
                dt = time.time() - t0
                met = compute_dual_metrics(W0_true, Wk_true, W0_p, Wk_p)
                met.update({"method": "PRCD-MAP(per-group)", "d": d, "seed": seed,
                            "runtime": dt, "setting": setting})
                all_results.append(met)
                print(f"    per-group: {dt:.1f}s, AUROC={met['auroc']:.3f}")

            except Exception as e:
                traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="all",
                        choices=["scale", "nonlinear_large", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    cfg = Cfg(sub=args.sub)
    if args.seeds:
        cfg.seeds = args.seeds
    if args.skip_baselines:
        cfg.do_baselines = False

    ensure_dir(cfg.output_dir)
    t_global = time.time()
    all_results = []

    if cfg.sub in ("scale", "all"):
        all_results.extend(run_scalability(cfg))

    if cfg.sub in ("nonlinear_large", "all"):
        all_results.extend(run_nonlinear_large(cfg))

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(cfg.output_dir, f"exp3_{cfg.sub}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> Saved {len(df)} rows -> {csv_path}")

        # Summary table
        print("\n" + "=" * 80)
        print("SCALABILITY SUMMARY")
        print("=" * 80)
        for d_val in sorted(df["d"].unique()):
            sub = df[df["d"] == d_val]
            print(f"\n  d={d_val}:")
            for method in sub["method"].unique():
                m = sub[sub["method"] == method]
                print(f"    {method:30s}: AUROC={m['auroc'].mean():.3f}±{m['auroc'].std():.3f}"
                      f"  runtime={m['runtime'].mean():.1f}s")

    elapsed = time.time() - t_global
    print(f"\n>>> Exp3 complete in {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()
