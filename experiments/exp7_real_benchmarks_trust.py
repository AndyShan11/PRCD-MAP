"""
=============================================================================
Experiment 2 — Real-World Benchmarks with Trust Propagation
=============================================================================
Lorenz-96, CausalTime (AQI, Traffic, Medical), nonlinear synthetic.

Usage:
  python exp2_real_benchmarks.py --bench lorenz --seeds 0 1 2
  python exp2_real_benchmarks.py --bench causaltime --seeds 0 1 2
  python exp2_real_benchmarks.py --bench nonlinear --seeds 0 1 2
  python exp2_real_benchmarks.py --bench all --seeds 0 1 2
=============================================================================
"""

import os, sys, time, warnings, argparse, traceback
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_trust import *


LORENZ_SETTINGS = [
    (10,  500,  "d10_T500"),
    (20,  500,  "d20_T500"),
    (20,  200,  "d20_T200"),
    (40,  500,  "d40_T500"),
]


@dataclass
class Cfg:
    K:                int   = 1
    prior_accs:       List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    seeds:            List[int]   = field(default_factory=lambda: [0, 1, 2])
    lambda1:          float = 0.001
    lambda2:          float = 0.01
    max_iter:         int   = 35
    inner_iter:       int   = 400
    lr:               float = 1e-2
    do_baselines:     bool = True
    do_trust:         bool = True
    do_per_group:     bool = True
    do_nam_trust:     bool = True
    do_dynotears:     bool = True
    do_pcmci:         bool = True
    do_varlingam:     bool = True
    bench:            str = "all"
    # Paths
    causaltime_dir:   str = "/home/shanxh/PRCD/data/causaltime"
    causaltime_datasets: List[str] = field(default_factory=lambda: ["AQI", "Traffic", "Medical"])
    causaltime_n_samples: int = 10
    electricity_xlsx: str = "/home/shanxh/PRCD/0227test.xlsx"
    electricity_prior: str = "/home/shanxh/PRCD/Auto_Generated_Prior.csv"
    output_dir:       str = "exp2_trust_results"


def run_lorenz_benchmark(cfg):
    """Lorenz-96 benchmark with trust propagation."""
    print("\n>>> Benchmark: Lorenz-96")
    all_results = []

    for d_lorenz, T_lorenz, label in LORENZ_SETTINGS:
        for prior_acc in cfg.prior_accs:
            for seed in cfg.seeds:
                setting = f"Lorenz_{label}_acc{prior_acc:.1f}"
                print(f"\n  {setting} seed={seed}")
                t0 = time.time()
                try:
                    X, B_true = generate_lorenz96(d=d_lorenz, T=T_lorenz, seed=seed)
                    d = X.shape[1]
                    K = cfg.K

                    # Generate prior from ground truth
                    P_prior = gen_prior_from_truth(B_true, acc=prior_acc, seed=seed)

                    # Create fake W0/Wk for metrics
                    W0_true = B_true.astype(float)
                    Wk_true = [np.zeros_like(W0_true)]

                    results = run_single_setting(
                        X, d, K, W0_true, Wk_true, P_prior, seed,
                        lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr,
                        do_baselines=cfg.do_baselines,
                        do_trust=cfg.do_trust,
                        do_per_group=cfg.do_per_group,
                        do_nam_trust=cfg.do_nam_trust and (d <= 10),
                        do_dynotears=cfg.do_dynotears,
                        do_pcmci=cfg.do_pcmci,
                        do_varlingam=cfg.do_varlingam,
                    )

                    for r in results:
                        r.update({
                            "setting": setting, "d": d, "T": T_lorenz,
                            "prior_acc": prior_acc, "benchmark": "Lorenz-96",
                        })
                    all_results.extend(results)

                    dt = time.time() - t0
                    print(f"    [{fmt_time(dt)}] done")
                except Exception as e:
                    traceback.print_exc()
                    print(f"    [ERROR] {e}")

    return all_results


def run_causaltime_benchmark(cfg):
    """CausalTime benchmark (AQI, Traffic, Medical)."""
    print("\n>>> Benchmark: CausalTime")
    all_results = []

    for ds_name in cfg.causaltime_datasets:
        X, B_true = load_causaltime(cfg.causaltime_dir, ds_name,
                                     n_samples=cfg.causaltime_n_samples)
        if X is None:
            warnings.warn(f"CausalTime {ds_name} not found, skipping")
            continue
        d = X.shape[1]
        K = cfg.K
        bench_name = f"CausalTime_{ds_name}_d{d}"
        print(f"\n  {bench_name}: d={d}, T={X.shape[0]}")

        W0_true = B_true.astype(float)
        Wk_true = [np.zeros_like(W0_true)]

        for prior_acc in cfg.prior_accs:
            for seed in cfg.seeds:
                setting = f"{bench_name}_acc{prior_acc:.1f}"
                print(f"    {setting} seed={seed}")
                t0 = time.time()
                try:
                    P_prior = gen_prior_from_truth(B_true, acc=prior_acc, seed=seed)

                    results = run_single_setting(
                        X, d, K, W0_true, Wk_true, P_prior, seed,
                        lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr,
                        do_baselines=cfg.do_baselines,
                        do_trust=cfg.do_trust,
                        do_per_group=cfg.do_per_group,
                        do_nam_trust=cfg.do_nam_trust and (d <= 10),
                        do_dynotears=cfg.do_dynotears,
                        do_pcmci=cfg.do_pcmci,
                        do_varlingam=cfg.do_varlingam,
                    )

                    for r in results:
                        r.update({
                            "setting": setting, "d": d, "T": X.shape[0],
                            "prior_acc": prior_acc, "benchmark": bench_name,
                        })
                    all_results.extend(results)

                    dt = time.time() - t0
                    print(f"      [{fmt_time(dt)}] done")
                except Exception as e:
                    traceback.print_exc()

    return all_results


def run_nonlinear_benchmark(cfg):
    """Nonlinear synthetic benchmark (NAM vs linear)."""
    print("\n>>> Benchmark: Nonlinear Synthetic")
    all_results = []

    for d in [10, 20]:
        T = 500
        for prior_acc in cfg.prior_accs:
            for seed in cfg.seeds:
                setting = f"NL_d{d}_T{T}_acc{prior_acc:.1f}"
                print(f"\n  {setting} seed={seed}")
                t0 = time.time()
                try:
                    set_seed(seed)
                    W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
                    Wk_true = make_lag_matrices(d, 1, edge_prob=0.10, seed=seed)
                    X = simulate_svar_nonlinear(T, W0_true, Wk_true, seed=seed)
                    if X is None:
                        continue
                    X = standardize(X)
                    P_prior = gen_prior(W0_true, Wk_true, acc=prior_acc, seed=seed)

                    results = run_single_setting(
                        X, d, cfg.K, W0_true, Wk_true, P_prior, seed,
                        lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr,
                        do_baselines=cfg.do_baselines,
                        do_trust=cfg.do_trust,
                        do_per_group=cfg.do_per_group,
                        do_nam_trust=(d <= 10),  # NAM仅d=10, d=20太慢
                        do_dynotears=cfg.do_dynotears,
                        do_pcmci=cfg.do_pcmci,
                        do_varlingam=cfg.do_varlingam,
                    )

                    for r in results:
                        r.update({
                            "setting": setting, "d": d, "T": T,
                            "prior_acc": prior_acc, "benchmark": "Nonlinear",
                        })
                    all_results.extend(results)

                    dt = time.time() - t0
                    print(f"    [{fmt_time(dt)}] done")
                except Exception as e:
                    traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="all",
                        choices=["lorenz", "causaltime", "nonlinear", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    cfg = Cfg(bench=args.bench)
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.skip_baselines:
        cfg.do_baselines = False
        cfg.do_dynotears = False
        cfg.do_pcmci = False
        cfg.do_varlingam = False

    ensure_dir(cfg.output_dir)
    t_global = time.time()
    all_results = []

    if cfg.bench in ("lorenz", "all"):
        all_results.extend(run_lorenz_benchmark(cfg))

    if cfg.bench in ("causaltime", "all"):
        all_results.extend(run_causaltime_benchmark(cfg))

    if cfg.bench in ("nonlinear", "all"):
        all_results.extend(run_nonlinear_benchmark(cfg))

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(cfg.output_dir, f"exp2_{cfg.bench}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> Saved {len(df)} rows -> {csv_path}")

        # Summary
        print("\n" + "=" * 80)
        for bench in df["benchmark"].unique():
            print(f"\n--- {bench} ---")
            sub = df[df["benchmark"] == bench]
            for metric in ["auroc", "f1_opt"]:
                try:
                    pivot = sub.groupby("method")[metric].agg(["mean", "std"])
                    print(f"  {metric}: ")
                    for m, row in pivot.iterrows():
                        print(f"    {m:30s}: {row['mean']:.3f} ± {row['std']:.3f}")
                except Exception:
                    pass

    elapsed = time.time() - t_global
    print(f"\n>>> Exp2 complete in {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()
