"""
=============================================================================
Experiment 1 — Synthetic Validation: Trust Propagation vs Per-Group Temperature
=============================================================================
验证 structure-aware trust propagation 在 d=20 合成数据上相比 per-group τ 的改进.

Settings:
  - d=20, T=500, ER graph, K=1
  - 4 noise types × 6 prior accuracies × 3 seeds
  - Methods: PRCD-MAP(trust), PRCD-MAP(per-group), PRCD-MAP(trust+NAM),
             DYNOTEARS, PCMCI+, VARLiNGAM
  - Metrics: AUROC, AUPRC, F1, SHD (combined W0+Wk)

Usage:
  python exp1_trust_validation.py --sub prior --seeds 0 1 2
  python exp1_trust_validation.py --sub noise --seeds 0 1 2
  python exp1_trust_validation.py --sub nonlinear --seeds 0 1 2
  python exp1_trust_validation.py --quick
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
    dims:           List[int]   = field(default_factory=lambda: [20])
    sample_sizes:   List[int]   = field(default_factory=lambda: [500])
    graph_types:    List[str]   = field(default_factory=lambda: ["ER"])
    noise_types:    List[str]   = field(default_factory=lambda: ["gaussian"])
    lag_orders:     List[int]   = field(default_factory=lambda: [1])
    nonlinear:      bool        = False
    prior_accs:     List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior_modes:    List[str]   = field(default_factory=lambda: ["random"])
    seeds:          List[int]   = field(default_factory=lambda: [0, 1, 2])
    edge_prob_w0:   float = 0.15
    ba_m:           int   = 2
    edge_prob_wk:   float = 0.10
    lambda1:        float = 0.001
    lambda2:        float = 0.01
    max_iter:       int   = 35
    inner_iter:     int   = 400
    lr:             float = 8e-3
    do_baselines:   bool = True
    do_trust:       bool = True
    do_per_group:   bool = True
    do_nam_trust:   bool = True
    do_dynotears:   bool = True
    do_pcmci:       bool = True
    do_varlingam:   bool = True
    output_dir:     str  = "exp1_trust_results"


def cfg_sub(name: str):
    if name == "prior":
        return Cfg(
            dims=[20], noise_types=["gaussian"],
            prior_accs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            seeds=[0, 1, 2],
            do_nam_trust=False,  # 线性数据不需要NAM
            output_dir="exp1_trust_prior",
        )
    elif name == "noise":
        return Cfg(
            dims=[20],
            noise_types=["gaussian", "laplace", "student_t", "heteroscedastic"],
            prior_accs=[0.0, 0.4, 0.6, 0.9],
            seeds=[0, 1, 2],
            do_nam_trust=False,  # 线性数据不需要NAM
            output_dir="exp1_trust_noise",
        )
    elif name == "nonlinear":
        return Cfg(
            dims=[10, 20],
            nonlinear=True,
            noise_types=["gaussian"],
            prior_accs=[0.2, 0.6, 1.0],
            seeds=[0, 1, 2],
            do_nam_trust=True,
            output_dir="exp1_trust_nonlinear",
        )
    elif name == "quick":
        return Cfg(
            dims=[10], seeds=[0],
            prior_accs=[0.4, 0.8],
            max_iter=10, inner_iter=100,
            do_baselines=False, do_nam_trust=False,
            output_dir="exp1_trust_quick",
        )
    raise ValueError(f"Unknown sub: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="prior",
                        choices=["prior", "noise", "nonlinear", "quick"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    cfg = cfg_sub(args.sub)
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.skip_baselines:
        cfg.do_baselines = False
        cfg.do_dynotears = False
        cfg.do_pcmci = False
        cfg.do_varlingam = False

    ensure_dir(cfg.output_dir)
    print(f">>> Exp1 Trust Validation: sub={args.sub}")
    print(f"    dims={cfg.dims}, noise={cfg.noise_types}, accs={cfg.prior_accs}")
    print(f"    seeds={cfg.seeds}, nonlinear={cfg.nonlinear}")

    all_results = []
    t_global = time.time()

    for d in cfg.dims:
        for T in cfg.sample_sizes:
            for graph_type in cfg.graph_types:
                for noise_type in cfg.noise_types:
                    for K in cfg.lag_orders:
                        for prior_acc in cfg.prior_accs:
                            for prior_mode in cfg.prior_modes:
                                for seed in cfg.seeds:
                                    setting = (f"d{d}_T{T}_{graph_type}_{noise_type}"
                                               f"_K{K}_acc{prior_acc:.1f}_{prior_mode}"
                                               f"{'_NL' if cfg.nonlinear else ''}")
                                    print(f"\n--- {setting} seed={seed} ---")
                                    t0 = time.time()

                                    try:
                                        set_seed(seed)
                                        if graph_type == "ER":
                                            W0_true = make_er_dag(d, edge_prob=cfg.edge_prob_w0, seed=seed)
                                        else:
                                            W0_true = make_ba_dag(d, m=cfg.ba_m, seed=seed)
                                        Wk_true = make_lag_matrices(d, K, edge_prob=cfg.edge_prob_wk, seed=seed)

                                        if cfg.nonlinear:
                                            X = simulate_svar_nonlinear(T, W0_true, Wk_true,
                                                                        noise_type=noise_type, seed=seed)
                                        else:
                                            X = simulate_svar_linear(T, W0_true, Wk_true,
                                                                     noise_type=noise_type, seed=seed)
                                        if X is None:
                                            print("  [SKIP] simulation failed")
                                            continue

                                        X = standardize(X)
                                        P_prior = gen_prior(W0_true, Wk_true, acc=prior_acc,
                                                            mode=prior_mode, seed=seed)

                                        results = run_single_setting(
                                            X, d, K, W0_true, Wk_true, P_prior, seed,
                                            lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                                            max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                                            lr=cfg.lr,
                                            do_baselines=cfg.do_baselines,
                                            do_trust=cfg.do_trust,
                                            do_per_group=cfg.do_per_group,
                                            do_nam_trust=cfg.do_nam_trust and (d <= 30),
                                            do_dynotears=cfg.do_dynotears,
                                            do_pcmci=cfg.do_pcmci,
                                            do_varlingam=cfg.do_varlingam,
                                        )

                                        for r in results:
                                            r.update({
                                                "setting": setting,
                                                "d": d, "T": T, "graph_type": graph_type,
                                                "noise_type": noise_type, "K": K,
                                                "prior_acc": prior_acc, "prior_mode": prior_mode,
                                                "nonlinear": cfg.nonlinear,
                                            })
                                        all_results.extend(results)

                                        dt = time.time() - t0
                                        methods_done = [r["method"] for r in results]
                                        print(f"  [{fmt_time(dt)}] {methods_done}")

                                    except Exception as e:
                                        traceback.print_exc()
                                        print(f"  [ERROR] {e}")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(cfg.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> Saved {len(df)} rows -> {csv_path}")

        # Print summary table
        print("\n" + "=" * 80)
        print("SUMMARY: Trust Propagation vs Per-Group Temperature")
        print("=" * 80)

        for metric in ["auroc", "f1_opt"]:
            print(f"\n--- {metric.upper()} ---")
            try:
                pivot = df.groupby(["method", "prior_acc"])[metric].agg(["mean", "std"])
                pivot_wide = pivot["mean"].unstack("prior_acc")
                print(pivot_wide.to_string(float_format="%.3f"))
            except Exception:
                pass

        # Trust vs Per-group comparison
        print("\n--- Trust vs Per-Group Δ ---")
        for acc in sorted(df["prior_acc"].unique()):
            trust_rows = df[(df["method"] == "PRCD-MAP(trust)") & (df["prior_acc"] == acc)]
            pg_rows = df[(df["method"] == "PRCD-MAP(per-group)") & (df["prior_acc"] == acc)]
            if len(trust_rows) > 0 and len(pg_rows) > 0:
                delta_auroc = trust_rows["auroc"].mean() - pg_rows["auroc"].mean()
                delta_f1 = trust_rows["f1_opt"].mean() - pg_rows["f1_opt"].mean()
                print(f"  acc={acc:.1f}: ΔAUROC={delta_auroc:+.4f}, ΔF1={delta_f1:+.4f}")

    elapsed = time.time() - t_global
    print(f"\n>>> Exp1 complete in {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()
