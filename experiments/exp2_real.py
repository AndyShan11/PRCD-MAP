"""
=============================================================================
Experiment 2 — Real-World Benchmarks for PRCD-MAP
=============================================================================
Lorenz-96 difficulty gradient + electricity case study:

  Lorenz-96 settings (easy -> hard):
    d=10  T=500   — low-dim, scarce data
    d=20  T=500   — medium-dim, scarce data
    d=20  T=200   — medium-dim, very scarce data
    d=40  T=500   — high-dim, scarce data

  Case Study: Electricity consumption (no ground truth, interpretability)

Baselines: DYNOTEARS, PCMCI+, VARLiNGAM, PRCD-MAP variants
Metrics:   AUROC, AUPRC, Best-F1, Directed SHD, Normalized SHD

Scaled for RTX 2080 Ti (11 GB VRAM).

Usage:
  python exp2_real_benchmarks.py                       # full Lorenz-96 gradient
  python exp2_real_benchmarks.py --bench electricity   # electricity case study
  python exp2_real_benchmarks.py --quick               # fast subset (2 settings)
  python exp2_real_benchmarks.py --seeds 0 1 2 3 4
=============================================================================
"""

import os, sys, time, warnings, argparse
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- shared utilities (baselines, metrics, data gen, plotting, etc.) ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_utils import *


# ====================================================================
# Configuration
# ====================================================================

# Each tuple: (d, T, label_suffix)
LORENZ_SETTINGS_FULL: List[Tuple[int, int, str]] = [
    (10,  500,  "d10_T500"),    # low-dim, scarce
    (20,  500,  "d20_T500"),    # medium-dim, scarce
    (20,  200,  "d20_T200"),    # medium-dim, very scarce
    (40,  500,  "d40_T500"),    # high-dim, scarce
]

LORENZ_SETTINGS_QUICK: List[Tuple[int, int, str]] = [
    (10,  500,  "d10_T500"),
    (20,  200,  "d20_T200"),
]


@dataclass
class Cfg:
    # Lorenz-96
    lorenz_F:         float = 10.0
    lorenz_settings:  List[Tuple[int, int, str]] = field(
        default_factory=lambda: list(LORENZ_SETTINGS_FULL))

    # General
    K:                int   = 1
    prior_accs:       List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    seeds:            List[int]   = field(default_factory=lambda: [0, 1, 2])

    # Optimization (scaled for RTX 2080 Ti)
    lambda1:          float = 0.001
    lambda2:          float = 0.01
    max_iter:         int   = 35
    inner_iter:       int   = 400
    lr:               float = 1e-2

    # Baselines
    do_dynotears:     bool = True
    do_pcmci:         bool = True
    do_varlingam:     bool = True
    do_rhino:         bool = True
    do_ngc:           bool = True
    do_nam:           bool = True

    # Electricity paths
    electricity_xlsx: str = "/home/shanxh/PRCD/0227test.xlsx"
    electricity_prior: str = "/home/shanxh/PRCD/Auto_Generated_Prior.csv"

    # Netsim fMRI
    netsim_dir:       str = "/home/shanxh/PRCD/data/netsim"
    netsim_sims:      List[int] = field(default_factory=lambda: [3, 4, 15])

    # CausalTime
    causaltime_dir:   str = "/home/shanxh/PRCD/data/causaltime"
    causaltime_datasets: List[str] = field(default_factory=lambda: ["AQI", "Traffic", "Medical"])

    # Which benchmark to run
    bench:            str = "causaltime"

    # Output
    output_dir:       str = "exp2_results"


# ====================================================================
# Single Benchmark Runner (with ground truth)
# ====================================================================

def run_single_benchmark(
    bench_name: str,
    X: np.ndarray,
    B_true: np.ndarray,
    K: int,
    prior_accs: List[float],
    seeds: List[int],
    lambda1: float = 0.01,
    lambda2: float = 0.01,
    max_iter: int = 20,
    inner_iter: int = 400,
    lr: float = 1e-2,
    do_dynotears: bool = True,
    do_pcmci: bool = True,
    do_varlingam: bool = True,
    do_rhino: bool = True,
    do_ngc: bool = True,
    do_nam: bool = True,
) -> pd.DataFrame:
    """
    Run all methods on one benchmark dataset with known ground truth.

    Returns DataFrame with columns:
        bench, method, seed, prior_acc, prior_mode, tau, time, + all metrics
    """
    T, d = X.shape
    rows = []
    n_edges = int(B_true.sum() - np.trace(B_true))
    print(f"\n  [{bench_name}] d={d}, T={T}, K={K}, "
          f"true_edges={n_edges}, density={n_edges / (d * (d - 1) + 1e-12):.3f}")

    for seed in seeds:
        # ---- Baselines (prior-independent, once per seed) ----

        # DYNOTEARS
        if do_dynotears:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_dynotears(
                    X, d, K, lam=lambda1,
                    max_outer=max_iter, inner=inner_iter, lr=lr, seed=seed)
                W_eval = combine_W0_Wk(W0_est, Wk_est)
                met = compute_all_metrics(B_true, W_eval)
                rows.append(dict(
                    bench=bench_name, method="DYNOTEARS",
                    seed=seed, prior_acc=np.nan, prior_mode="none",
                    tau=np.nan, time=time.time() - t0, **met))
            except Exception as e:
                warnings.warn(f"DYNOTEARS failed on {bench_name}: {e}")

        # PCMCI+
        if do_pcmci and HAS_TIGRAMITE and d <= 80:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_pcmci_plus(X, d, K, seed=seed)
                if W0_est is not None:
                    W_eval = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_all_metrics(B_true, W_eval)
                    rows.append(dict(
                        bench=bench_name, method="PCMCI+",
                        seed=seed, prior_acc=np.nan, prior_mode="none",
                        tau=np.nan, time=time.time() - t0, **met))
            except Exception as e:
                warnings.warn(f"PCMCI+ failed on {bench_name}: {e}")

        # VARLiNGAM
        if do_varlingam and HAS_LINGAM:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_varlingam(X, d, K, seed=seed)
                if W0_est is not None:
                    W_eval = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_all_metrics(B_true, W_eval)
                    rows.append(dict(
                        bench=bench_name, method="VARLiNGAM",
                        seed=seed, prior_acc=np.nan, prior_mode="none",
                        tau=np.nan, time=time.time() - t0, **met))
            except Exception as e:
                warnings.warn(f"VARLiNGAM failed on {bench_name}: {e}")

        # RHINO (official codebase / CUTS+ fallback)
        if do_rhino and (HAS_RHINO or HAS_CUTS):
            try:
                t0 = time.time()
                W0_est, Wk_est = run_rhino(X, d, K, seed=seed)
                if W0_est is not None:
                    W_eval = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_all_metrics(B_true, W_eval)
                    rows.append(dict(
                        bench=bench_name, method="RHINO",
                        seed=seed, prior_acc=np.nan, prior_mode="none",
                        tau=np.nan, time=time.time() - t0, **met))
            except Exception as e:
                warnings.warn(f"RHINO failed on {bench_name}: {e}")

        # NGC (Neural Granger Causality)
        if do_ngc:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_ngc(X, d, K, seed=seed)
                if W0_est is not None:
                    W_eval = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_all_metrics(B_true, W_eval)
                    rows.append(dict(
                        bench=bench_name, method="NGC",
                        seed=seed, prior_acc=np.nan, prior_mode="none",
                        tau=np.nan, time=time.time() - t0, **met))
            except Exception as e:
                warnings.warn(f"NGC failed on {bench_name}: {e}")

        # ---- PRCD-MAP variants (sweep prior accuracy) ----
        for acc in prior_accs:
            P_prior = gen_prior_from_truth(B_true, acc, mode="random",
                                           seed=seed + 999)

            # PRCD-MAP (learn tau)
            try:
                t0 = time.time()
                W0_est, Wk_est, tau = run_prcd_map(
                    X, P_prior, d, K, lambda1, lambda2,
                    learn_tau=True, seed=seed,
                    max_iter=max_iter, inner_iter=inner_iter, lr=lr,
                    loss_type="huber", prior_l1_weight=True)
                W_eval = combine_W0_Wk(W0_est, Wk_est)
                met = compute_all_metrics(B_true, W_eval)
                rows.append(dict(
                    bench=bench_name, method="PRCD-MAP(learn_tau)",
                    seed=seed, prior_acc=acc, prior_mode="random",
                    tau=float(tau), time=time.time() - t0, **met))
            except Exception as e:
                warnings.warn(f"PRCD-MAP(learn_tau) failed: {e}")

            # PRCD-MAP (fixed tau=1) — Fix 8: prior_l1_weight=True for fair comparison
            try:
                t0 = time.time()
                W0_est, Wk_est, tau = run_prcd_map(
                    X, P_prior, d, K, lambda1, lambda2,
                    learn_tau=False, tau0=1.0, seed=seed,
                    max_iter=max_iter, inner_iter=inner_iter, lr=lr,
                    loss_type="huber", prior_l1_weight=True)
                W_eval = combine_W0_Wk(W0_est, Wk_est)
                met = compute_all_metrics(B_true, W_eval)
                rows.append(dict(
                    bench=bench_name, method="PRCD-MAP(fixed_tau)",
                    seed=seed, prior_acc=acc, prior_mode="random",
                    tau=float(tau), time=time.time() - t0, **met))
            except Exception as e:
                warnings.warn(f"PRCD-MAP(fixed_tau) failed: {e}")

        # PRCD-MAP (uniform prior) -- once per seed
        try:
            P_unif = np.full((d, d), 0.5)
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_unif, d, K, lambda1, lambda2,
                learn_tau=True, seed=seed,
                max_iter=max_iter, inner_iter=inner_iter, lr=lr,
                loss_type="huber", prior_l1_weight=False)
            W_eval = combine_W0_Wk(W0_est, Wk_est)
            met = compute_all_metrics(B_true, W_eval)
            rows.append(dict(
                bench=bench_name, method="PRCD-MAP(uniform)",
                seed=seed, prior_acc=np.nan, prior_mode="uniform",
                tau=float(tau), time=time.time() - t0, **met))
        except Exception as e:
            warnings.warn(f"PRCD-MAP(uniform) failed: {e}")

        # PRCD-MAP (NAM) -- nonlinear extension, once per seed with best prior
        if do_nam and d <= 30:
            best_acc = max(prior_accs) if prior_accs else 0.6
            P_prior_nam = gen_prior_from_truth(B_true, best_acc, mode="random",
                                               seed=seed + 999)
            try:
                t0 = time.time()
                W0_est, Wk_est, tau = run_prcd_map_nam(
                    X, P_prior_nam, d, K, lambda1, lambda2,
                    learn_tau=True, seed=seed,
                    max_iter=max_iter, inner_iter=inner_iter, lr=5e-4)
                W_eval = combine_W0_Wk(W0_est, Wk_est)
                met = compute_all_metrics(B_true, W_eval)
                rows.append(dict(
                    bench=bench_name, method="PRCD-MAP(NAM)",
                    seed=seed, prior_acc=best_acc, prior_mode="random",
                    tau=float(tau), time=time.time() - t0, **met))
            except Exception as e:
                warnings.warn(f"PRCD-MAP(NAM) failed on {bench_name}: {e}")

        print(f"    seed={seed} done ({len(rows)} rows so far)")

    return pd.DataFrame(rows)


# ====================================================================
# Electricity Case Study (no ground truth)
# ====================================================================

def run_electricity_case_study(
    cfg: Cfg,
    output_dir: str,
) -> pd.DataFrame:
    """
    Run all methods on electricity data WITHOUT ground truth.
    Outputs top-K edge list + heatmap for interpretability.
    """
    # Load data
    df_diff, P_prior, split, col_names = load_electricity(
        cfg.electricity_xlsx, cfg.electricity_prior)
    if df_diff is None:
        print("  Electricity data not found or failed to load.")
        print(f"  Expected: {cfg.electricity_xlsx}")
        print(f"  Prior:    {cfg.electricity_prior}")
        return pd.DataFrame()

    X = standardize(df_diff.values.astype(np.float64))
    d = X.shape[1]
    K = cfg.K
    seeds = cfg.seeds
    topk_per_seed = min(120, d * (d - 1))

    print(f"\n  [Electricity Case Study] d={d}, T={X.shape[0]}, K={K}")

    results = {}  # method_name -> list of combined W matrices

    # ---- PRCD-MAP (with domain prior) ----
    print("    Running PRCD-MAP (domain prior)...")
    prcd_Ws, prcd_taus = [], []
    for seed in seeds:
        try:
            W0, Wk, tau = run_prcd_map(
                X, P_prior, d, K,
                lambda1=0.001, lambda2=0.02,
                learn_tau=True, seed=seed,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr,
                loss_type="huber", prior_l1_weight=True)
            prcd_Ws.append(combine_W0_Wk(W0, Wk))
            prcd_taus.append(tau)
        except Exception as e:
            warnings.warn(f"PRCD-MAP seed={seed} failed: {e}")
    results["PRCD-MAP"] = prcd_Ws
    if prcd_taus:
        print(f"    PRCD-MAP tau: {np.mean(prcd_taus):.4f} "
              f"+/- {np.std(prcd_taus):.4f}")

    # ---- DYNOTEARS ----
    if cfg.do_dynotears:
        print("    Running DYNOTEARS...")
        ws = []
        for seed in seeds:
            try:
                W0, Wk = run_dynotears(X, d, K, lam=0.001,
                                        max_outer=cfg.max_iter,
                                        inner=cfg.inner_iter,
                                        lr=cfg.lr, seed=seed)
                ws.append(combine_W0_Wk(W0, Wk))
            except Exception:
                pass
        results["DYNOTEARS"] = ws

    # ---- PCMCI+ ----
    if cfg.do_pcmci and HAS_TIGRAMITE and d <= 80:
        print("    Running PCMCI+...")
        ws = []
        for seed in seeds:
            try:
                W0, Wk = run_pcmci_plus(X, d, K, seed=seed)
                if W0 is not None:
                    ws.append(combine_W0_Wk(W0, Wk))
            except Exception:
                pass
        results["PCMCI+"] = ws

    # ---- VARLiNGAM ----
    if cfg.do_varlingam and HAS_LINGAM:
        print("    Running VARLiNGAM...")
        ws = []
        for seed in seeds:
            try:
                W0, Wk = run_varlingam(X, d, K, seed=seed)
                if W0 is not None:
                    ws.append(combine_W0_Wk(W0, Wk))
            except Exception:
                pass
        results["VARLiNGAM"] = ws

    # ---- Compute stability + top edges ----
    names_en = [ZH_TO_EN.get(c, c) for c in col_names]
    summary_rows = []

    for mname, W_list in results.items():
        if not W_list:
            continue
        n_runs = len(W_list)
        appear = np.zeros((d, d), dtype=int)
        w_sum = np.zeros((d, d))

        for W in W_list:
            Wabs = np.abs(W).copy()
            np.fill_diagonal(Wabs, 0.0)
            flat = Wabs.reshape(-1)
            k = min(topk_per_seed, d * (d - 1))
            if k > 0 and flat.max() > 0:
                idx = np.argpartition(flat, -k)[-k:]
                mask = np.zeros_like(flat, dtype=bool)
                mask[idx] = True
                mask = mask.reshape(d, d)
                np.fill_diagonal(mask, False)
                appear += mask.astype(int)
                w_sum += W * mask

        freq = appear / max(n_runs, 1)
        w_mean = np.zeros_like(w_sum)
        nz = appear > 0
        w_mean[nz] = w_sum[nz] / appear[nz]

        score = freq * np.abs(w_mean)
        np.fill_diagonal(score, 0.0)

        top_n = min(60, d * (d - 1))
        flat_score = score.reshape(-1)
        top_idx = np.argpartition(flat_score, -top_n)[-top_n:]

        for idx in top_idx:
            i, j = divmod(idx, d)
            if abs(w_mean[i, j]) < 1e-12:
                continue
            summary_rows.append(dict(
                method=mname,
                src=names_en[i], dst=names_en[j],
                weight=float(w_mean[i, j]),
                frequency=float(freq[i, j]),
                score=float(score[i, j]),
            ))

    df_edges = pd.DataFrame(summary_rows)
    if not df_edges.empty:
        df_edges = df_edges.sort_values(["method", "score"],
                                        ascending=[True, False])
        p = os.path.join(output_dir, "electricity_case_study_edges.csv")
        ensure_dir(p)
        df_edges.to_csv(p, index=False)
        print(f"    Saved edge list: {p}")

    # ---- Heatmap for PRCD-MAP ----
    if prcd_Ws:
        _plot_electricity_heatmap(prcd_Ws, names_en, output_dir)

    # ---- Inter-method agreement ----
    method_names = [m for m in results if results[m]]
    if len(method_names) >= 2 and not df_edges.empty:
        print("\n    Inter-method edge agreement (top-60 edges):")
        for i, m1 in enumerate(method_names):
            for j, m2 in enumerate(method_names):
                if j <= i:
                    continue
                e1 = set(zip(df_edges[df_edges["method"] == m1]["src"],
                             df_edges[df_edges["method"] == m1]["dst"]))
                e2 = set(zip(df_edges[df_edges["method"] == m2]["src"],
                             df_edges[df_edges["method"] == m2]["dst"]))
                overlap = len(e1 & e2)
                union = len(e1 | e2)
                jaccard = overlap / max(union, 1)
                print(f"      {m1} vs {m2}: overlap={overlap}, "
                      f"Jaccard={jaccard:.3f}")

    return df_edges


def _plot_electricity_heatmap(W_list, names_en, output_dir):
    """Plot mean absolute weight heatmap from PRCD-MAP runs."""
    W_stack = np.stack([np.abs(W) for W in W_list], axis=0)
    W_mean = W_stack.mean(axis=0)
    np.fill_diagonal(W_mean, 0.0)

    d = W_mean.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, d * 0.5), max(7, d * 0.45)))
    im = ax.imshow(W_mean, cmap="Reds", aspect="auto")
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(names_en, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(names_en, fontsize=7)
    ax.set_xlabel("Effect (j)")
    ax.set_ylabel("Cause (i)")
    ax.set_title("PRCD-MAP Causal Strength (Electricity)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="|W| mean")
    plt.tight_layout()
    save_fig(os.path.join(output_dir, "electricity_heatmap"))


# ====================================================================
# Plotting: Per-Benchmark Bar Charts (AUROC, SHD)
# ====================================================================

def plot_benchmark_bars(df: pd.DataFrame, output_dir: str):
    """Grouped bar charts for AUROC and SHD across difficulty settings."""
    if df.empty:
        return

    # For PRCD-MAP with prior, use acc closest to 0.6
    prcd_l = df[df["method"] == "PRCD-MAP(learn_tau)"]
    if not prcd_l.empty and "prior_acc" in prcd_l.columns:
        avail = prcd_l["prior_acc"].dropna().unique()
        if len(avail) > 0:
            best = avail[np.argmin(np.abs(avail - 0.6))]
            prcd_l = prcd_l[prcd_l["prior_acc"] == best]

    others = df[~df["method"].isin(["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)"])]
    df_plot = pd.concat([others, prcd_l], ignore_index=True)

    methods = sorted(df_plot["method"].unique())
    benchmarks = sorted(df_plot["bench"].unique())
    n_m, n_b = len(methods), len(benchmarks)
    if n_m == 0 or n_b == 0:
        return

    for metric, ylabel, higher_better in [("auroc", "AUROC", True),
                                            ("f1_opt", "Best F1", True),
                                            ("shd_norm", "Normalized SHD", False)]:
        if metric not in df_plot.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(8, n_b * 2.5), 5.5))
        x = np.arange(n_b)
        w = 0.8 / max(n_m, 1)

        for i, m in enumerate(methods):
            vals, errs = [], []
            for b in benchmarks:
                s = df_plot[(df_plot["method"] == m) & (df_plot["bench"] == b)]
                vals.append(s[metric].mean() if len(s) else 0)
                errs.append(s[metric].std() if len(s) else 0)
            ax.bar(x + i * w, vals, w, yerr=errs,
                   label=m, color=COLORS.get(m, "grey"),
                   alpha=0.85, capsize=2)

        ax.set_xlabel("Lorenz-96 Setting (difficulty $\\rightarrow$)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + w * (n_m - 1) / 2)
        ax.set_xticklabels([b.replace("Lorenz96_", "L96_") for b in benchmarks],
                           rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.set_title(f"{ylabel} across Lorenz-96 Difficulty Gradient",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_fig(os.path.join(output_dir, f"bar_{metric}"))


def plot_prior_degradation(df: pd.DataFrame, output_dir: str):
    """AUROC and F1 vs prior accuracy for PRCD-MAP variants, per difficulty."""
    prcd_df = df[df["method"].isin(["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)"])]
    if prcd_df.empty or "prior_acc" not in prcd_df.columns:
        return
    if prcd_df["prior_acc"].dropna().nunique() < 2:
        return

    baseline_df = df[df["method"].isin(["DYNOTEARS", "PCMCI+", "VARLiNGAM",
                                         "PRCD-MAP(uniform)"])]

    # One row of subplots per difficulty setting
    benchmarks = sorted(df["bench"].unique())
    n_bench = len(benchmarks)

    fig, axes = plt.subplots(n_bench, 2,
                             figsize=(14, max(5, 3.5 * n_bench)),
                             squeeze=False)
    for row, bench in enumerate(benchmarks):
        prcd_sub = prcd_df[prcd_df["bench"] == bench]
        base_sub = baseline_df[baseline_df["bench"] == bench]

        for col, (metric, yl) in enumerate([
            ("f1_opt", "F1 (optimal threshold)"),
            ("auroc",  "AUROC"),
        ]):
            ax = axes[row, col]
            for m in ["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)"]:
                sub = prcd_sub[prcd_sub["method"] == m]
                if sub.empty:
                    continue
                agg = sub.groupby("prior_acc").agg(
                    y=(metric, "mean"), e=(metric, "std")).reset_index()
                ax.errorbar(agg["prior_acc"], agg["y"], yerr=agg["e"],
                            label=m, color=COLORS.get(m, "grey"),
                            marker=MARKERS.get(m, "x"),
                            linewidth=2, markersize=7, capsize=3)

            for m in base_sub["method"].unique():
                sub = base_sub[base_sub["method"] == m]
                if sub.empty:
                    continue
                mean_val = sub[metric].mean()
                ax.axhline(y=mean_val, color=COLORS.get(m, "grey"),
                           linestyle="--", alpha=0.7, linewidth=1.5,
                           label=f"{m} (avg)")

            ax.set_xlabel("Prior Accuracy", fontsize=10)
            ax.set_ylabel(yl, fontsize=10)
            ax.set_title(bench.replace("Lorenz96_", "L96_"), fontsize=10)
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, ls=":", alpha=0.5)
            ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Prior Degradation across Lorenz-96 Difficulty Settings",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(os.path.join(output_dir, "prior_degradation"))


def plot_difficulty_curve(df: pd.DataFrame, output_dir: str):
    """Line plot: metric vs difficulty setting, one line per method."""
    if df.empty:
        return

    # For PRCD-MAP with prior, use acc closest to 0.6
    prcd_l = df[df["method"] == "PRCD-MAP(learn_tau)"]
    if not prcd_l.empty and "prior_acc" in prcd_l.columns:
        avail = prcd_l["prior_acc"].dropna().unique()
        if len(avail) > 0:
            best = avail[np.argmin(np.abs(avail - 0.6))]
            prcd_l = prcd_l[prcd_l["prior_acc"] == best]

    others = df[~df["method"].isin(["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)"])]
    df_plot = pd.concat([others, prcd_l], ignore_index=True)

    benchmarks = sorted(df_plot["bench"].unique())
    methods = sorted(df_plot["method"].unique())
    n_bench = len(benchmarks)
    if n_bench < 2:
        return

    for metric, ylabel in [("auroc", "AUROC"), ("f1_opt", "Best F1"),
                            ("shd_norm", "Normalized SHD")]:
        if metric not in df_plot.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(8, n_bench * 1.5), 5.5))
        x = np.arange(n_bench)

        for m in methods:
            means, stds = [], []
            for b in benchmarks:
                s = df_plot[(df_plot["method"] == m) & (df_plot["bench"] == b)]
                means.append(s[metric].mean() if len(s) else np.nan)
                stds.append(s[metric].std() if len(s) else 0)
            ax.errorbar(x, means, yerr=stds,
                        label=m, color=COLORS.get(m, "grey"),
                        marker=MARKERS.get(m, "x"),
                        linewidth=2, markersize=7, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("Lorenz96_", "L96_") for b in benchmarks],
                           rotation=20, ha="right", fontsize=9)
        ax.set_xlabel("Lorenz-96 Setting (difficulty $\\rightarrow$)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_title(f"{ylabel} vs Difficulty",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_fig(os.path.join(output_dir, f"difficulty_{metric}"))


# ====================================================================
# RHINO-Style Summary Table
# ====================================================================

def make_rhino_table(df: pd.DataFrame, output_dir: str):
    """Print and save RHINO-style summary tables for AUROC and SHD."""
    if df.empty:
        return

    # Build a 'setting' column for the RHINO table
    def _make_setting(row):
        m = row["method"]
        if "PRCD-MAP" in m and not pd.isna(row.get("prior_acc", np.nan)):
            return f"acc={row['prior_acc']:.1f}"
        return "default"

    df = df.copy()
    df["setting"] = df.apply(_make_setting, axis=1)

    for metric in ["auroc", "f1_opt", "shd_norm"]:
        if metric not in df.columns:
            continue
        title = f"RHINO Table: {metric.upper()} (Exp2)"
        print_rhino_table(df, metric=metric, group_col="method",
                          setting_col="bench", title=title)


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Real-World Benchmarks for PRCD-MAP")
    parser.add_argument("--bench", type=str, default="causaltime",
                        choices=["causaltime", "electricity", "all"],
                        help="Benchmark to run (default: causaltime)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (2 Lorenz-96 settings only)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds (default: 0 1 2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--K", type=int, default=None, help="Lag order")
    parser.add_argument("--no-pcmci", action="store_true")
    parser.add_argument("--no-varlingam", action="store_true")
    parser.add_argument("--no-dynotears", action="store_true")
    parser.add_argument("--no-rhino", action="store_true")
    parser.add_argument("--no-ngc", action="store_true")
    parser.add_argument("--no-nam", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip all external baselines, only run PRCD-MAP variants")
    args = parser.parse_args()

    # Build config
    cfg = Cfg()
    if args.quick:
        cfg.lorenz_settings = list(LORENZ_SETTINGS_QUICK)
        cfg.prior_accs = [0.3, 0.9]
        cfg.seeds = [0, 1]
        cfg.max_iter = 10
        cfg.inner_iter = 200
        cfg.do_pcmci = False
        cfg.do_varlingam = False

    # CLI overrides
    cfg.bench = args.bench
    if args.output:
        cfg.output_dir = args.output
    if args.seeds:
        cfg.seeds = args.seeds
    if args.K is not None:
        cfg.K = args.K
    if args.skip_baselines:
        cfg.do_dynotears = False
        cfg.do_pcmci = False
        cfg.do_varlingam = False
        cfg.do_rhino = False
        cfg.do_ngc = False
        cfg.do_nam = False
    if args.no_pcmci:      cfg.do_pcmci = False
    if args.no_varlingam:  cfg.do_varlingam = False
    if args.no_dynotears:  cfg.do_dynotears = False
    if args.no_rhino:      cfg.do_rhino = False
    if args.no_ngc:        cfg.do_ngc = False
    if args.no_nam:        cfg.do_nam = False

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 64)
    print(" Experiment 2: Real-World Benchmarks for PRCD-MAP")
    print("=" * 64)
    print(f"  benchmark    = {cfg.bench}")
    if cfg.bench == "lorenz96":
        print(f"  settings     = {[(d, T) for d, T, _ in cfg.lorenz_settings]}")
    print(f"  K            = {cfg.K}")
    print(f"  prior_accs   = {cfg.prior_accs}")
    print(f"  seeds        = {cfg.seeds}")
    print(f"  max_iter     = {cfg.max_iter}, inner_iter = {cfg.inner_iter}")
    print(f"  baselines    = DYNOTEARS={cfg.do_dynotears}, "
          f"PCMCI+={cfg.do_pcmci and HAS_TIGRAMITE}, "
          f"VARLiNGAM={cfg.do_varlingam and HAS_LINGAM}")
    print(f"  output       = {cfg.output_dir}")
    print("=" * 64)

    t_global = time.time()

    # ==============================================================
    # Lorenz-96 Benchmark (difficulty gradient)
    # ==============================================================
    if cfg.bench == "lorenz96":
        print("\n>>> Benchmark: Lorenz-96 Difficulty Gradient")
        print(f"    F={cfg.lorenz_F}")

        all_dfs = []
        for lorenz_d, lorenz_T, label in cfg.lorenz_settings:
            print(f"\n{'─' * 50}")
            print(f"  Setting: d={lorenz_d}, T={lorenz_T}")
            print(f"{'─' * 50}")

            setting_dfs = []
            for seed in cfg.seeds:
                X, B_true = generate_lorenz96(
                    d=lorenz_d, T=lorenz_T, F=cfg.lorenz_F, seed=seed)
                if not np.all(np.isfinite(X)):
                    warnings.warn(f"Lorenz96 d={lorenz_d} T={lorenz_T} "
                                  f"seed={seed}: non-finite, skipping")
                    continue
                bench_name = f"Lorenz96_{label}"
                df = run_single_benchmark(
                    bench_name=bench_name, X=X, B_true=B_true,
                    K=cfg.K, prior_accs=cfg.prior_accs, seeds=[seed],
                    lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, do_dynotears=cfg.do_dynotears,
                    do_pcmci=cfg.do_pcmci, do_varlingam=cfg.do_varlingam,
                    do_rhino=cfg.do_rhino, do_ngc=cfg.do_ngc,
                    do_nam=cfg.do_nam)
                setting_dfs.append(df)

            if setting_dfs:
                df_setting = pd.concat(setting_dfs, ignore_index=True)
                all_dfs.append(df_setting)

                # Quick per-setting summary
                metric_cols = [c for c in ["auroc", "f1_opt", "shd_norm"]
                               if c in df_setting.columns]
                if metric_cols:
                    print(f"\n  Summary for d={lorenz_d}, T={lorenz_T}:")
                    print(df_setting.groupby("method")[metric_cols].mean()
                          .round(4).to_string())

        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            csv_path = os.path.join(cfg.output_dir, "exp2_lorenz96_results.csv")
            df_all.to_csv(csv_path, index=False)
            print(f"\n>>> Saved {len(df_all)} rows -> {csv_path}")

            # Plots
            plot_benchmark_bars(df_all, cfg.output_dir)
            plot_prior_degradation(df_all, cfg.output_dir)
            plot_difficulty_curve(df_all, cfg.output_dir)

            # RHINO table
            make_rhino_table(df_all, cfg.output_dir)

            # Console summary
            print("\n" + "=" * 60)
            print("  Mean metrics by method (across all settings)")
            print("=" * 60)
            metric_cols = [c for c in ["auroc", "auprc", "f1_opt", "shd_norm"]
                           if c in df_all.columns]
            if metric_cols:
                print(df_all.groupby("method")[metric_cols].mean()
                      .round(4).to_string())

            # Per-setting summary
            print("\n" + "=" * 60)
            print("  Mean metrics by method x setting")
            print("=" * 60)
            if metric_cols:
                print(df_all.groupby(["bench", "method"])[metric_cols].mean()
                      .round(4).to_string())
        else:
            print("\n>>> No Lorenz-96 results to aggregate.")

    # ==============================================================
    # Electricity Case Study
    # ==============================================================
    elif cfg.bench == "electricity":
        print("\n>>> Case Study: Electricity Consumption")
        df_edges = run_electricity_case_study(cfg, cfg.output_dir)
        if not df_edges.empty:
            print(f"\n>>> Top edges per method:")
            for m in df_edges["method"].unique():
                sub = df_edges[df_edges["method"] == m].head(15)
                print(f"\n  --- {m} (top 15) ---")
                for _, row in sub.iterrows():
                    print(f"    {row['src']:>20s} -> {row['dst']:<20s}  "
                          f"w={row['weight']:+.4f}  freq={row['frequency']:.2f}  "
                          f"score={row['score']:.4f}")

    # ==============================================================
    # Sparse Synthetic Benchmark (Fix 15: complement dense Lorenz-96)
    # ==============================================================
    elif cfg.bench == "sparse_synth":
        print("\n>>> Benchmark: Sparse Synthetic SVAR")
        print("    Sparse ER graph (edge_prob=0.08, density<0.1)")

        from exp_utils import make_er_dag, make_lag_matrices, simulate_svar_linear

        sparse_settings = [
            (20, 500, "sparse_d20_T500"),
            (20, 200, "sparse_d20_T200"),
            (40, 500, "sparse_d40_T500"),
        ]
        all_dfs = []
        for sp_d, sp_T, label in sparse_settings:
            print(f"\n  Setting: d={sp_d}, T={sp_T} (sparse)")
            setting_dfs = []
            for seed in cfg.seeds:
                W0_true = make_er_dag(sp_d, 0.08, seed=seed)
                Wk_true = make_lag_matrices(sp_d, cfg.K, 0.05, seed=seed)
                X = simulate_svar_linear(sp_T, W0_true, Wk_true, "gaussian",
                                          seed=seed)
                if X is None or not np.all(np.isfinite(X)):
                    continue
                X = standardize(X)
                B_true = (np.abs(W0_true) > 1e-10).astype(int)
                for Wk_t in Wk_true:
                    B_true = np.maximum(B_true,
                                         (np.abs(Wk_t) > 1e-10).astype(int))

                df = run_single_benchmark(
                    bench_name=label, X=X, B_true=B_true.astype(float),
                    K=cfg.K, prior_accs=cfg.prior_accs, seeds=[seed],
                    lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, do_dynotears=cfg.do_dynotears,
                    do_pcmci=cfg.do_pcmci, do_varlingam=cfg.do_varlingam,
                    do_rhino=cfg.do_rhino, do_ngc=cfg.do_ngc,
                    do_nam=cfg.do_nam)
                setting_dfs.append(df)

            if setting_dfs:
                df_setting = pd.concat(setting_dfs, ignore_index=True)
                all_dfs.append(df_setting)

        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            csv_path = os.path.join(cfg.output_dir,
                                     "exp2_sparse_synth_results.csv")
            df_all.to_csv(csv_path, index=False)
            print(f"\n>>> Saved {len(df_all)} rows -> {csv_path}")
            make_rhino_table(df_all, cfg.output_dir)

    # ==============================================================
    # Netsim fMRI Benchmark
    # ==============================================================
    if cfg.bench in ("netsim", "all"):
        print("\n>>> Benchmark: Netsim fMRI")
        all_dfs = []
        for sim_id in cfg.netsim_sims:
            X, B_true = load_netsim(cfg.netsim_dir, sim_id)
            if X is None:
                warnings.warn(f"Netsim sim{sim_id} not found, skipping")
                continue
            d = X.shape[1]
            bench_name = f"Netsim_sim{sim_id}_d{d}"
            print(f"\n  {bench_name}: d={d}, T={X.shape[0]}")
            df = run_single_benchmark(
                bench_name=bench_name, X=X, B_true=B_true.astype(float),
                K=cfg.K, prior_accs=cfg.prior_accs, seeds=cfg.seeds,
                lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                lr=cfg.lr, do_dynotears=cfg.do_dynotears,
                do_pcmci=cfg.do_pcmci, do_varlingam=cfg.do_varlingam,
                do_rhino=cfg.do_rhino, do_ngc=cfg.do_ngc,
                do_nam=cfg.do_nam)
            all_dfs.append(df)

        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            csv_path = os.path.join(cfg.output_dir, "exp2_netsim_results.csv")
            df_all.to_csv(csv_path, index=False)
            print(f"\n>>> Saved {len(df_all)} rows -> {csv_path}")

    # ==============================================================
    # CausalTime Benchmark
    # ==============================================================
    if cfg.bench in ("causaltime", "all"):
        print("\n>>> Benchmark: CausalTime")
        all_dfs = []
        for ds_name in cfg.causaltime_datasets:
            # n_samples=10: concatenate 10 samples for effective T=400
            n_samp = getattr(cfg, "causaltime_n_samples", 10)
            X, B_true = load_causaltime(cfg.causaltime_dir, ds_name, n_samples=n_samp)
            if X is None:
                warnings.warn(f"CausalTime {ds_name} not found, skipping")
                continue
            d = X.shape[1]
            bench_name = f"CausalTime_{ds_name}_d{d}"
            print(f"\n  {bench_name}: d={d}, T={X.shape[0]} (n_samples={n_samp})")
            df = run_single_benchmark(
                bench_name=bench_name, X=X, B_true=B_true.astype(float),
                K=cfg.K, prior_accs=cfg.prior_accs, seeds=cfg.seeds,
                lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                lr=cfg.lr, do_dynotears=cfg.do_dynotears,
                do_pcmci=cfg.do_pcmci, do_varlingam=cfg.do_varlingam,
                do_rhino=cfg.do_rhino, do_ngc=cfg.do_ngc,
                do_nam=cfg.do_nam)
            all_dfs.append(df)

        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            csv_path = os.path.join(cfg.output_dir, "exp2_causaltime_results.csv")
            df_all.to_csv(csv_path, index=False)
            print(f"\n>>> Saved {len(df_all)} rows -> {csv_path}")

    elapsed = time.time() - t_global
    print(f"\n>>> Experiment 2 complete in {elapsed:.1f}s")
    print(f">>> Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
