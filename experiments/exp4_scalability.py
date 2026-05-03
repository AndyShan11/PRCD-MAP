"""
=============================================================================
Experiment 5 — Scalability & Hyperparameter Sensitivity Analysis for PRCD-MAP
=============================================================================

Part A: Scalability Timing
  Wall-clock runtime vs dimension d for PRCD-MAP and baselines.
  Log-log plot + summary table.

Part B: Hyperparameter Sensitivity
  lambda1 x lambda2 grid on synthetic d=20, T=500. Heatmap of F1.

Part C: Temperature tau Deep-Dive
  - tau vs prior_accuracy curve (dual y-axis with F1) on synthetic data
  - tau trajectory during training (via train_prcd_alm_with_logging)
  - tau on real electricity data
  - Per-edge Omega weight visualization

Part D: Convergence Analysis
  h(W0) trajectory and loss curves across ALM outer iterations.

Usage:
  python exp5_scalability_sensitivity.py                   # default
  python exp5_scalability_sensitivity.py --quick           # tiny test
  python exp5_scalability_sensitivity.py --full            # full sweep
  python exp5_scalability_sensitivity.py --part A          # scalability only
  python exp5_scalability_sensitivity.py --part B C        # combine parts
  python exp5_scalability_sensitivity.py --dims 10 20 50
  python exp5_scalability_sensitivity.py --seeds 0 1 2 3 4
=============================================================================
"""

import os, time, warnings, argparse
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils import (
    set_seed, make_lag_tensors, standardize, ensure_dir,
    make_er_dag, make_lag_matrices, simulate_svar_linear,
    gen_prior, compute_all_metrics, combine_W0_Wk, compute_dual_metrics,
    run_dynotears, run_pcmci_plus, run_varlingam, run_prcd_map,
    run_rhino, run_ngc, run_prcd_map_nam,
    validate_laplace_approximation,
    load_electricity, train_prcd_alm_with_logging,
    PRCD_MAP_Model, train_prcd_alm,
    COLORS, MARKERS, save_fig, ZH_TO_EN, fmt_time,
    HAS_TIGRAMITE, HAS_LINGAM, HAS_RHINO, HAS_CUTS,
)


# ====================================================================
# Configuration
# ====================================================================
@dataclass
class Cfg:
    # Part A: Scalability
    scale_dims:       List[int]   = field(default_factory=lambda: [10, 20, 50, 100])
    scale_T:          int         = 500
    scale_K:          int         = 1
    scale_prior_acc:  float       = 0.6

    # Part B: Hyperparameter Sensitivity
    hp_d:             int         = 20
    hp_T:             int         = 500
    hp_K:             int         = 1
    hp_prior_acc:     float       = 0.6
    hp_lam1_grid:     List[float] = field(default_factory=lambda: [0.0005, 0.001, 0.003, 0.005, 0.01, 0.05])
    hp_lam2_grid:     List[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1])

    # Part C: Temperature tau Deep-Dive
    tau_d:            int         = 20
    tau_T:            int         = 1000
    tau_K:            int         = 1
    tau_prior_accs:   List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Part D: Convergence
    conv_d:           int         = 20
    conv_T:           int         = 500
    conv_K:           int         = 1
    conv_prior_accs:  List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])

    # Real data paths
    electricity_xlsx:  str = "./data/electricity.xlsx"
    electricity_prior: str = "./data/electricity_prior.csv"

    # General
    seeds:            List[int]   = field(default_factory=lambda: [0, 1, 2, 3, 4])
    edge_prob:        float       = 0.15
    lambda1:          float       = 0.001
    lambda2:          float       = 0.01
    max_iter:         int         = 35
    inner_iter:       int         = 400
    lr:               float       = 1e-2

    # Baselines
    do_dynotears:     bool = True
    do_pcmci:         bool = True
    do_varlingam:     bool = True

    # Part E: Laplace approximation validation
    laplace_d:        int         = 20
    laplace_T:        int         = 500
    laplace_K:        int         = 1
    laplace_prior_accs: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    laplace_n_grid:   int         = 50

    # Which parts to run
    parts:            List[str]   = field(default_factory=lambda: ["A", "B", "C", "D", "E"])

    # Output
    output_dir:       str = "exp5_results"


def cfg_quick() -> Cfg:
    return Cfg(
        scale_dims=[10, 20],
        scale_T=300,
        hp_lam1_grid=[0.01, 0.1],
        hp_lam2_grid=[0.01, 0.1],
        tau_prior_accs=[0.0, 0.3, 0.6, 0.9],
        tau_T=500,
        conv_prior_accs=[0.3, 0.7],
        seeds=[0, 1],
        max_iter=10, inner_iter=100,
        do_pcmci=False, do_varlingam=False,
    )


def cfg_full() -> Cfg:
    return Cfg(
        scale_dims=[10, 20, 50, 100],
        hp_lam1_grid=[0.0005, 0.001, 0.003, 0.005, 0.01, 0.05],
        hp_lam2_grid=[0.001, 0.005, 0.01, 0.05, 0.1],
        tau_prior_accs=[round(x * 0.05, 2) for x in range(21)],
        conv_prior_accs=[0.1, 0.3, 0.5, 0.7, 0.9],
        seeds=list(range(5)),
        max_iter=35, inner_iter=500,
    )


# ====================================================================
# Helpers
# ====================================================================
def _make_data(d, T, K, seed, edge_prob=0.15, noise="laplace"):
    """Generate synthetic SVAR data + ground truth. Returns None on failure."""
    W0 = make_er_dag(d, edge_prob, seed=seed)
    Wk = make_lag_matrices(d, K, seed=seed)
    X = simulate_svar_linear(T, W0, Wk, noise, seed=seed)
    if X is None or not np.all(np.isfinite(X)):
        return None
    return dict(W0=W0, Wk=Wk, X=standardize(X))


def _build_model(d, K, P_prior, learn_tau=True, tau0=1.0,
                 lambda1=0.001, lambda2=0.01):
    """Build PRCD_MAP_Model with valid constructor args only."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau, tau0=tau0,
        tau_min=0.05, tau_max=3.0,
        loss_type="huber", prior_l1_weight=True,
        n_tau_groups=4,
    ).to(dev)


def _eval_combined(W0_true, Wk_true, W0_est, Wk_est):
    """Evaluate combined (W0 + lag) graph recovery."""
    B_comb = (np.abs(W0_true) > 1e-10).astype(int)
    for Wk_t in Wk_true:
        B_comb = np.maximum(B_comb, (np.abs(Wk_t) > 1e-10).astype(int))
    W_comb_est = combine_W0_Wk(W0_est, Wk_est)
    return compute_all_metrics(B_comb.astype(float), W_comb_est)


# ====================================================================
# Part A: Scalability Timing
# ====================================================================
def run_part_A(cfg: Cfg) -> pd.DataFrame:
    print("\n" + "=" * 64)
    print(" Part A: Scalability -- Runtime vs Dimension")
    print("=" * 64)

    T, K, acc = cfg.scale_T, cfg.scale_K, cfg.scale_prior_acc
    rows = []

    for d in cfg.scale_dims:
        for seed in cfg.seeds:
            data = _make_data(d, T, K, seed, cfg.edge_prob)
            if data is None:
                continue
            X_std = data["X"]
            P_prior = gen_prior(data["W0"], data["Wk"], acc, "random",
                                seed=seed + 999)
            base = dict(d=d, T=T, K=K, seed=seed)

            # PRCD-MAP (learn_tau)
            try:
                t0 = time.time()
                run_prcd_map(X_std, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                             learn_tau=True, seed=seed,
                             max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                             lr=cfg.lr)
                rows.append({**base, "method": "PRCD-MAP(learn_tau)",
                             "time": time.time() - t0})
            except Exception:
                pass

            # PRCD-MAP (fixed_tau)
            try:
                t0 = time.time()
                run_prcd_map(X_std, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                             learn_tau=False, tau0=1.0, seed=seed,
                             max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                             lr=cfg.lr)
                rows.append({**base, "method": "PRCD-MAP(fixed_tau)",
                             "time": time.time() - t0})
            except Exception:
                pass

            # DYNOTEARS
            if cfg.do_dynotears:
                try:
                    t0 = time.time()
                    run_dynotears(X_std, d, K, cfg.lambda1,
                                  cfg.max_iter, cfg.inner_iter, cfg.lr, seed)
                    rows.append({**base, "method": "DYNOTEARS",
                                 "time": time.time() - t0})
                except Exception:
                    pass

            # PCMCI+ (skip d > 50)
            if cfg.do_pcmci and HAS_TIGRAMITE and d <= 50:
                try:
                    t0 = time.time()
                    run_pcmci_plus(X_std, d, K, seed=seed)
                    rows.append({**base, "method": "PCMCI+",
                                 "time": time.time() - t0})
                except Exception:
                    pass

            # VARLiNGAM
            if cfg.do_varlingam and HAS_LINGAM:
                try:
                    t0 = time.time()
                    run_varlingam(X_std, d, K, seed=seed)
                    rows.append({**base, "method": "VARLiNGAM",
                                 "time": time.time() - t0})
                except Exception:
                    pass

            # RHINO
            if (HAS_RHINO or HAS_CUTS):
                try:
                    t0 = time.time()
                    run_rhino(X_std, d, K, seed=seed)
                    rows.append({**base, "method": "RHINO",
                                 "time": time.time() - t0})
                except Exception:
                    pass

            # NGC
            try:
                t0 = time.time()
                run_ngc(X_std, d, K, seed=seed)
                rows.append({**base, "method": "NGC",
                             "time": time.time() - t0})
            except Exception:
                pass

            # PRCD-MAP (NAM) — only d <= 30
            if d <= 30:
                try:
                    t0 = time.time()
                    run_prcd_map_nam(X_std, P_prior, d, K,
                                    cfg.lambda1, cfg.lambda2,
                                    learn_tau=True, seed=seed,
                                    max_iter=cfg.max_iter,
                                    inner_iter=cfg.inner_iter, lr=5e-4)
                    rows.append({**base, "method": "PRCD-MAP(NAM)",
                                 "time": time.time() - t0})
                except Exception:
                    pass

            print(f"  d={d}, seed={seed} done")

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No scalability results.")
        return df

    p = os.path.join(cfg.output_dir, "partA_scalability.csv")
    df.to_csv(p, index=False)
    print(f">>> {p}")
    return df


def plot_part_A(df: pd.DataFrame, cfg: Cfg):
    if df.empty:
        return
    out = cfg.output_dir

    # Summary table
    agg = df.groupby(["method", "d"]).agg(
        time_mean=("time", "mean"), time_std=("time", "std"),
    ).reset_index()
    piv = agg.pivot(index="method", columns="d", values="time_mean")
    print("\n--- Scalability Summary (mean seconds) ---")
    print(piv.round(2).to_string())

    p2 = os.path.join(out, "partA_scalability_summary.csv")
    agg.to_csv(p2, index=False)
    print(f">>> {p2}")

    # Figure: linear + log-log
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, use_log, suffix in [(axes[0], False, "Linear"),
                                 (axes[1], True, "Log-Log")]:
        for m in sorted(df["method"].unique()):
            sub = df[df["method"] == m].groupby("d").agg(
                t_m=("time", "mean"), t_s=("time", "std")).reset_index()
            ax.errorbar(sub["d"], sub["t_m"], yerr=sub["t_s"],
                        label=m, marker=MARKERS.get(m, "x"),
                        color=COLORS.get(m, "grey"), lw=2, capsize=3, ms=7)
        ax.set_xlabel("Number of Variables (d)", fontsize=12)
        ax.set_ylabel("Wall-clock Time (s)", fontsize=12)
        ax.set_title(f"Scalability ({suffix})", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, ls=":", alpha=0.5)
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

    plt.tight_layout()
    save_fig(os.path.join(out, "figA_scalability"))


# ====================================================================
# Part B: Hyperparameter Sensitivity
# ====================================================================
def run_part_B(cfg: Cfg) -> pd.DataFrame:
    print("\n" + "=" * 64)
    print(" Part B: Hyperparameter Sensitivity (lambda1 x lambda2)")
    print("=" * 64)

    d, T, K, acc = cfg.hp_d, cfg.hp_T, cfg.hp_K, cfg.hp_prior_acc
    n_total = len(cfg.hp_lam1_grid) * len(cfg.hp_lam2_grid) * len(cfg.seeds)
    print(f"  d={d}, T={T}, prior_acc={acc}, total runs={n_total}")
    rows, count = [], 0

    for seed in cfg.seeds:
        data = _make_data(d, T, K, seed, cfg.edge_prob)
        if data is None:
            continue
        P_prior = gen_prior(data["W0"], data["Wk"], acc, "random",
                            seed=seed + 999)

        for l1 in cfg.hp_lam1_grid:
            for l2 in cfg.hp_lam2_grid:
                count += 1
                try:
                    W0_est, Wk_est, tau = run_prcd_map(
                        data["X"], P_prior, d, K, l1, l2,
                        learn_tau=True, seed=seed,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr)
                    met = _eval_combined(data["W0"], data["Wk"],
                                         W0_est, Wk_est)
                    rows.append(dict(
                        lambda1=l1, lambda2=l2, seed=seed, tau=tau,
                        f1=met["f1_opt"], auroc=met["auroc"],
                        shd_norm=met["shd_norm"],
                    ))
                except Exception as e:
                    warnings.warn(f"HP l1={l1} l2={l2} seed={seed}: {e}")

                if count % max(1, n_total // 5) == 0:
                    print(f"  [{count}/{n_total}]")

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No HP results.")
        return df

    p = os.path.join(cfg.output_dir, "partB_hyperparam.csv")
    df.to_csv(p, index=False)
    print(f">>> {p}")
    return df


def plot_part_B(df: pd.DataFrame, cfg: Cfg):
    if df.empty:
        return
    out = cfg.output_dir

    agg = df.groupby(["lambda1", "lambda2"]).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auroc_mean=("auroc", "mean"), tau_mean=("tau", "mean"),
    ).reset_index()

    # F1 heatmap
    piv = agg.pivot(index="lambda1", columns="lambda2", values="f1_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin = max(0, piv.values.min() - 0.05)
    vmax = min(1.0, piv.values.max() + 0.05)
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto",
                   origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v:.3f}" for v in piv.columns], fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in piv.index], fontsize=9)
    ax.set_xlabel("$\\lambda_2$ (prior weight)", fontsize=12)
    ax.set_ylabel("$\\lambda_1$ (sparsity)", fontsize=12)
    ax.set_title(f"F1 Sensitivity (d={cfg.hp_d}, T={cfg.hp_T})",
                 fontsize=13, fontweight="bold")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv.values[i, j]
            c = "white" if val < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=c, fontweight="bold")
    plt.colorbar(im, ax=ax, label="F1")
    plt.tight_layout()
    save_fig(os.path.join(out, "figB_heatmap_f1"))

    # tau heatmap
    piv_tau = agg.pivot(index="lambda1", columns="lambda2", values="tau_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv_tau.values, cmap="coolwarm", aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(piv_tau.columns)))
    ax.set_xticklabels([f"{v:.3f}" for v in piv_tau.columns], fontsize=9)
    ax.set_yticks(range(len(piv_tau.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in piv_tau.index], fontsize=9)
    ax.set_xlabel("$\\lambda_2$", fontsize=12)
    ax.set_ylabel("$\\lambda_1$", fontsize=12)
    ax.set_title("Learned $\\tau$ vs ($\\lambda_1$, $\\lambda_2$)",
                 fontsize=13, fontweight="bold")
    for i in range(len(piv_tau.index)):
        for j in range(len(piv_tau.columns)):
            ax.text(j, i, f"{piv_tau.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="$\\tau$")
    plt.tight_layout()
    save_fig(os.path.join(out, "figB_heatmap_tau"))

    # Print best
    best = agg.loc[agg["f1_mean"].idxmax()]
    print(f"\n  Best: l1={best['lambda1']:.3f}, l2={best['lambda2']:.3f}"
          f" -> F1={best['f1_mean']:.4f}")


# ====================================================================
# Part C: Temperature tau Deep-Dive
# ====================================================================
def run_part_C(cfg: Cfg) -> pd.DataFrame:
    print("\n" + "=" * 64)
    print(" Part C: Temperature tau Deep-Dive")
    print("=" * 64)

    d, T, K = cfg.tau_d, cfg.tau_T, cfg.tau_K
    rows = []

    # --- C.1: tau vs prior accuracy (synthetic) ---
    print("\n  --- C.1: tau vs Prior Accuracy ---")
    for acc in cfg.tau_prior_accs:
        for seed in cfg.seeds:
            data = _make_data(d, T, K, seed, cfg.edge_prob, "laplace")
            if data is None:
                continue
            P_prior = gen_prior(data["W0"], data["Wk"], acc, "random",
                                seed=seed + 999)

            # learn_tau
            try:
                W0e, Wke, tau = run_prcd_map(
                    data["X"], P_prior, d, K, cfg.lambda1, cfg.lambda2,
                    learn_tau=True, seed=seed,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr)
                met = _eval_combined(data["W0"], data["Wk"], W0e, Wke)
                rows.append(dict(experiment="synth", prior_acc=acc, seed=seed,
                                 method="learn_tau", tau=tau, **met))
            except Exception:
                pass

            # fixed_tau
            try:
                W0e, Wke, tau = run_prcd_map(
                    data["X"], P_prior, d, K, cfg.lambda1, cfg.lambda2,
                    learn_tau=False, tau0=1.0, seed=seed,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr)
                met = _eval_combined(data["W0"], data["Wk"], W0e, Wke)
                rows.append(dict(experiment="synth", prior_acc=acc, seed=seed,
                                 method="fixed_tau", tau=tau, **met))
            except Exception:
                pass

            # uniform prior baseline
            try:
                P_unif = np.full((d, d), 0.5)
                W0e, Wke, tau = run_prcd_map(
                    data["X"], P_unif, d, K, cfg.lambda1, cfg.lambda2,
                    learn_tau=True, seed=seed,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr)
                met = _eval_combined(data["W0"], data["Wk"], W0e, Wke)
                rows.append(dict(experiment="synth", prior_acc=acc, seed=seed,
                                 method="uniform", tau=tau, **met))
            except Exception:
                pass

        print(f"    acc={acc:.1f} done")

    df_synth = pd.DataFrame(rows)

    # --- C.2: tau trajectory during training ---
    print("\n  --- C.2: tau Training Trajectory ---")
    traj_logs = []
    for acc in [0.3, 0.6, 0.9]:
        seed = cfg.seeds[0]
        data = _make_data(d, T, K, seed, cfg.edge_prob, "laplace")
        if data is None:
            continue
        P_prior = gen_prior(data["W0"], data["Wk"], acc, "random",
                            seed=seed + 999)
        set_seed(seed)
        X_t, X_lags = make_lag_tensors(data["X"], K)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t, X_lags = X_t.to(dev), [x.to(dev) for x in X_lags]
        model = _build_model(d, K, P_prior, learn_tau=True,
                             lambda1=cfg.lambda1, lambda2=cfg.lambda2)
        try:
            _, _, _, log_df = train_prcd_alm_with_logging(
                model, X_t, X_lags,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                lr=cfg.lr, tau_warmup=2)
            log_df["prior_acc"] = acc
            traj_logs.append(log_df)
        except Exception as e:
            warnings.warn(f"Tau trajectory acc={acc}: {e}")

    df_traj = pd.concat(traj_logs, ignore_index=True) if traj_logs else pd.DataFrame()

    # --- C.3: tau on real electricity data ---
    print("\n  --- C.3: tau on Real Electricity Data ---")
    real_rows = []
    omega_data = None
    try:
        df_ts, P_prior_real, split, col_names = load_electricity(
            cfg.electricity_xlsx, cfg.electricity_prior)
        d_real = len(col_names)
        X_real = standardize(df_ts.values)
        print(f"    d={d_real}, T={len(X_real)}")

        for seed in cfg.seeds:
            try:
                W0, Wk, tau = run_prcd_map(
                    X_real, P_prior_real, d_real, K,
                    lambda1=0.001, lambda2=0.02,
                    learn_tau=True, seed=seed,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr)
                real_rows.append(dict(seed=seed, method="learn_tau",
                                      tau=tau, d=d_real))
            except Exception as e:
                warnings.warn(f"Real data seed={seed}: {e}")
            print(f"    seed={seed} done")

        # Omega visualization for seed=0
        if real_rows:
            set_seed(0)
            X_t, X_lags = make_lag_tensors(X_real, K)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_t, X_lags = X_t.to(dev), [x.to(dev) for x in X_lags]
            model = _build_model(d_real, K, P_prior_real, learn_tau=True,
                                 lambda1=0.001, lambda2=0.02)
            W0, Wk, tau = train_prcd_alm(
                model, X_t, X_lags,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr,
                rho_0=1.0, gamma=2.0, tol=1e-6,
                verbose=False, postprocess=False, tau_warmup=2)
            with torch.no_grad():
                tau_matrix = model._expand_tau()
                Omega = model.omega_mask(tau_matrix).cpu().numpy()
                P_cal = model.calibrated_prior(tau_matrix).cpu().numpy()
            names_en = [ZH_TO_EN.get(c, c) for c in col_names]
            omega_data = dict(Omega=Omega, P_cal=P_cal,
                              P_raw=P_prior_real, names=names_en, tau=float(tau))

    except Exception as e:
        warnings.warn(f"Failed to load electricity data: {e}")

    df_real = pd.DataFrame(real_rows)

    # Save all CSVs
    df_all = pd.concat([df_synth, df_real], ignore_index=True)
    if not df_all.empty:
        p = os.path.join(cfg.output_dir, "partC_tau.csv")
        df_all.to_csv(p, index=False)
        print(f">>> {p}")
    if not df_traj.empty:
        p = os.path.join(cfg.output_dir, "partC_tau_trajectory.csv")
        df_traj.to_csv(p, index=False)
        print(f">>> {p}")

    return df_all, df_traj, df_real, omega_data


def plot_part_C(df_synth, df_traj, df_real, omega_data, cfg):
    out = cfg.output_dir

    # --- C1: tau vs prior accuracy (dual y-axis) ---
    if not df_synth.empty:
        sub_learn = df_synth[df_synth["method"] == "learn_tau"]
        if not sub_learn.empty and sub_learn["prior_acc"].nunique() >= 3:
            agg = sub_learn.groupby("prior_acc").agg(
                tau_m=("tau", "mean"), tau_s=("tau", "std"),
                f1_m=("f1_opt", "mean"), f1_s=("f1_opt", "std"),
            ).reset_index()

            fig, ax1 = plt.subplots(figsize=(9, 6))
            c_tau, c_f1 = "#E74C3C", "#2C3E50"

            ax1.errorbar(agg["prior_acc"], agg["tau_m"], yerr=agg["tau_s"],
                         color=c_tau, marker="o", lw=2.5, ms=8, capsize=4,
                         label="Learned $\\tau$")
            ax1.set_xlabel("Prior Accuracy", fontsize=13)
            ax1.set_ylabel("Learned $\\tau$", fontsize=13, color=c_tau)
            ax1.tick_params(axis="y", labelcolor=c_tau)
            ax1.set_xlim(-0.05, 1.05)

            ax2 = ax1.twinx()
            ax2.errorbar(agg["prior_acc"], agg["f1_m"], yerr=agg["f1_s"],
                         color=c_f1, marker="s", lw=2, ms=7, capsize=3,
                         ls="--", label="F1 (learn $\\tau$)")

            sub_fixed = df_synth[df_synth["method"] == "fixed_tau"]
            if not sub_fixed.empty:
                af = sub_fixed.groupby("prior_acc").agg(
                    f1_m=("f1_opt", "mean"), f1_s=("f1_opt", "std"),
                ).reset_index()
                ax2.errorbar(af["prior_acc"], af["f1_m"], yerr=af["f1_s"],
                             color="#E67E22", marker="^", lw=2, ms=6,
                             capsize=3, ls="-.", label="F1 (fixed $\\tau$=1)")

            sub_unif = df_synth[df_synth["method"] == "uniform"]
            if not sub_unif.empty:
                f1_u = sub_unif["f1_opt"].mean()
                ax2.axhline(y=f1_u, color="#9B59B6", ls=":", lw=2,
                            alpha=0.8, label=f"F1 (no prior) = {f1_u:.3f}")

            ax2.set_ylabel("F1", fontsize=13, color=c_f1)
            ax2.tick_params(axis="y", labelcolor=c_f1)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9)
            ax1.set_title("$\\tau$ Adaptation vs Prior Quality",
                          fontsize=13, fontweight="bold")
            ax1.grid(True, ls=":", alpha=0.4)
            plt.tight_layout()
            save_fig(os.path.join(out, "figC1_tau_vs_prior"))

    # --- C2: tau trajectory during training ---
    if not df_traj.empty:
        accs = sorted(df_traj["prior_acc"].unique())
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(accs)))
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for idx, acc in enumerate(accs):
            sub = df_traj[df_traj["prior_acc"] == acc]
            ax.plot(sub["outer_iter"], sub["tau"],
                    marker="o", lw=2, ms=5, color=cmap[idx],
                    label=f"acc={acc:.1f}")
        ax.set_xlabel("Outer ALM Iteration", fontsize=12)
        ax.set_ylabel("$\\tau$", fontsize=12)
        ax.set_title("$\\tau$ Trajectory During Training", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
        plt.tight_layout()
        save_fig(os.path.join(out, "figC2_tau_trajectory"))

    # --- C3: tau on real data ---
    if not df_real.empty:
        tau_vals = df_real[df_real["method"] == "learn_tau"]["tau"].values
        if len(tau_vals) > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.boxplot([tau_vals], labels=["Electricity"], widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor="#E74C3C", alpha=0.6),
                       medianprops=dict(color="black", lw=2))
            ax.scatter(np.ones(len(tau_vals)), tau_vals,
                       color="#E74C3C", s=60, alpha=0.7, zorder=3,
                       edgecolors="white", lw=0.8)
            ax.set_ylabel("Learned $\\tau$", fontsize=13)
            tau_m, tau_s = np.mean(tau_vals), np.std(tau_vals)
            ax.set_title(f"$\\tau$ on Electricity: {tau_m:.3f} +/- {tau_s:.3f}",
                         fontsize=13, fontweight="bold")
            ax.grid(True, axis="y", ls=":", alpha=0.4)
            plt.tight_layout()
            save_fig(os.path.join(out, "figC3_tau_real"))

    # --- C4: Omega visualization ---
    if omega_data is not None:
        Omega = omega_data["Omega"]
        P_cal = omega_data["P_cal"]
        P_raw = omega_data["P_raw"]
        names = omega_data["names"]
        tau_val = omega_data["tau"]
        dd = min(Omega.shape[0], 20)
        idx = list(range(dd))
        Om, Pc, Pr = Omega[np.ix_(idx, idx)], P_cal[np.ix_(idx, idx)], P_raw[np.ix_(idx, idx)]
        nm = [names[i] for i in idx]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for ax, mat, title, cmap_n in [
            (axes[0], Pr, "Raw Prior $P$", "RdYlBu_r"),
            (axes[1], Pc, f"Calibrated $\\hat{{P}}(\\tau={tau_val:.2f})$", "RdYlBu_r"),
            (axes[2], Om, "$\\Omega(\\tau)$", "YlOrRd"),
        ]:
            im = ax.imshow(mat, cmap=cmap_n, aspect="auto")
            ax.set_xticks(range(dd))
            ax.set_xticklabels(nm, rotation=90, fontsize=7)
            ax.set_yticks(range(dd))
            ax.set_yticklabels(nm, fontsize=7)
            ax.set_title(title, fontsize=11, fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f"Prior Calibration Pipeline (learned $\\tau$={tau_val:.3f})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_fig(os.path.join(out, "figC4_omega"))


# ====================================================================
# Part D: Convergence Analysis
# ====================================================================
def run_part_D(cfg: Cfg) -> pd.DataFrame:
    print("\n" + "=" * 64)
    print(" Part D: Convergence Analysis")
    print("=" * 64)

    d, T, K = cfg.conv_d, cfg.conv_T, cfg.conv_K
    all_logs = []

    for acc in cfg.conv_prior_accs:
        for seed in cfg.seeds:
            data = _make_data(d, T, K, seed, cfg.edge_prob, "laplace")
            if data is None:
                continue
            P_prior = gen_prior(data["W0"], data["Wk"], acc, "random",
                                seed=seed + 999)
            X_t, X_lags = make_lag_tensors(data["X"], K)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_t, X_lags = X_t.to(dev), [x.to(dev) for x in X_lags]

            # PRCD-MAP (learn_tau) with logging
            set_seed(seed)
            model = _build_model(d, K, P_prior, learn_tau=True,
                                 lambda1=cfg.lambda1, lambda2=cfg.lambda2)
            try:
                W0, Wk, tau, log_df = train_prcd_alm_with_logging(
                    model, X_t, X_lags,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, tau_warmup=2)
                log_df["prior_acc"] = acc
                log_df["seed"] = seed
                log_df["method"] = "PRCD-MAP(learn_tau)"
                all_logs.append(log_df)
            except Exception as e:
                warnings.warn(f"Convergence learn_tau acc={acc} seed={seed}: {e}")

            # PRCD-MAP (fixed_tau) with logging — for convergence comparison
            set_seed(seed)
            model_ft = _build_model(d, K, P_prior, learn_tau=False, tau0=1.0,
                                    lambda1=cfg.lambda1, lambda2=cfg.lambda2)
            try:
                W0, Wk, tau, log_df = train_prcd_alm_with_logging(
                    model_ft, X_t, X_lags,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, tau_warmup=0)
                log_df["prior_acc"] = acc
                log_df["seed"] = seed
                log_df["method"] = "PRCD-MAP(fixed_tau)"
                all_logs.append(log_df)
            except Exception as e:
                warnings.warn(f"Convergence fixed_tau acc={acc} seed={seed}: {e}")

        print(f"  acc={acc} done")

    if not all_logs:
        print("  No convergence logs.")
        return pd.DataFrame()

    df = pd.concat(all_logs, ignore_index=True)
    p = os.path.join(cfg.output_dir, "partD_convergence.csv")
    df.to_csv(p, index=False)
    print(f">>> {p}")
    return df


def plot_part_D(df: pd.DataFrame, cfg: Cfg):
    if df.empty:
        return
    out = cfg.output_dir

    learn = df[df["method"] == "PRCD-MAP(learn_tau)"]
    fixed = df[df["method"] == "PRCD-MAP(fixed_tau)"]
    accs = sorted(learn["prior_acc"].unique())
    acc_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(accs), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: h(W0) trajectory — learn_tau (solid) vs fixed_tau (dashed)
    ax = axes[0][0]
    for idx, acc in enumerate(accs):
        sub_l = learn[learn["prior_acc"] == acc]
        agg_l = sub_l.groupby("outer_iter")["h_val"].agg(["mean", "std"]).reset_index()
        ax.semilogy(agg_l["outer_iter"], np.abs(agg_l["mean"]) + 1e-12,
                    marker="o", lw=2, ms=5, color=acc_colors[idx],
                    label=f"learn acc={acc:.1f}")
        sub_f = fixed[fixed["prior_acc"] == acc]
        if not sub_f.empty:
            agg_f = sub_f.groupby("outer_iter")["h_val"].agg(["mean", "std"]).reset_index()
            ax.semilogy(agg_f["outer_iter"], np.abs(agg_f["mean"]) + 1e-12,
                        marker="x", lw=1.5, ms=5, ls="--", color=acc_colors[idx],
                        label=f"fixed acc={acc:.1f}")
    ax.axhline(y=1e-6, color="red", ls="--", lw=1, alpha=0.7, label="tol")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("|h(W0)| (log)")
    ax.set_title("DAG Constraint: learn vs fixed", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 2: tau trajectory
    ax = axes[0][1]
    for idx, acc in enumerate(accs):
        sub = learn[learn["prior_acc"] == acc]
        agg = sub.groupby("outer_iter")["tau"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["outer_iter"], agg["mean"], yerr=agg["std"],
                    marker="s", lw=2, ms=5, color=acc_colors[idx],
                    capsize=2, label=f"acc={acc:.1f}")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("$\\tau$")
    ax.set_title("$\\tau$ Evolution (learn\\_tau)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 3: total loss — learn (solid) vs fixed (dashed)
    ax = axes[1][0]
    for idx, acc in enumerate(accs):
        sub_l = learn[learn["prior_acc"] == acc]
        agg_l = sub_l.groupby("outer_iter")["loss_alm"].mean().reset_index()
        ax.plot(agg_l["outer_iter"], agg_l["loss_alm"],
                marker="D", lw=2, ms=4, color=acc_colors[idx],
                label=f"learn acc={acc:.1f}")
        sub_f = fixed[fixed["prior_acc"] == acc]
        if not sub_f.empty:
            agg_f = sub_f.groupby("outer_iter")["loss_alm"].mean().reset_index()
            ax.plot(agg_f["outer_iter"], agg_f["loss_alm"],
                    marker="x", lw=1.5, ms=4, ls="--", color=acc_colors[idx],
                    label=f"fixed acc={acc:.1f}")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Total ALM Loss")
    ax.set_title("Loss: learn vs fixed", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 4: loss decomposition (middle accuracy)
    ax = axes[1][1]
    if accs:
        mid_acc = accs[len(accs) // 2]
        sub_mid = learn[learn["prior_acc"] == mid_acc]
        loss_cols = ["loss_mse", "loss_l1", "loss_prior"]
        avail = [c for c in loss_cols if c in sub_mid.columns]
        if avail:
            agg_mid = sub_mid.groupby("outer_iter")[avail].mean().reset_index()
            comp_c = {"loss_mse": "#3498DB", "loss_l1": "#E67E22",
                      "loss_prior": "#E74C3C"}
            comp_l = {"loss_mse": "MSE", "loss_l1": "L1",
                      "loss_prior": "Prior"}
            for col in avail:
                ax.plot(agg_mid["outer_iter"], agg_mid[col],
                        marker="o", lw=2, ms=4,
                        color=comp_c.get(col, "grey"),
                        label=comp_l.get(col, col))
            ax.set_xlabel("Outer Iteration")
            ax.set_ylabel("Loss Component")
            ax.set_title(f"Decomposition (acc={mid_acc:.1f})",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, ls=":", alpha=0.5)
    else:
        ax.set_visible(False)

    fig.suptitle("Convergence Analysis: PRCD-MAP Training Dynamics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(os.path.join(out, "figD_convergence"))


# ====================================================================
# Part E: Laplace Approximation Validation
# ====================================================================

def run_part_E(cfg: Cfg) -> pd.DataFrame:
    """
    Validate Laplace approximation of EB objective against grid-sweep exact computation.
    For each (prior_acc, seed): train → grid sweep τ → compare Laplace vs exact.
    """
    print("\n" + "=" * 64)
    print(" Part E: Laplace Approximation Validation")
    print("=" * 64)

    d, T, K = cfg.laplace_d, cfg.laplace_T, cfg.laplace_K
    rows = []

    for acc in cfg.laplace_prior_accs:
        for seed in cfg.seeds:
            print(f"  acc={acc}, seed={seed} ...", end="", flush=True)
            W0 = make_er_dag(d, cfg.edge_prob, seed=seed)
            Wk = make_lag_matrices(d, K, seed=seed)
            X = simulate_svar_linear(T, W0, Wk, "gaussian", seed=seed)
            if X is None or not np.all(np.isfinite(X)):
                print(" skip (non-finite)")
                continue
            X_std = standardize(X)
            P_prior = gen_prior(W0, Wk, acc, "random", seed=seed + 999)

            try:
                result = validate_laplace_approximation(
                    X_std, d, K, P_prior,
                    lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    seed=seed, max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, n_grid=cfg.laplace_n_grid,
                )
                rows.append(dict(
                    d=d, T=T, K=K, prior_acc=acc, seed=seed,
                    tau_laplace_opt=result["tau_laplace_opt"],
                    tau_exact_opt=result["tau_exact_opt"],
                    gap=result["gap"],
                    relative_error=result["relative_error"],
                ))
                print(f" gap={result['gap']:.4f}, "
                      f"τ_laplace={result['tau_laplace_opt']:.3f}, "
                      f"τ_exact={result['tau_exact_opt']:.3f}")
            except Exception as e:
                warnings.warn(f"  Laplace validation failed: {e}")
                print(f" FAILED: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = os.path.join(cfg.output_dir, "partE_laplace_validation.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved -> {csv_path}")
    return df


def plot_part_E(df: pd.DataFrame, cfg: Cfg):
    """Plot Laplace approximation quality: gap by prior_acc."""
    if df.empty:
        return

    ensure_dir(cfg.output_dir)

    # Figure 1: Gap vs prior accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: |τ_exact - τ_laplace| vs prior_acc
    ax = axes[0]
    agg = df.groupby("prior_acc")["gap"].agg(["mean", "std"]).reset_index()
    ax.bar(agg["prior_acc"].astype(str), agg["mean"],
           yerr=agg["std"], color="#3498DB", capsize=4, alpha=0.8)
    ax.set_xlabel("Prior Accuracy", fontsize=12)
    ax.set_ylabel("|τ_exact - τ_Laplace|", fontsize=12)
    ax.set_title("Laplace Approximation Gap", fontsize=13, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5)

    # Right: relative error vs prior_acc
    ax = axes[1]
    agg2 = df.groupby("prior_acc")["relative_error"].agg(["mean", "std"]).reset_index()
    ax.bar(agg2["prior_acc"].astype(str), agg2["mean"],
           yerr=agg2["std"], color="#E74C3C", capsize=4, alpha=0.8)
    ax.set_xlabel("Prior Accuracy", fontsize=12)
    ax.set_ylabel("Relative Error", fontsize=12)
    ax.set_title("EB Objective Relative Error", fontsize=13, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    save_fig(os.path.join(cfg.output_dir, "figE_laplace_validation"))

    # Summary table
    print("\n--- Laplace Approximation Validation Summary ---")
    summary = df.groupby("prior_acc").agg(
        gap_mean=("gap", "mean"), gap_std=("gap", "std"),
        rel_err_mean=("relative_error", "mean"),
        tau_laplace_mean=("tau_laplace_opt", "mean"),
        tau_exact_mean=("tau_exact_opt", "mean"),
    ).round(4)
    print(summary.to_string())


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Exp5: Scalability & Sensitivity")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--part", nargs="+", type=str, default=None,
                        choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dims", nargs="+", type=int, default=None)
    parser.add_argument("--no-pcmci", action="store_true")
    parser.add_argument("--no-varlingam", action="store_true")
    parser.add_argument("--no-dynotears", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip all external baselines, only run PRCD-MAP variants")
    args = parser.parse_args()

    if args.quick:
        cfg = cfg_quick()
    elif args.full:
        cfg = cfg_full()
    else:
        cfg = Cfg()

    if args.output:       cfg.output_dir   = args.output
    if args.seeds:        cfg.seeds        = args.seeds
    if args.dims:         cfg.scale_dims   = args.dims
    if args.part:         cfg.parts        = args.part
    if args.skip_baselines:
        cfg.do_dynotears = False
        cfg.do_pcmci = False
        cfg.do_varlingam = False
    if args.no_pcmci:     cfg.do_pcmci     = False
    if args.no_varlingam: cfg.do_varlingam = False
    if args.no_dynotears: cfg.do_dynotears = False

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 64)
    print(" Experiment 5: Scalability & Sensitivity")
    print("=" * 64)
    print(f"  parts  = {cfg.parts}")
    print(f"  seeds  = {cfg.seeds}")
    print(f"  output = {cfg.output_dir}")
    if "A" in cfg.parts:
        print(f"  dims   = {cfg.scale_dims}")
    print("=" * 64)

    t0 = time.time()
    results = {}

    if "A" in cfg.parts:
        df_a = run_part_A(cfg)
        plot_part_A(df_a, cfg)
        results["A"] = df_a

    if "B" in cfg.parts:
        df_b = run_part_B(cfg)
        plot_part_B(df_b, cfg)
        results["B"] = df_b

    if "C" in cfg.parts:
        df_synth, df_traj, df_real, omega_data = run_part_C(cfg)
        plot_part_C(df_synth, df_traj, df_real, omega_data, cfg)
        results["C"] = df_synth

    if "D" in cfg.parts:
        df_d = run_part_D(cfg)
        plot_part_D(df_d, cfg)
        results["D"] = df_d

    if "E" in cfg.parts:
        df_e = run_part_E(cfg)
        plot_part_E(df_e, cfg)
        results["E"] = df_e

    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f" Experiment 5 complete in {fmt_time(elapsed)} ({elapsed:.1f}s)")
    print("=" * 64)
    for name, df in results.items():
        n = len(df) if df is not None and not df.empty else 0
        print(f"  Part {name}: {n} rows")
    print(f"\n>>> All results in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
