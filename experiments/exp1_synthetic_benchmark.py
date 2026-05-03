"""
=============================================================================
Experiment 1 — Synthetic Benchmark for PRCD-MAP (RHINO-style)
=============================================================================
Covers: ER/BA graphs, 4 noise types, nonlinearity, prior corruption sweep.
Baselines: DYNOTEARS, PCMCI+, VARLiNGAM, PRCD-MAP variants.

Usage:
  python exp1_full_benchmark.py --sub prior --seeds 0 1 2
  python exp1_full_benchmark.py --sub noise --seeds 0 1 2
  python exp1_full_benchmark.py --sub graph --seeds 0 1 2
  python exp1_full_benchmark.py --sub nonlinear --seeds 0 1 2
  python exp1_full_benchmark.py --sub scale --seeds 0 1 2
  python exp1_full_benchmark.py --quick
=============================================================================
"""

import os, sys, time, warnings, argparse, traceback
from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd

from utils import *


# ====================================================================
# Configuration
# ====================================================================

@dataclass
class Cfg:
    dims:          List[int]   = field(default_factory=lambda: [10, 20])
    sample_sizes:  List[int]   = field(default_factory=lambda: [500])
    graph_types:   List[str]   = field(default_factory=lambda: ["ER"])
    noise_types:   List[str]   = field(default_factory=lambda: ["gaussian"])
    lag_orders:    List[int]   = field(default_factory=lambda: [1])
    nonlinear:     bool        = False
    prior_accs:    List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior_modes:   List[str]   = field(default_factory=lambda: ["random"])
    seeds:         List[int]   = field(default_factory=lambda: [0, 1, 2, 3, 4])
    missing_ratios: List[float] = field(default_factory=lambda: [0.0])
    edge_prob_w0:  float = 0.15
    ba_m:          int   = 2
    edge_prob_wk:  float = 0.10
    lambda1:       float = 0.001
    lambda2:       float = 0.01
    max_iter:      int   = 35
    inner_iter:    int   = 400
    lr:            float = 8e-3
    do_dynotears:  bool = True
    do_pcmci:      bool = True
    do_varlingam:  bool = True
    do_rhino:      bool = True
    do_ngc:        bool = True
    do_nam:        bool = True
    output_dir:    str  = "exp1_results"


def cfg_quick():
    return Cfg(dims=[10], sample_sizes=[500], seeds=[0, 1],
               prior_accs=[0.2, 0.6, 1.0], max_iter=10, inner_iter=100,
               do_pcmci=False, do_varlingam=False,
               do_rhino=False, do_ngc=False, do_nam=False)


def cfg_sub(name: str):
    if name == "noise":
        # Paper §5.2 Table 1: 4 noise × 4 accs × 3 seeds = 48 settings
        return Cfg(
            dims=[20], sample_sizes=[500],
            noise_types=["gaussian", "laplace", "student_t", "heteroscedastic"],
            prior_accs=[0.0, 0.4, 0.6, 0.9],
            seeds=[0, 1, 2],
            output_dir="exp1_noise",
        )
    elif name == "sample_size":
        # Paper §5.2 Table 2 + Figure 2 (CORE):
        # 4 sample sizes × 3 accs × 3 seeds = 36 settings
        return Cfg(
            dims=[20], sample_sizes=[50, 100, 200, 500],
            prior_accs=[0.4, 0.6, 0.9],
            seeds=[0, 1, 2],
            output_dir="exp1_sample_size",
        )
    elif name == "nonlinear":
        # Appendix A: 2 dims × 3 accs × 3 seeds = 18 settings
        return Cfg(
            dims=[10, 20], sample_sizes=[500],
            nonlinear=True,
            noise_types=["gaussian"],
            prior_accs=[0.2, 0.6, 1.0],
            seeds=[0, 1, 2],
            output_dir="exp1_nonlinear",
        )
    elif name == "scale":
        # Appendix B: 3 dims × 1 acc × 3 seeds = 9 settings
        return Cfg(
            dims=[10, 20, 50],
            sample_sizes=[500],
            prior_accs=[0.6],
            seeds=[0, 1, 2],
            do_pcmci=False,
            output_dir="exp1_scale",
        )
    else:
        raise ValueError(f"Unknown sub-experiment: {name}")


# ====================================================================
# Experiment Loop
# ====================================================================

def _setting_key(st: dict) -> str:
    mr = st.get("missing_ratio", 0.0)
    return (f"d={st['d']}_T={st['T']}_{st['graph']}_{st['noise']}_K={st['K']}"
            f"_acc={st['prior_acc']}_{st['prior_mode']}_nl={st['nonlinear']}"
            f"_mr={mr}_s={st['seed']}")


def _build_settings(cfg: Cfg) -> list:
    settings = []
    for d in cfg.dims:
        for T in cfg.sample_sizes:
            for gt in cfg.graph_types:
                for nt in cfg.noise_types:
                    for K in cfg.lag_orders:
                        for acc in cfg.prior_accs:
                            for pm in cfg.prior_modes:
                                for mr in cfg.missing_ratios:
                                    for s in cfg.seeds:
                                        settings.append(dict(
                                            d=d, T=T, graph=gt, noise=nt, K=K,
                                            prior_acc=acc, prior_mode=pm,
                                            missing_ratio=mr,
                                            seed=s, nonlinear=False))
    if cfg.nonlinear:
        for d in cfg.dims:
            for T in cfg.sample_sizes:
                for gt in cfg.graph_types:
                    for K in cfg.lag_orders:
                        for acc in cfg.prior_accs:
                            for s in cfg.seeds:
                                settings.append(dict(
                                    d=d, T=T, graph=gt, noise="gaussian", K=K,
                                    prior_acc=acc, prior_mode="random",
                                    missing_ratio=0.0,
                                    seed=s, nonlinear=True))
    return settings


def _load_checkpoint(output_dir: str):
    ckpt_path = os.path.join(output_dir, "_intermediate.csv")
    if os.path.exists(ckpt_path):
        try:
            df_ckpt = pd.read_csv(ckpt_path)
            if len(df_ckpt) > 0:
                done_keys = set()
                for _, row in df_ckpt.iterrows():
                    key = (f"d={int(row['d'])}_T={int(row['T'])}_{row['graph']}"
                           f"_{row['noise']}_K={int(row['K'])}"
                           f"_acc={row['prior_acc']}_{row['prior_mode']}"
                           f"_nl={row['nonlinear']}_s={int(row['seed'])}")
                    done_keys.add(key)
                rows = df_ckpt.to_dict("records")
                print(f">>> [RESUME] {len(done_keys)} settings, {len(rows)} rows")
                return done_keys, rows
        except Exception as e:
            warnings.warn(f"Checkpoint load failed: {e}")
    return set(), []


def run_experiment(cfg: Cfg) -> pd.DataFrame:
    ensure_dir(cfg.output_dir)
    settings = _build_settings(cfg)
    n_total = len(settings)
    done_keys, all_rows = _load_checkpoint(cfg.output_dir)

    method_names = ["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)",
                     "PRCD-MAP(no_l1_weight)", "PRCD-MAP(uniform)"]
    if cfg.do_dynotears:
        method_names.append("DYNOTEARS")
    if cfg.do_pcmci and HAS_TIGRAMITE:
        method_names.append("PCMCI+")
    if cfg.do_varlingam and HAS_LINGAM:
        method_names.append("VARLiNGAM")
    if cfg.do_rhino and (HAS_RHINO or HAS_CUTS):
        method_names.append("RHINO")
    if cfg.do_ngc:
        method_names.append("NGC")
    if cfg.do_nam:
        method_names.append("PRCD-MAP(NAM)")

    print(f">>> {n_total} settings x {len(method_names)} methods")
    t_global = time.time()
    ckpt_path = os.path.join(cfg.output_dir, "_intermediate.csv")
    n_skipped = 0

    for idx, st in enumerate(settings):
        d, T, K, seed = st["d"], st["T"], st["K"], st["seed"]
        gt, nt = st["graph"], st["noise"]
        acc, pm = st["prior_acc"], st["prior_mode"]
        nl = st["nonlinear"]
        mr = st.get("missing_ratio", 0.0)
        skey = _setting_key(st)

        if skey in done_keys:
            n_skipped += 1
            continue

        elapsed = time.time() - t_global
        done_so_far = (idx + 1) - n_skipped
        if done_so_far > 1:
            eta = (n_total - idx - 1) * elapsed / done_so_far
            eta_str = f" ETA {fmt_time(eta)}"
        else:
            eta_str = ""
        mr_str = f" mr={mr}" if mr > 0 else ""
        print(f"  [{idx+1}/{n_total}] d={d} T={T} {gt} {nt} K={K} "
              f"acc={acc} {pm} nl={nl}{mr_str} s={seed}{eta_str}")

        base = dict(d=d, T=T, graph=gt, noise=nt, K=K,
                    prior_acc=acc, prior_mode=pm, seed=seed, nonlinear=nl,
                    missing_ratio=mr)

        # Ground truth
        W0_true = make_er_dag(d, cfg.edge_prob_w0, seed=seed) if gt == "ER" \
            else make_ba_dag(d, cfg.ba_m, seed=seed)
        Wk_true = make_lag_matrices(d, K, cfg.edge_prob_wk, seed=seed)

        # Simulate
        try:
            if nl:
                X = simulate_svar_nonlinear(T, W0_true, Wk_true, nt, seed=seed)
            else:
                X = simulate_svar_linear(T, W0_true, Wk_true, nt, seed=seed)
        except Exception:
            X = None

        if X is None or not np.all(np.isfinite(X)) or X.std() < 1e-10:
            done_keys.add(skey)
            continue

        # Inject missing data (MCAR) if requested
        X_clean = X  # Keep clean copy for baselines that need imputation
        if mr > 0:
            X = inject_missing(X, ratio=mr, seed=seed + 7777)
            # For baselines (PCMCI+, VARLiNGAM etc.): mean imputation
            X_imputed = X.copy()
            col_means = np.nanmean(X_imputed, axis=0)
            for j in range(X_imputed.shape[1]):
                mask_col = np.isnan(X_imputed[:, j])
                X_imputed[mask_col, j] = col_means[j]
        else:
            X_imputed = X

        P_prior = gen_prior(W0_true, Wk_true, acc, pm, seed=seed + 999)
        rows_this = []

        # PRCD-MAP (learn tau)
        try:
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                learn_tau=True, seed=seed,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr,
                loss_type="huber", prior_l1_weight=True)
            met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
            rows_this.append({**base, "method": "PRCD-MAP(learn_tau)",
                              "tau": float(tau), "time": time.time()-t0, **met})
        except Exception as e:
            warnings.warn(f"  PRCD-MAP(learn_tau) failed: {e}")

        # PRCD-MAP (fixed tau=1) -- Fix 8: keep prior_l1_weight=True
        # Differs from learn_tau only in whether tau is learned -- single-variable ablation
        try:
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                learn_tau=False, tau0=1.0, seed=seed,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr,
                loss_type="huber", prior_l1_weight=True)
            met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
            rows_this.append({**base, "method": "PRCD-MAP(fixed_tau)",
                              "tau": float(tau), "time": time.time()-t0, **met})
        except Exception as e:
            warnings.warn(f"  PRCD-MAP(fixed_tau) failed: {e}")

        # PRCD-MAP (no prior_l1) -- Fix 8: ablate prior_l1_weight in isolation
        try:
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                learn_tau=True, seed=seed,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr,
                loss_type="huber", prior_l1_weight=False)
            met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
            rows_this.append({**base, "method": "PRCD-MAP(no_l1_weight)",
                              "tau": float(tau), "time": time.time()-t0, **met})
        except Exception as e:
            warnings.warn(f"  PRCD-MAP(no_l1_weight) failed: {e}")

        # PRCD-MAP (uniform prior)
        try:
            P_unif = np.full((d, d), 0.5)
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_unif, d, K, cfg.lambda1, cfg.lambda2,
                learn_tau=True, seed=seed,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr,
                loss_type="huber", prior_l1_weight=False)
            met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
            rows_this.append({**base, "method": "PRCD-MAP(uniform)",
                              "tau": float(tau), "time": time.time()-t0, **met})
        except Exception as e:
            warnings.warn(f"  PRCD-MAP(uniform) failed: {e}")

        # DYNOTEARS (uses imputed data if missing)
        if cfg.do_dynotears:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_dynotears(X_imputed, d, K, cfg.lambda1,
                                                cfg.max_iter, cfg.inner_iter,
                                                cfg.lr, seed)
                met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
                rows_this.append({**base, "method": "DYNOTEARS",
                                  "tau": np.nan, "time": time.time()-t0, **met})
            except Exception as e:
                warnings.warn(f"  DYNOTEARS failed: {e}")

        # PCMCI+
        if cfg.do_pcmci and HAS_TIGRAMITE and d <= 50:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_pcmci_plus(X_imputed, d, K, seed=seed)
                if W0_est is not None:
                    met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
                    rows_this.append({**base, "method": "PCMCI+",
                                      "tau": np.nan, "time": time.time()-t0, **met})
            except Exception:
                pass

        # VARLiNGAM
        if cfg.do_varlingam and HAS_LINGAM:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_varlingam(X_imputed, d, K, seed=seed)
                if W0_est is not None:
                    met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
                    rows_this.append({**base, "method": "VARLiNGAM",
                                      "tau": np.nan, "time": time.time()-t0, **met})
            except Exception:
                pass

        # RHINO (official codebase / CUTS+ fallback)
        if cfg.do_rhino and (HAS_RHINO or HAS_CUTS):
            try:
                t0 = time.time()
                W0_est, Wk_est = run_rhino(X, d, K, seed=seed)
                if W0_est is not None:
                    met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
                    rows_this.append({**base, "method": "RHINO",
                                      "tau": np.nan, "time": time.time()-t0, **met})
            except Exception as e:
                warnings.warn(f"  RHINO failed: {e}")

        # NGC (Neural Granger Causality)
        if cfg.do_ngc:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_ngc(X, d, K, seed=seed)
                if W0_est is not None:
                    met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
                    rows_this.append({**base, "method": "NGC",
                                      "tau": np.nan, "time": time.time()-t0, **met})
            except Exception as e:
                warnings.warn(f"  NGC failed: {e}")

        # PRCD-MAP (NAM nonlinear extension)
        if cfg.do_nam and d <= 30:
            try:
                t0 = time.time()
                W0_est, Wk_est, tau = run_prcd_map_nam(
                    X, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                    learn_tau=True, seed=seed,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=5e-4)
                met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
                rows_this.append({**base, "method": "PRCD-MAP(NAM)",
                                  "tau": float(tau), "time": time.time()-t0, **met})
            except Exception as e:
                warnings.warn(f"  PRCD-MAP(NAM) failed: {e}")

        all_rows.extend(rows_this)
        done_keys.add(skey)

        # Periodic checkpoint
        if (idx + 1) % 5 == 0 and all_rows:
            pd.DataFrame(all_rows).to_csv(ckpt_path, index=False)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(os.path.join(cfg.output_dir, "all_results.csv"), index=False)

    return df


# ====================================================================
# Summary Tables & Plots
# ====================================================================

def generate_summary(df: pd.DataFrame, cfg: Cfg):
    if df.empty:
        print("  No results to summarize.")
        return

    ensure_dir(cfg.output_dir)

    # RHINO-style table: AUROC by method x prior_acc
    # Fix 11: report both w0_ and comb_ metrics
    if "prior_acc" in df.columns and df["prior_acc"].nunique() > 1:
        df_with_setting = df.copy()
        df_with_setting["setting"] = df_with_setting["prior_acc"].apply(
            lambda x: f"acc={x:.1f}")
        for metric in ["comb_auroc", "comb_f1_opt", "w0_auroc", "w0_f1_opt",
                        "auroc", "f1_opt", "shd"]:
            if metric in df.columns:
                print_rhino_table(df_with_setting, metric=metric,
                                  group_col="method", setting_col="setting",
                                  title=f"{metric.upper()} by Method x Prior Accuracy")

    # AUROC by method x noise type
    if "noise" in df.columns and df["noise"].nunique() > 1:
        df_noise = df.copy()
        df_noise["setting"] = df_noise["noise"]
        print_rhino_table(df_noise, metric="auroc",
                          group_col="method", setting_col="setting",
                          title="AUROC by Method x Noise Type")

    # AUROC by method x graph type
    if "graph" in df.columns and df["graph"].nunique() > 1:
        df_graph = df.copy()
        df_graph["setting"] = df_graph["graph"]
        print_rhino_table(df_graph, metric="auroc",
                          group_col="method", setting_col="setting",
                          title="AUROC by Method x Graph Type")

    # Overall summary — Fix 11: report both w0 and combined metrics
    summary_cols = []
    for prefix in ["", "w0_", "comb_"]:
        for m in ["auroc", "f1_opt", "shd"]:
            col = f"{prefix}{m}"
            if col in df.columns:
                summary_cols.append(col)
    summary_cols = list(dict.fromkeys(summary_cols))  # dedup

    agg_dict = {}
    for col in summary_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")
    agg_dict["time_mean"] = ("time", "mean")

    summary = df.groupby("method").agg(**agg_dict).round(4)
    print("\n--- Overall Summary ---")
    print(summary.to_string())
    summary.to_csv(os.path.join(cfg.output_dir, "summary.csv"))

    # Statistical significance tests (Fix 17)
    try:
        sig_df = compute_significance(
            df, metric="auroc", method_col="method",
            group_cols=["d", "T", "graph", "noise", "K", "prior_acc",
                         "prior_mode", "seed", "nonlinear"])
        if not sig_df.empty:
            sig_df.to_csv(os.path.join(cfg.output_dir, "significance_tests.csv"),
                          index=False)
            print("\n--- Pairwise Significance (Wilcoxon) ---")
            for _, row in sig_df.iterrows():
                star = "***" if row["p_value"] < 0.001 else \
                       "**" if row["p_value"] < 0.01 else \
                       "*" if row["p_value"] < 0.05 else "n.s."
                print(f"  {row['method1']} vs {row['method2']}: "
                      f"p={row['p_value']:.4f} {star} "
                      f"(mean_diff={row['mean_diff']:.4f})")
    except Exception as e:
        warnings.warn(f"Significance test failed: {e}")

    # Prior accuracy sweep plot
    if "prior_acc" in df.columns and df["prior_acc"].nunique() > 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for metric, ax, title in [("auroc", axes[0], "AUROC"),
                                   ("f1_opt", axes[1], "Best F1")]:
            for m in sorted(df["method"].unique()):
                sub = df[df["method"] == m]
                agg = sub.groupby("prior_acc")[metric].agg(["mean", "std"]).reset_index()
                ax.errorbar(agg["prior_acc"], agg["mean"], yerr=agg["std"],
                            label=m, marker=MARKERS.get(m, "x"),
                            color=COLORS.get(m, "grey"), lw=2, capsize=3)
            ax.set_xlabel("Prior Accuracy", fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f"{title} vs Prior Quality", fontsize=13, fontweight="bold")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, ls=":", alpha=0.5)
        plt.tight_layout()
        save_fig(os.path.join(cfg.output_dir, "fig_prior_sweep"))


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp1: Synthetic Benchmark")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sub", type=str, default=None,
                        choices=["noise", "sample_size", "nonlinear", "scale"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--Ts", type=int, nargs="+", default=None)
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip all external baselines (DYNOTEARS/PCMCI+/VARLiNGAM/RHINO/NGC/NAM), only run PRCD-MAP variants")
    parser.add_argument("--no-rhino", action="store_true")
    parser.add_argument("--no-ngc", action="store_true")
    parser.add_argument("--no-nam", action="store_true")
    parser.add_argument("--no-pcmci", action="store_true")
    parser.add_argument("--no-varlingam", action="store_true")
    parser.add_argument("--no-dynotears", action="store_true")
    args = parser.parse_args()

    if args.quick:
        cfg = cfg_quick()
    elif args.sub:
        cfg = cfg_sub(args.sub)
    else:
        cfg = Cfg()

    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.dims is not None:
        cfg.dims = args.dims
    if args.Ts is not None:
        cfg.sample_sizes = args.Ts
    if args.skip_baselines:
        cfg.do_dynotears = False
        cfg.do_pcmci = False
        cfg.do_varlingam = False
        cfg.do_rhino = False
        cfg.do_ngc = False
        cfg.do_nam = False
    if args.no_rhino:      cfg.do_rhino = False
    if args.no_ngc:        cfg.do_ngc = False
    if args.no_nam:        cfg.do_nam = False
    if args.no_pcmci:      cfg.do_pcmci = False
    if args.no_varlingam:  cfg.do_varlingam = False
    if args.no_dynotears:  cfg.do_dynotears = False

    print(f">>> Exp1: dims={cfg.dims}, T={cfg.sample_sizes}, seeds={cfg.seeds}")
    print(f">>> Output: {cfg.output_dir}")

    t0 = time.time()
    df = run_experiment(cfg)
    print(f"\n>>> Total time: {fmt_time(time.time() - t0)}")

    generate_summary(df, cfg)
    print(">>> Exp1 complete.")


if __name__ == "__main__":
    main()
