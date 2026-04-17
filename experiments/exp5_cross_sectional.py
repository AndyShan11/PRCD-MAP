"""
exp6_cross_sectional.py — Generalization to Cross-Sectional Structure Learning

Demonstrates that learnable trust (PRCD-MAP) is a general principle, not a
temporal causal discovery trick.  Uses NOTEARS-style cross-sectional data
(no lag), optimizes only W₀.

Settings:
  d = 20,  n ∈ {100, 500},  prior_acc ∈ {0.4, 0.6, 0.9}
  Baselines: NOTEARS, NOTEARS + hard mask, DAGMA

Produces:
  results/exp6_cross_sectional/  — CSV + RHINO-style table
"""

import os, sys, time, warnings, math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    set_seed, make_er_dag, gen_prior_from_truth,
    binarize_prior_to_mask, compute_all_metrics,
    print_rhino_table, ensure_dir, fmt_time, save_fig,
)
from model_prcd_map import PRCD_MAP_Model, train_prcd_alm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ====================================================================
# Cross-sectional data generation (NOTEARS-style)
# ====================================================================

def simulate_cross_sectional(n: int, W0: np.ndarray,
                              noise_type: str = "gaussian",
                              noise_scale: float = 1.0,
                              seed: int = 0) -> np.ndarray:
    """
    Generate cross-sectional (i.i.d.) data from a linear SEM:
        X = X @ W0 + noise   =>   X = noise @ (I - W0)^{-1}

    This is exactly the NOTEARS data generation process.
    """
    rng = np.random.default_rng(seed)
    d = W0.shape[0]
    M = np.eye(d) - W0
    cond = np.linalg.cond(M)
    if cond > 1e10:
        warnings.warn(f"(I-W0) ill-conditioned (cond={cond:.1e})")
        return None

    A_inv = np.linalg.inv(M)

    if noise_type == "gaussian":
        noise = rng.normal(0, noise_scale, size=(n, d))
    elif noise_type == "laplace":
        noise = rng.laplace(0, noise_scale, size=(n, d))
    else:
        noise = rng.normal(0, noise_scale, size=(n, d))

    X = noise @ A_inv
    if not np.all(np.isfinite(X)):
        return None

    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


# ====================================================================
# Baselines: NOTEARS (cross-sectional)
# ====================================================================

class _NOTEARS(nn.Module):
    """Standard NOTEARS (Zheng et al. 2018) — cross-sectional."""

    def __init__(self, d: int, lam: float = 0.01):
        super().__init__()
        self.d = d
        self.lam = lam
        self.W = nn.Parameter(1e-2 * torch.randn(d, d))
        self.register_buffer("mask", 1.0 - torch.eye(d))

    def _adj(self):
        return self.W * self.mask

    def _h(self):
        A = torch.clamp(self._adj(), -3.0, 3.0)
        return torch.trace(torch.matrix_exp(A * A)) - self.d

    def loss(self, X, rho, alpha):
        A = self._adj()
        pred = X @ A
        mse = 0.5 * torch.sum((X - pred) ** 2) / X.shape[0]
        l1 = self.lam * torch.norm(A, p=1)
        h = self._h()
        return mse + l1 + alpha * h + 0.5 * rho * h ** 2, h


def run_notears(X_np, d, lam=0.01, max_outer=35, inner=400, lr=1e-2, seed=0):
    """NOTEARS baseline for cross-sectional data."""
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X_np, dtype=torch.float32).to(dev)
    m = _NOTEARS(d, lam).to(dev)
    opt = optim.Adam(m.parameters(), lr=lr)
    rho, alpha = 1.0, 0.0
    for _ in range(max_outer):
        for __ in range(inner):
            loss, h = m.loss(X, rho, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        h_val = float(m._h().detach())
        if abs(h_val) < 1e-6:
            break
        alpha += rho * h_val
        rho = min(rho * 3.0, 1e6)
    with torch.no_grad():
        W0 = m._adj().cpu().numpy()
    return W0


# ====================================================================
# Baseline: NOTEARS + hard mask
# ====================================================================

class _NOTEARS_HardMask(nn.Module):
    """NOTEARS with a binary prior mask: edges where mask=0 are forced to zero."""

    def __init__(self, d: int, hard_mask: np.ndarray, lam: float = 0.01):
        super().__init__()
        self.d = d
        self.lam = lam
        self.W = nn.Parameter(1e-2 * torch.randn(d, d))
        self.register_buffer("mask", 1.0 - torch.eye(d))
        self.register_buffer("hard_mask",
                             torch.tensor(hard_mask, dtype=torch.float32))

    def _adj(self):
        return self.W * self.mask * self.hard_mask

    def _h(self):
        A = torch.clamp(self._adj(), -3.0, 3.0)
        return torch.trace(torch.matrix_exp(A * A)) - self.d

    def loss(self, X, rho, alpha):
        A = self._adj()
        pred = X @ A
        mse = 0.5 * torch.sum((X - pred) ** 2) / X.shape[0]
        l1 = self.lam * torch.norm(A, p=1)
        h = self._h()
        return mse + l1 + alpha * h + 0.5 * rho * h ** 2, h


def run_notears_hard_mask(X_np, d, hard_mask, lam=0.01,
                           max_outer=35, inner=400, lr=1e-2, seed=0):
    """NOTEARS + hard binary mask baseline."""
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X_np, dtype=torch.float32).to(dev)
    m = _NOTEARS_HardMask(d, hard_mask, lam).to(dev)
    opt = optim.Adam(m.parameters(), lr=lr)
    rho, alpha = 1.0, 0.0
    for _ in range(max_outer):
        for __ in range(inner):
            loss, h = m.loss(X, rho, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        h_val = float(m._h().detach())
        if abs(h_val) < 1e-6:
            break
        alpha += rho * h_val
        rho = min(rho * 3.0, 1e6)
    with torch.no_grad():
        W0 = m._adj().cpu().numpy()
    return W0


# ====================================================================
# Baseline: DAGMA (cross-sectional)
# ====================================================================

class _DAGMA(nn.Module):
    """DAGMA (Bello et al. 2022) — cross-sectional."""

    def __init__(self, d: int, lam: float = 0.01, s: float = None):
        super().__init__()
        self.d = d
        self.lam = lam
        self.s = max(1.0, math.log(d)) if s is None else s
        self.W = nn.Parameter(1e-2 * torch.randn(d, d))
        self.register_buffer("mask", 1.0 - torch.eye(d))

    def _adj(self):
        return self.W * self.mask

    def _h(self):
        A = self._adj()
        W2 = A * A
        M = self.s * torch.eye(self.d, device=A.device) - W2
        sign, logabsdet = torch.linalg.slogdet(M)
        if sign.item() <= 0:
            excess = torch.clamp(W2.sum(dim=1) - self.s, min=0.0)
            return self.d * 1.0 + excess.sum()
        return -logabsdet + self.d * math.log(self.s)

    def loss(self, X, rho, alpha):
        A = self._adj()
        pred = X @ A
        mse = 0.5 * torch.sum((X - pred) ** 2) / X.shape[0]
        l1 = self.lam * torch.norm(A, p=1)
        h = self._h()
        return mse + l1 + alpha * h + 0.5 * rho * h ** 2, h


def run_dagma(X_np, d, lam=0.01, max_outer=35, inner=400, lr=1e-2, seed=0):
    """DAGMA baseline for cross-sectional data."""
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X_np, dtype=torch.float32).to(dev)
    m = _DAGMA(d, lam).to(dev)
    opt = optim.Adam(m.parameters(), lr=lr)
    rho, alpha = 1.0, 0.0
    for _ in range(max_outer):
        for __ in range(inner):
            loss, h = m.loss(X, rho, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        h_val = float(m._h().detach())
        if abs(h_val) < 1e-6:
            break
        alpha += rho * h_val
        rho = min(rho * 3.0, 1e6)
    with torch.no_grad():
        W0 = m._adj().cpu().numpy()
    return W0


# ====================================================================
# PRCD-MAP cross-sectional wrapper (lag=0, only W₀)
# ====================================================================

def run_prcd_map_cross_sectional(X_np, P_prior, d,
                                  lambda1=0.001, lambda2=0.01,
                                  learn_tau=True, tau0=1.0,
                                  max_iter=35, inner_iter=400,
                                  lr=1e-2, seed=0):
    """
    PRCD-MAP with K=0: no lag matrices, only W₀ optimization.
    This proves learnable trust is a general principle, not a temporal trick.
    """
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # K=0: X_t = full data, X_lags = empty list
    X_t = torch.tensor(X_np, dtype=torch.float32).to(dev)
    X_lags = []  # no lag matrices

    model = PRCD_MAP_Model(
        num_vars=d, lag_k=0, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau, tau0=tau0,
        tau_min=0.05, tau_max=3.0,
        loss_type="huber",
        prior_l1_weight=True,
        n_tau_groups=4,
    ).to(dev)

    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=False, postprocess=False,
    )

    return W0, tau


# ====================================================================
# Config
# ====================================================================

@dataclass
class Cfg:
    d: int = 20
    sample_sizes: List[int] = field(default_factory=lambda: [100, 500])
    prior_accs: List[float] = field(default_factory=lambda: [0.4, 0.6, 0.9])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    noise_type: str = "gaussian"
    edge_prob: float = 0.15
    out_dir: str = "results/exp6_cross_sectional"


# ====================================================================
# Main experiment
# ====================================================================

def run_experiment(cfg: Cfg = None):
    if cfg is None:
        cfg = Cfg()
    ensure_dir(cfg.out_dir)
    rows = []
    total = len(cfg.sample_sizes) * len(cfg.prior_accs) * len(cfg.seeds)
    done = 0
    t0 = time.time()

    for n in cfg.sample_sizes:
        for acc in cfg.prior_accs:
            for seed in cfg.seeds:
                done += 1
                tag = f"n={n}, acc={acc:.1f}, seed={seed}"
                print(f"\n[{done}/{total}] {tag}")

                # ---- Generate DAG + data ----
                set_seed(seed)
                W0_true = make_er_dag(cfg.d, edge_prob=cfg.edge_prob, seed=seed)
                X = simulate_cross_sectional(n, W0_true, noise_type=cfg.noise_type,
                                              seed=seed)
                if X is None:
                    print(f"  SKIP (data gen failed)")
                    continue

                B_true = (np.abs(W0_true) > 1e-10).astype(int)
                P_prior = gen_prior_from_truth(B_true, acc=acc, seed=seed)
                hard_mask = binarize_prior_to_mask(P_prior)
                setting = f"n={n},acc={acc:.1f}"

                # ---- 1. NOTEARS ----
                t1 = time.time()
                W0_notears = run_notears(X, cfg.d, seed=seed)
                dt_notears = time.time() - t1
                met = compute_all_metrics(B_true, W0_notears)
                met.update(method="NOTEARS", setting=setting,
                           n=n, acc=acc, seed=seed, time=dt_notears)
                rows.append(met)
                print(f"  NOTEARS:      AUROC={met['auroc']:.3f}  "
                      f"F1={met['f1_opt']:.3f}  SHD={met['shd']}")

                # ---- 2. NOTEARS + hard mask ----
                t1 = time.time()
                W0_hm = run_notears_hard_mask(X, cfg.d, hard_mask, seed=seed)
                dt_hm = time.time() - t1
                met = compute_all_metrics(B_true, W0_hm)
                met.update(method="NOTEARS+mask", setting=setting,
                           n=n, acc=acc, seed=seed, time=dt_hm)
                rows.append(met)
                print(f"  NOTEARS+mask: AUROC={met['auroc']:.3f}  "
                      f"F1={met['f1_opt']:.3f}  SHD={met['shd']}")

                # ---- 3. DAGMA ----
                t1 = time.time()
                W0_dagma = run_dagma(X, cfg.d, seed=seed)
                dt_dagma = time.time() - t1
                met = compute_all_metrics(B_true, W0_dagma)
                met.update(method="DAGMA", setting=setting,
                           n=n, acc=acc, seed=seed, time=dt_dagma)
                rows.append(met)
                print(f"  DAGMA:        AUROC={met['auroc']:.3f}  "
                      f"F1={met['f1_opt']:.3f}  SHD={met['shd']}")

                # ---- 4. PRCD-MAP (learn_tau, K=0) ----
                t1 = time.time()
                W0_prcd, tau_val = run_prcd_map_cross_sectional(
                    X, P_prior, cfg.d, seed=seed)
                dt_prcd = time.time() - t1
                met = compute_all_metrics(B_true, W0_prcd)
                met.update(method="PRCD-MAP(learn_tau)", setting=setting,
                           n=n, acc=acc, seed=seed, time=dt_prcd,
                           tau=float(tau_val))
                rows.append(met)
                print(f"  PRCD-MAP:     AUROC={met['auroc']:.3f}  "
                      f"F1={met['f1_opt']:.3f}  SHD={met['shd']}  "
                      f"tau={tau_val:.3f}")

    elapsed = time.time() - t0
    print(f"\n>>> Total time: {fmt_time(elapsed)}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.out_dir, "exp6_results.csv")
    df.to_csv(csv_path, index=False)
    print(f">>> Saved: {csv_path}")

    # ---- Print RHINO-style tables ----
    for metric in ["auroc", "auprc", "f1_opt", "shd"]:
        print_rhino_table(df, metric=metric,
                          title=f"Exp 6: Cross-Sectional — {metric.upper()}")

    # ---- Summary table: mean over seeds ----
    summary = (df.groupby(["method", "setting"])
               [["auroc", "auprc", "f1_opt", "shd"]]
               .agg(["mean", "std"])
               .round(3))
    summary_path = os.path.join(cfg.out_dir, "exp6_summary.csv")
    summary.to_csv(summary_path)
    print(f">>> Saved: {summary_path}")

    # ---- Plot: F1 vs prior accuracy for each n ----
    _plot_f1_vs_acc(df, cfg)

    return df


def _plot_f1_vs_acc(df, cfg):
    """Bar/line plot: F1 vs prior_accuracy, grouped by n."""
    colors = {
        "NOTEARS":              "#2C3E50",
        "NOTEARS+mask":         "#3498DB",
        "DAGMA":                "#27AE60",
        "PRCD-MAP(learn_tau)":  "#E74C3C",
    }
    markers = {
        "NOTEARS":              "D",
        "NOTEARS+mask":         "s",
        "DAGMA":                "^",
        "PRCD-MAP(learn_tau)":  "o",
    }
    methods = ["NOTEARS", "NOTEARS+mask", "DAGMA", "PRCD-MAP(learn_tau)"]

    for n_val in cfg.sample_sizes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax_idx, metric in enumerate(["f1_opt", "auroc"]):
            ax = axes[ax_idx]
            for meth in methods:
                sub = df[(df["method"] == meth) & (df["n"] == n_val)]
                if sub.empty:
                    continue
                agg = sub.groupby("acc")[metric].agg(["mean", "std"]).reset_index()
                ax.errorbar(agg["acc"], agg["mean"], yerr=agg["std"],
                            label=meth, color=colors.get(meth, "gray"),
                            marker=markers.get(meth, "o"),
                            capsize=3, linewidth=2, markersize=7)
            ax.set_xlabel("Prior Accuracy", fontsize=12)
            ax.set_ylabel(metric.upper().replace("_", "-"), fontsize=12)
            ax.set_title(f"d={cfg.d}, n={n_val}", fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.3, 1.0)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        prefix = os.path.join(cfg.out_dir, f"exp6_f1_auroc_n{n_val}")
        save_fig(prefix)


# ====================================================================
# Entry point
# ====================================================================

if __name__ == "__main__":
    cfg = Cfg()
    df = run_experiment(cfg)
