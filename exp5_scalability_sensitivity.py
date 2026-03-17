"""
=============================================================================
Experiment 5 — Scalability & Hyperparameter Sensitivity Analysis for PRCD-MAP
=============================================================================
NeurIPS-grade analysis covering three complementary axes:

  Part A: Scalability Timing
    Wall-clock runtime vs dimension d ∈ {10, 20, 50, 100, 200} for
    PRCD-MAP and all baselines. Log-log plot + table.

  Part B: Hyperparameter Sensitivity
    λ₁ × λ₂ grid on a representative synthetic setting, heatmap of F1.
    Proves performance is stable across a broad region (not cherry-picked).

  Part C: Temperature τ Deep-Dive  (core differentiating contribution)
    - τ vs prior accuracy curve (dual y-axis with F1)  — synthetic data
    - τ trajectory during training (convergence analysis)
    - τ on real electricity data (interpretability)
    - Per-edge Omega weight visualization

  Part D: Convergence Analysis
    h(W₀) trajectory and loss curves across ALM outer iterations.

Baselines (where applicable):
  DYNOTEARS, PCMCI+, VARLiNGAM, PRCD-MAP variants

Usage:
  python exp5_scalability_sensitivity.py                    # default
  python exp5_scalability_sensitivity.py --quick            # tiny test
  python exp5_scalability_sensitivity.py --full             # full sweep
  python exp5_scalability_sensitivity.py --part A           # scalability only
  python exp5_scalability_sensitivity.py --part B           # hyperparam only
  python exp5_scalability_sensitivity.py --part C           # tau deep-dive
  python exp5_scalability_sensitivity.py --part D           # convergence
  python exp5_scalability_sensitivity.py --part A B         # combine parts
  python exp5_scalability_sensitivity.py --dims 10 20 50 100
  python exp5_scalability_sensitivity.py --seeds 0 1 2 3 4
=============================================================================
"""

import os, sys, time, warnings, argparse, traceback, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# ---- core model ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_prcd_map import PRCD_MAP_Model, train_prcd_alm

# ---- optional baselines (graceful fallback) ----
HAS_TIGRAMITE = False
HAS_LINGAM = False

try:
    import tigramite
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite import data_processing as pp
    HAS_TIGRAMITE = True
except ImportError:
    warnings.warn("tigramite not installed — PCMCI+ baseline skipped.")

try:
    import lingam
    HAS_LINGAM = True
except ImportError:
    warnings.warn("lingam not installed — VARLiNGAM baseline skipped.")

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ====================================================================
# 1. Utilities
# ====================================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_lag_tensors(X: np.ndarray, K: int):
    X_t = torch.tensor(X[K:], dtype=torch.float32)
    X_lags = [torch.tensor(X[K - k: -k], dtype=torch.float32)
              for k in range(1, K + 1)]
    return X_t, X_lags


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


ZH_TO_EN = {
    '大工业电量': 'Large Ind.', '非普工业': 'Non-Std Ind.',
    '居民生活': 'Residential', '商业用电': 'Commercial',
    '农业': 'Agriculture', '黑色金属冶炼': 'Ferrous Metal',
    '非金属矿物制品业(水泥)': 'Nonmetallic Min.',
    '非金属矿物制品业': 'Nonmetallic Min.',
    '化学制品制造业': 'Chemicals', '纺织业': 'Textiles',
    '有色金属冶炼': 'Non-ferrous', '通信设备制造业': 'Comm. Equip.',
    '金属制品业': 'Fab. Metal', '橡胶和塑料': 'Rubber/Plastics',
    '机械': 'Machinery', '电子': 'Electronics', '石化': 'Petrochem.',
    '食品': 'Food', '建筑业': 'Construction',
    '交通仓储和邮政业': 'Logistics', '信息传输': 'IT/Telecom',
    '金融业': 'Finance', '房地产业': 'Real Estate',
    '制造业': 'Manufacturing', '皮革行业': 'Leather',
    '化学纤维制造业': 'Chemical Fiber', '农林牧渔业': 'Agri./Fore./Fish.',
    '住宿和餐饮业': 'Hotels/Catering', '租赁和商务服务': 'Leasing/Business',
    '抽水蓄能': 'Pumped Storage', '非居照明': 'Non-Res. Lighting',
    '工业': 'Industry', '公共服务及管理组织': 'Public Admin.',
    '造纸和纸制品业': 'Paper Products', '批发和零售业': 'Wholesale/Retail',
    '批发和零售业务': 'Wholesale/Retail', '服装': 'Apparel',
    '服饰业': 'Apparel',
}


# ====================================================================
# 2. Synthetic Data Generation
# ====================================================================
def make_er_dag(d: int, edge_prob: float = 0.15,
                w_range=(0.3, 0.8), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                W[i, j] = rng.uniform(*w_range) * rng.choice([-1, 1])
    perm = rng.permutation(d)
    W = W[perm][:, perm]
    return W


def make_lag_matrices(d: int, K: int, edge_prob: float = 0.10,
                      scale: float = 0.25, seed: int = 0) -> list:
    rng = np.random.default_rng(seed + 1000)
    Wk_list = []
    for _ in range(K):
        Wk = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if rng.random() < edge_prob:
                    Wk[i, j] = rng.uniform(0.05, scale) * rng.choice([-1, 1])
        Wk_list.append(Wk)
    return Wk_list


def simulate_svar_linear(T: int, W0: np.ndarray, Wk_list: list,
                         noise_type: str = "laplace",
                         noise_scale: float = 1.0,
                         seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = W0.shape[0]
    K = len(Wk_list)
    A_inv = np.linalg.inv(np.eye(d) - W0)
    T_total = T + K + 50
    X = np.zeros((T_total, d))
    for t in range(K, T_total):
        if noise_type == "gaussian":
            eps = rng.normal(0, noise_scale, size=d)
        elif noise_type == "laplace":
            eps = rng.laplace(0, noise_scale, size=d)
        elif noise_type == "student_t":
            eps = rng.standard_t(df=5, size=d) * noise_scale
        else:
            eps = rng.normal(0, noise_scale, size=d)
        lag_sum = sum(X[t - k - 1] @ Wk_list[k] for k in range(K))
        X[t] = (lag_sum + eps) @ A_inv
    X = X[K + 50:]
    if not np.all(np.isfinite(X)):
        warnings.warn("simulate_svar_linear: non-finite values")
        return np.zeros((T, d))
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - X.mean(0)) / std


# ====================================================================
# 3. Prior Generation
# ====================================================================
def gen_prior(W0_true: np.ndarray, Wk_true: list,
              acc: float, mode: str = "random",
              seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = W0_true.shape[0]
    B_all = (np.abs(W0_true) > 1e-10).astype(int)
    for Wk in Wk_true:
        B_all = np.maximum(B_all, (np.abs(Wk) > 1e-10).astype(int))
    P = np.full((d, d), 0.5)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            true_edge = B_all[i, j] == 1
            agree = rng.random() < acc
            if agree:
                P[i, j] = rng.uniform(0.75, 0.95) if true_edge else rng.uniform(0.05, 0.25)
            else:
                P[i, j] = rng.uniform(0.05, 0.25) if true_edge else rng.uniform(0.75, 0.95)
    return P


# ====================================================================
# 4. Evaluation Metrics
# ====================================================================
def compute_graph_metrics(W0_true: np.ndarray,
                          W0_est_continuous: np.ndarray) -> dict:
    d = W0_true.shape[0]
    B_true = (np.abs(W0_true) > 1e-10).astype(int)
    np.fill_diagonal(B_true, 0)
    scores = np.abs(W0_est_continuous).copy()
    np.fill_diagonal(scores, 0.0)
    mask = ~np.eye(d, dtype=bool)
    y_true = B_true[mask]
    y_score = scores[mask]
    res = {}
    n_pos, n_neg = int(y_true.sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        res["auroc"] = 0.5
        res["auprc"] = float(n_pos) / (n_pos + n_neg + 1e-12)
    else:
        try:
            res["auroc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            res["auroc"] = 0.5
        try:
            res["auprc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            res["auprc"] = float(n_pos) / (n_pos + n_neg + 1e-12)
    # Best-F1
    if y_score.max() > 0:
        thresholds = np.linspace(0, float(y_score.max()), 200)
    else:
        thresholds = [0.0]
    best_f1, best_thr, best_p, best_r = 0.0, 0.0, 0.0, 0.0
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        if f1 > best_f1:
            best_f1, best_thr, best_p, best_r = f1, thr, p, r
    res["f1_opt"] = best_f1
    res["prec_opt"] = best_p
    res["rec_opt"] = best_r
    res["best_thr"] = best_thr
    # SHD
    B_opt = (scores >= best_thr).astype(int)
    np.fill_diagonal(B_opt, 0)
    res["shd"] = int((B_true[mask] != B_opt[mask]).sum())
    res["shd_norm"] = res["shd"] / (d * (d - 1))
    return res


def combine_W0_Wk(W0: np.ndarray, Wk_list: list) -> np.ndarray:
    abs_mats = [np.abs(W0)] + [np.abs(wk) for wk in Wk_list]
    return np.stack(abs_mats, axis=0).max(axis=0)


# ====================================================================
# 5. Baseline Implementations
# ====================================================================
# ---- 5a. DYNOTEARS ----
class _DYNOTEARS(nn.Module):
    def __init__(self, d: int, K: int, lam: float = 0.01):
        super().__init__()
        self.d, self.K, self.lam = d, K, lam
        s = 1e-2
        self.W0 = nn.Parameter(s * torch.randn(d, d))
        self.Wk = nn.ParameterList(
            [nn.Parameter(s * torch.randn(d, d)) for _ in range(K)])
        self.register_buffer("mask", 1.0 - torch.eye(d))

    def _adj(self):
        return self.W0 * self.mask

    def _h(self):
        A = torch.clamp(self._adj(), -3.0, 3.0)
        return torch.trace(torch.matrix_exp(A * A)) - self.d

    def loss(self, X_t, X_lags, rho, alpha):
        A = self._adj()
        pred = X_t @ A
        for k in range(self.K):
            pred = pred + X_lags[k] @ self.Wk[k]
        mse = 0.5 * torch.sum((X_t - pred) ** 2) / X_t.shape[0]
        l1 = self.lam * (torch.norm(A, p=1)
                         + sum(torch.norm(w, p=1) for w in self.Wk))
        h = self._h()
        return mse + l1 + alpha * h + 0.5 * rho * h ** 2, h


def run_dynotears(X, d, K, lam=0.01, max_outer=30, inner=200,
                  lr=1e-2, seed=0):
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    m = _DYNOTEARS(d, K, lam)
    opt = optim.Adam(m.parameters(), lr=lr)
    rho, alpha = 1.0, 0.0
    for _ in range(max_outer):
        for __ in range(inner):
            loss, h = m.loss(X_t, X_lags, rho, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        h_val = float(m._h().detach())
        if abs(h_val) < 1e-6:
            break
        alpha += rho * h_val
        rho *= 2.0
    with torch.no_grad():
        W0 = m._adj().cpu().numpy()
        Wk = [w.detach().cpu().numpy() for w in m.Wk]
    return W0, Wk


# ---- 5b. PCMCI+ ----
def run_pcmci_plus(X, d, K, alpha_level=0.05, seed=0):
    if not HAS_TIGRAMITE:
        return None, None
    try:
        df = pp.DataFrame(X, var_names=[f"V{i}" for i in range(d)])
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=df, cond_ind_test=parcorr, verbosity=0)
        res = pcmci.run_pcmciplus(tau_min=0, tau_max=K, pc_alpha=alpha_level)
        val = res["val_matrix"]
        graph = res["graph"]
        W0 = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                g = str(graph[i, j, 0])
                if "-->" in g:
                    W0[i, j] = abs(val[i, j, 0])
                elif "o-o" in g or "x-x" in g:
                    W0[i, j] = abs(val[i, j, 0]) * 0.5
        Wk = []
        for k in range(1, K + 1):
            Ak = np.abs(val[:, :, k])
            np.fill_diagonal(Ak, 0.0)
            Wk.append(Ak)
        return W0, Wk
    except Exception as e:
        warnings.warn(f"PCMCI+ failed: {e}")
        return None, None


# ---- 5c. VARLiNGAM ----
def run_varlingam(X, d, K, seed=0):
    if not HAS_LINGAM:
        return None, None
    try:
        model = lingam.VARLiNGAM(lags=K, random_state=seed)
        model.fit(X)
        B0 = model.adjacency_matrices_[0]
        W0 = B0.T
        np.fill_diagonal(W0, 0.0)
        Wk = [model.adjacency_matrices_[k].T for k in range(1, K + 1)]
        return W0, Wk
    except Exception as e:
        warnings.warn(f"VARLiNGAM failed: {e}")
        return None, None


# ---- 5d. PRCD-MAP wrapper ----
def run_prcd_map(X, P_prior, d, K, lambda1=0.01, lambda2=0.01,
                 learn_tau=True, tau0=1.0,
                 max_iter=30, inner_iter=500, lr=1e-2, seed=0):
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau, tau0=tau0,
        tau_min=0.1, tau_max=10.0,
    )
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
        verbose=False, postprocess=False,
    )
    return W0, Wk, tau


# ====================================================================
# 6. PRCD-MAP with full training trajectory logging
# ====================================================================
def train_prcd_alm_with_logging(
    model: PRCD_MAP_Model,
    X_t: torch.Tensor,
    X_lags,
    max_iter: int = 30,
    inner_iter: int = 500,
    lr: float = 1e-2,
    rho_0: float = 1.0,
    gamma: float = 2.0,
    tol: float = 1e-6,
    grad_clip: float = 5.0,
    tau_ema: float = 0.1,
    tau_warmup: int = 1,
) -> Tuple[np.ndarray, list, float, pd.DataFrame]:
    """
    Same as train_prcd_alm but logs per-outer-iteration metrics:
      outer_iter, rho, alpha, h_val, tau, agreement, loss_alm, loss_mse, loss_l1, loss_prior
    Returns (W0, Wk, tau, log_df).
    """
    rho = float(rho_0)
    alpha = 0.0
    log_rows = []

    for it in range(max_iter):
        # Fresh optimizer each outer iteration to avoid stale momentum
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=inner_iter, eta_min=lr * 0.01)
        # --- Inner loop: fix tau, optimize W0 and Wk ---
        for inner in range(inner_iter):
            loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau = \
                model.compute_losses(X_t, X_lags, rho, alpha)
            optimizer.zero_grad(set_to_none=True)
            loss_alm.backward()
            if grad_clip > 0:
                effective_clip = float(grad_clip) * max(1.0, math.log1p(rho))
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=effective_clip)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            h_now = float(model._compute_h_w0().detach().cpu().item())
            tau_now = float(model.get_tau().detach().cpu().item())
            l_alm = float(loss_alm.detach().cpu().item())
            l_mse = float(loss_mse.detach().cpu().item())
            l_l1 = float(loss_l1.detach().cpu().item())
            l_prior_val = float(loss_prior.detach().cpu().item()) \
                if isinstance(loss_prior, torch.Tensor) else float(loss_prior)

        # --- Agreement-based tau update (outer loop, with warmup) ---
        agreement = None
        if model.learn_tau and it >= tau_warmup:
            agreement = model.compute_agreement()
            # Exponential schedule: high agreement => tau near tau_min; low => near tau_max
            tau_target = model.tau_min * (model.tau_max / model.tau_min) ** (1.0 - agreement)
            # Adaptive EMA: high agreement => fast descent (low smoothing)
            effective_ema = tau_ema * (1.0 - agreement)
            tau_new = effective_ema * tau_now + (1.0 - effective_ema) * tau_target
            model.set_tau(tau_new)
            tau_now = tau_new

        log_rows.append(dict(
            outer_iter=it + 1,
            rho=rho,
            alpha=alpha,
            h_val=h_now,
            tau=tau_now,
            agreement=agreement,
            loss_alm=l_alm,
            loss_mse=l_mse,
            loss_l1=l_l1,
            loss_prior=l_prior_val,
        ))

        if abs(h_now) <= tol:
            break
        alpha = alpha + rho * h_now
        rho = rho * float(gamma)

    with torch.no_grad():
        W0 = model.get_W0_adj().detach().cpu().numpy()
        Wk = [wk.detach().cpu().numpy() for wk in model.Wk]
        tau_est = float(model.get_tau().detach().cpu().item())

    return W0, Wk, tau_est, pd.DataFrame(log_rows)


# ====================================================================
# 7. Configuration
# ====================================================================
@dataclass
class Cfg:
    # Part A: Scalability
    scale_dims:       List[int]   = field(default_factory=lambda: [10, 20, 50, 100, 200])
    scale_T:          int         = 500
    scale_K:          int         = 1
    scale_prior_acc:  float       = 0.6

    # Part B: Hyperparameter Sensitivity
    hp_d:             int         = 20
    hp_T:             int         = 500
    hp_K:             int         = 1
    hp_prior_acc:     float       = 0.6
    hp_lam1_grid:     List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 0.5])
    hp_lam2_grid:     List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 0.5])

    # Part C: Temperature τ Deep-Dive
    tau_d:            int         = 20
    tau_T:            int         = 1000
    tau_K:            int         = 1
    tau_prior_accs:   List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tau_noises:       List[str]   = field(default_factory=lambda: [
        "gaussian", "laplace"])

    # Part D: Convergence
    conv_d:           int         = 20
    conv_T:           int         = 500
    conv_K:           int         = 1
    conv_prior_accs:  List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])

    # Real data paths
    electricity_xlsx:  str = r"E:\electricity\0227test.xlsx"
    electricity_prior: str = r"E:\electricity\Auto_Generated_Prior.csv"

    # General
    seeds:            List[int]   = field(default_factory=lambda: list(range(5)))
    edge_prob:        float       = 0.15
    lambda1:          float       = 0.01
    lambda2:          float       = 0.01
    max_iter:         int         = 30
    inner_iter:       int         = 500
    lr:               float       = 1e-2

    # Baselines
    do_dynotears:     bool = True
    do_pcmci:         bool = True
    do_varlingam:     bool = True

    # Which parts to run
    parts:            List[str]   = field(default_factory=lambda: [
        "A", "B", "C", "D"])

    # Output
    output_dir:       str = "exp5_results"


def cfg_quick():
    return Cfg(
        scale_dims=[10, 20, 50],
        scale_T=300,
        hp_lam1_grid=[0.01, 0.1],
        hp_lam2_grid=[0.01, 0.1],
        tau_prior_accs=[0.0, 0.3, 0.6, 0.9],
        tau_noises=["laplace"],
        conv_prior_accs=[0.3, 0.7],
        seeds=list(range(3)),
        max_iter=15, inner_iter=100,
        do_pcmci=False, do_varlingam=False,
    )


def cfg_full():
    return Cfg(
        scale_dims=[10, 20, 50, 100, 200],
        scale_T=500,
        hp_lam1_grid=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        hp_lam2_grid=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        tau_prior_accs=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                        0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                        0.8, 0.85, 0.9, 0.95, 1.0],
        tau_noises=["gaussian", "laplace", "student_t"],
        conv_prior_accs=[0.1, 0.3, 0.5, 0.7, 0.9],
        seeds=list(range(10)),
    )


# ====================================================================
# 8. Figure Utilities
# ====================================================================
COLORS = {
    "PRCD-MAP(learn_tau)": "#E74C3C",
    "PRCD-MAP(fixed_tau)": "#E67E22",
    "PRCD-MAP(uniform)":   "#9B59B6",
    "DYNOTEARS":           "#2C3E50",
    "PCMCI+":              "#27AE60",
    "VARLiNGAM":           "#3498DB",
}
MARKERS = {
    "PRCD-MAP(learn_tau)": "o",
    "PRCD-MAP(fixed_tau)": "s",
    "PRCD-MAP(uniform)":   "^",
    "DYNOTEARS":           "D",
    "PCMCI+":              "v",
    "VARLiNGAM":           "P",
}


def _save(prefix):
    plt.savefig(prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(prefix + ".pdf", bbox_inches="tight")
    plt.close()
    print(f">>> {prefix}.png / .pdf")


# ====================================================================
# Part A: Scalability Timing
# ====================================================================
def run_part_A(cfg: Cfg) -> pd.DataFrame:
    """
    Wall-clock runtime vs dimension d for each method.
    All methods use the same synthetic data at each (d, seed).
    """
    print("\n" + "=" * 64)
    print(" Part A: Scalability — Runtime vs Dimension")
    print("=" * 64)

    T = cfg.scale_T
    K = cfg.scale_K
    acc = cfg.scale_prior_acc
    rows = []

    for d in cfg.scale_dims:
        for seed in cfg.seeds:
            # Generate data once
            W0_true = make_er_dag(d, cfg.edge_prob, seed=seed)
            Wk_true = make_lag_matrices(d, K, seed=seed)
            X = simulate_svar_linear(T, W0_true, Wk_true, "laplace", seed=seed)
            if not np.all(np.isfinite(X)):
                continue
            X_std = standardize(X)
            P_prior = gen_prior(W0_true, Wk_true, acc, "random", seed=seed + 999)

            base = dict(d=d, T=T, K=K, seed=seed)
            n_params_svar = d * d + K * d * d  # approximate parameter count

            # --- PRCD-MAP (learn tau) ---
            try:
                t0 = time.time()
                run_prcd_map(X_std, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                             learn_tau=True, seed=seed,
                             max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                             lr=cfg.lr)
                rows.append({**base, "method": "PRCD-MAP(learn_tau)",
                             "time": time.time() - t0,
                             "n_params": n_params_svar + 1})
            except Exception:
                pass

            # --- PRCD-MAP (fixed tau) ---
            try:
                t0 = time.time()
                run_prcd_map(X_std, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                             learn_tau=False, tau0=1.0, seed=seed,
                             max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                             lr=cfg.lr)
                rows.append({**base, "method": "PRCD-MAP(fixed_tau)",
                             "time": time.time() - t0,
                             "n_params": n_params_svar})
            except Exception:
                pass

            # --- DYNOTEARS ---
            if cfg.do_dynotears:
                try:
                    t0 = time.time()
                    run_dynotears(X_std, d, K, cfg.lambda1, cfg.max_iter,
                                  cfg.inner_iter, cfg.lr, seed)
                    rows.append({**base, "method": "DYNOTEARS",
                                 "time": time.time() - t0,
                                 "n_params": n_params_svar})
                except Exception:
                    pass

            # --- PCMCI+ (skip for d > 80, too slow) ---
            if cfg.do_pcmci and HAS_TIGRAMITE and d <= 80:
                try:
                    t0 = time.time()
                    run_pcmci_plus(X_std, d, K, seed=seed)
                    rows.append({**base, "method": "PCMCI+",
                                 "time": time.time() - t0,
                                 "n_params": 0})
                except Exception:
                    pass

            # --- VARLiNGAM ---
            if cfg.do_varlingam and HAS_LINGAM:
                try:
                    t0 = time.time()
                    run_varlingam(X_std, d, K, seed=seed)
                    rows.append({**base, "method": "VARLiNGAM",
                                 "time": time.time() - t0,
                                 "n_params": 0})
                except Exception:
                    pass

            print(f"  d={d}, seed={seed} done")

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No scalability results.")
        return df

    # Save CSV
    p = os.path.join(cfg.output_dir, "partA_scalability.csv")
    df.to_csv(p, index=False)
    print(f">>> {p}")

    # --- Summary table ---
    agg = df.groupby(["method", "d"]).agg(
        time_mean=("time", "mean"),
        time_std=("time", "std"),
        n=("seed", "count"),
    ).reset_index()
    p2 = os.path.join(cfg.output_dir, "partA_scalability_summary.csv")
    agg.to_csv(p2, index=False)
    print(f">>> {p2}")

    print("\n--- Scalability Summary (mean time in seconds) ---")
    piv = agg.pivot(index="method", columns="d", values="time_mean")
    print(piv.round(2).to_string())

    # --- Figure 1: Runtime vs d (linear scale) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, log_scale, title_suffix in [
        (axes[0], False, "(Linear Scale)"),
        (axes[1], True,  "(Log-Log Scale)"),
    ]:
        for m in sorted(df["method"].unique()):
            sub = df[df["method"] == m]
            agg_m = sub.groupby("d").agg(
                t_m=("time", "mean"), t_s=("time", "std")).reset_index()
            ax.errorbar(agg_m["d"], agg_m["t_m"], yerr=agg_m["t_s"],
                        label=m, marker=MARKERS.get(m, "x"),
                        color=COLORS.get(m, "grey"), lw=2, capsize=3,
                        markersize=7)
        ax.set_xlabel("Number of Variables (d)", fontsize=12)
        ax.set_ylabel("Wall-clock Time (seconds)", fontsize=12)
        ax.set_title(f"Scalability: Runtime vs Dimension\n{title_suffix}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, ls=":", alpha=0.5)
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

    plt.tight_layout()
    _save(os.path.join(cfg.output_dir, "figA1_scalability"))

    # --- Figure 2: Relative speedup vs d ---
    methods = sorted(df["method"].unique())
    ref_method = "PRCD-MAP(learn_tau)"
    if ref_method in methods and len(methods) > 1:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ref_times = df[df["method"] == ref_method].groupby("d")["time"].mean()
        for m in methods:
            if m == ref_method:
                continue
            m_times = df[df["method"] == m].groupby("d")["time"].mean()
            common_dims = sorted(set(ref_times.index) & set(m_times.index))
            if not common_dims:
                continue
            ratio = [m_times[dd] / ref_times[dd] for dd in common_dims]
            ax.plot(common_dims, ratio, marker=MARKERS.get(m, "x"),
                    color=COLORS.get(m, "grey"), lw=2, markersize=7,
                    label=f"{m} / PRCD-MAP")
        ax.axhline(y=1.0, color="black", ls="--", lw=1.0, alpha=0.6)
        ax.set_xlabel("Number of Variables (d)", fontsize=12)
        ax.set_ylabel("Time Ratio (method / PRCD-MAP)", fontsize=12)
        ax.set_title("Relative Runtime Comparison", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
        plt.tight_layout()
        _save(os.path.join(cfg.output_dir, "figA2_relative_speed"))

    return df


# ====================================================================
# Part B: Hyperparameter Sensitivity
# ====================================================================
def run_part_B(cfg: Cfg) -> pd.DataFrame:
    """
    λ₁ × λ₂ grid on a representative synthetic setting.
    Reports F1, AUROC for each (λ₁, λ₂) combination.
    """
    print("\n" + "=" * 64)
    print(" Part B: Hyperparameter Sensitivity (λ₁ × λ₂)")
    print("=" * 64)

    d = cfg.hp_d
    T = cfg.hp_T
    K = cfg.hp_K
    acc = cfg.hp_prior_acc
    rows = []

    n_total = (len(cfg.hp_lam1_grid) * len(cfg.hp_lam2_grid)
               * len(cfg.seeds))
    print(f"  d={d}, T={T}, prior_acc={acc}")
    print(f"  λ₁ grid: {cfg.hp_lam1_grid}")
    print(f"  λ₂ grid: {cfg.hp_lam2_grid}")
    print(f"  Total: {n_total} runs")

    count = 0
    for seed in cfg.seeds:
        W0_true = make_er_dag(d, cfg.edge_prob, seed=seed)
        Wk_true = make_lag_matrices(d, K, seed=seed)
        X = simulate_svar_linear(T, W0_true, Wk_true, "laplace", seed=seed)
        if not np.all(np.isfinite(X)):
            continue
        X_std = standardize(X)
        P_prior = gen_prior(W0_true, Wk_true, acc, "random", seed=seed + 999)

        # Combined ground truth for evaluation
        B_true_comb = (np.abs(W0_true) > 1e-10).astype(int)
        for Wk_t in Wk_true:
            B_true_comb = np.maximum(
                B_true_comb, (np.abs(Wk_t) > 1e-10).astype(int))

        for l1 in cfg.hp_lam1_grid:
            for l2 in cfg.hp_lam2_grid:
                count += 1
                try:
                    W0_est, Wk_est, tau = run_prcd_map(
                        X_std, P_prior, d, K, l1, l2,
                        learn_tau=True, seed=seed,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr)
                    # Evaluate W0 only
                    met_w0 = compute_graph_metrics(W0_true, W0_est)
                    # Evaluate combined
                    W_comb = combine_W0_Wk(W0_est, Wk_est)
                    met_comb = compute_graph_metrics(
                        B_true_comb.astype(float) * 0.5, W_comb)
                    rows.append(dict(
                        lambda1=l1, lambda2=l2, seed=seed, tau=tau,
                        w0_auroc=met_w0["auroc"], w0_f1=met_w0["f1_opt"],
                        w0_shd_norm=met_w0["shd_norm"],
                        comb_auroc=met_comb["auroc"], comb_f1=met_comb["f1_opt"],
                        comb_shd_norm=met_comb["shd_norm"],
                    ))
                except Exception as e:
                    warnings.warn(f"HP l1={l1} l2={l2} seed={seed}: {e}")

                if count % max(1, n_total // 5) == 0:
                    print(f"  [{count}/{n_total}]")

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No HP sensitivity results.")
        return df

    p = os.path.join(cfg.output_dir, "partB_hyperparam.csv")
    df.to_csv(p, index=False)
    print(f">>> {p}")

    # --- Summary ---
    agg = df.groupby(["lambda1", "lambda2"]).agg(
        comb_f1_mean=("comb_f1", "mean"),
        comb_f1_std=("comb_f1", "std"),
        comb_auroc_mean=("comb_auroc", "mean"),
        comb_auroc_std=("comb_auroc", "std"),
        tau_mean=("tau", "mean"),
    ).reset_index()
    p2 = os.path.join(cfg.output_dir, "partB_hyperparam_summary.csv")
    agg.to_csv(p2, index=False)
    print(f">>> {p2}")

    # --- Figure 1: F1 Heatmap ---
    for metric, label in [("comb_f1", "F1"), ("comb_auroc", "AUROC")]:
        col_mean = f"{metric}_mean"
        piv = agg.pivot(index="lambda1", columns="lambda2", values=col_mean)

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
        ax.set_title(f"Hyperparameter Sensitivity: {label}\n"
                     f"(d={d}, T={T}, prior_acc={acc})",
                     fontsize=14, fontweight="bold")

        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.values[i, j]
                txt_color = "white" if val < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=txt_color, fontweight="bold")

        plt.colorbar(im, ax=ax, label=label)
        plt.tight_layout()
        _save(os.path.join(cfg.output_dir,
                           f"figB1_heatmap_{metric}"))

    # --- Figure 2: τ Heatmap ---
    piv_tau = agg.pivot(index="lambda1", columns="lambda2", values="tau_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv_tau.values, cmap="coolwarm", aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(piv_tau.columns)))
    ax.set_xticklabels([f"{v:.3f}" for v in piv_tau.columns], fontsize=9)
    ax.set_yticks(range(len(piv_tau.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in piv_tau.index], fontsize=9)
    ax.set_xlabel("$\\lambda_2$ (prior weight)", fontsize=12)
    ax.set_ylabel("$\\lambda_1$ (sparsity)", fontsize=12)
    ax.set_title(f"Learned Temperature $\\tau$ vs ($\\lambda_1$, $\\lambda_2$)",
                 fontsize=14, fontweight="bold")
    for i in range(len(piv_tau.index)):
        for j in range(len(piv_tau.columns)):
            ax.text(j, i, f"{piv_tau.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="$\\tau$")
    plt.tight_layout()
    _save(os.path.join(cfg.output_dir, "figB2_heatmap_tau"))

    # --- Figure 3: 1D slices (F1 vs λ₁ for different λ₂) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # λ₁ sweep (fix λ₂ at a few values)
    ax = axes[0]
    lam2_show = sorted(cfg.hp_lam2_grid)[:4]
    cmap_vals = plt.cm.viridis(np.linspace(0.2, 0.9, len(lam2_show)))
    for idx, l2_val in enumerate(lam2_show):
        sub = agg[np.isclose(agg["lambda2"], l2_val)]
        if sub.empty:
            continue
        sub = sub.sort_values("lambda1")
        ax.plot(sub["lambda1"], sub["comb_f1_mean"],
                marker="o", lw=2, markersize=6, color=cmap_vals[idx],
                label=f"$\\lambda_2$={l2_val:.3f}")
        ax.fill_between(sub["lambda1"],
                        sub["comb_f1_mean"] - sub["comb_f1_std"],
                        sub["comb_f1_mean"] + sub["comb_f1_std"],
                        alpha=0.15, color=cmap_vals[idx])
    ax.set_xlabel("$\\lambda_1$ (sparsity)", fontsize=12)
    ax.set_ylabel("F1 (combined)", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("F1 vs $\\lambda_1$ (varying $\\lambda_2$)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)

    # λ₂ sweep (fix λ₁ at a few values)
    ax = axes[1]
    lam1_show = sorted(cfg.hp_lam1_grid)[:4]
    cmap_vals2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(lam1_show)))
    for idx, l1_val in enumerate(lam1_show):
        sub = agg[np.isclose(agg["lambda1"], l1_val)]
        if sub.empty:
            continue
        sub = sub.sort_values("lambda2")
        ax.plot(sub["lambda2"], sub["comb_f1_mean"],
                marker="s", lw=2, markersize=6, color=cmap_vals2[idx],
                label=f"$\\lambda_1$={l1_val:.3f}")
        ax.fill_between(sub["lambda2"],
                        sub["comb_f1_mean"] - sub["comb_f1_std"],
                        sub["comb_f1_mean"] + sub["comb_f1_std"],
                        alpha=0.15, color=cmap_vals2[idx])
    ax.set_xlabel("$\\lambda_2$ (prior weight)", fontsize=12)
    ax.set_ylabel("F1 (combined)", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("F1 vs $\\lambda_2$ (varying $\\lambda_1$)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle("Hyperparameter Sensitivity: 1D Slices",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(cfg.output_dir, "figB3_hp_slices"))

    # Console summary
    print("\n--- Hyperparameter Sensitivity (mean F1) ---")
    print(piv.round(3).to_string())
    best_idx = agg["comb_f1_mean"].idxmax()
    best = agg.loc[best_idx]
    print(f"\n  Best: λ₁={best['lambda1']:.3f}, λ₂={best['lambda2']:.3f}"
          f" -> F1={best['comb_f1_mean']:.4f}")

    return df


# ====================================================================
# Part C: Temperature τ Deep-Dive
# ====================================================================
def run_part_C(cfg: Cfg) -> pd.DataFrame:
    """
    The core differentiating analysis:
    1. τ vs prior accuracy (dual y-axis with F1)  — on synthetic data
    2. τ across noise types
    3. τ on real electricity data
    4. Per-edge Omega weight visualization
    """
    print("\n" + "=" * 64)
    print(" Part C: Temperature τ Deep-Dive")
    print("=" * 64)

    d = cfg.tau_d
    T = cfg.tau_T
    K = cfg.tau_K
    rows = []

    # ====== C.1: τ vs prior accuracy (synthetic) ======
    print("\n  --- C.1: τ vs Prior Accuracy (Synthetic) ---")
    for noise in cfg.tau_noises:
        for acc in cfg.tau_prior_accs:
            for seed in cfg.seeds:
                W0_true = make_er_dag(d, cfg.edge_prob, seed=seed)
                Wk_true = make_lag_matrices(d, K, seed=seed)
                X = simulate_svar_linear(T, W0_true, Wk_true, noise, seed=seed)
                if not np.all(np.isfinite(X)):
                    continue
                X_std = standardize(X)
                P_prior = gen_prior(W0_true, Wk_true, acc, "random",
                                    seed=seed + 999)

                B_true_comb = (np.abs(W0_true) > 1e-10).astype(int)
                for Wk_t in Wk_true:
                    B_true_comb = np.maximum(
                        B_true_comb, (np.abs(Wk_t) > 1e-10).astype(int))

                # PRCD-MAP (learn tau)
                try:
                    W0_est, Wk_est, tau = run_prcd_map(
                        X_std, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                        learn_tau=True, seed=seed,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr)
                    W_comb = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_graph_metrics(
                        B_true_comb.astype(float) * 0.5, W_comb)
                    rows.append(dict(
                        experiment="synth_tau", noise=noise,
                        prior_acc=acc, seed=seed, d=d, T=T,
                        method="learn_tau", tau=tau,
                        **met))
                except Exception:
                    pass

                # PRCD-MAP (fixed tau=1)
                try:
                    W0_est, Wk_est, tau = run_prcd_map(
                        X_std, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                        learn_tau=False, tau0=1.0, seed=seed,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr)
                    W_comb = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_graph_metrics(
                        B_true_comb.astype(float) * 0.5, W_comb)
                    rows.append(dict(
                        experiment="synth_tau", noise=noise,
                        prior_acc=acc, seed=seed, d=d, T=T,
                        method="fixed_tau", tau=tau,
                        **met))
                except Exception:
                    pass

                # PRCD-MAP (uniform prior — no prior info baseline)
                try:
                    P_unif = np.full((d, d), 0.5)
                    W0_est, Wk_est, tau = run_prcd_map(
                        X_std, P_unif, d, K, cfg.lambda1, cfg.lambda2,
                        learn_tau=True, seed=seed,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr)
                    W_comb = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_graph_metrics(
                        B_true_comb.astype(float) * 0.5, W_comb)
                    rows.append(dict(
                        experiment="synth_tau", noise=noise,
                        prior_acc=acc, seed=seed, d=d, T=T,
                        method="uniform", tau=tau,
                        **met))
                except Exception:
                    pass

            print(f"    noise={noise}, acc={acc:.1f} done")

    df_synth = pd.DataFrame(rows)

    # ====== C.2: τ on Real Electricity Data ======
    print("\n  --- C.2: τ on Real Electricity Data ---")
    real_rows = []
    try:
        df_ts = pd.read_excel(cfg.electricity_xlsx, index_col=0)
        df_diff = df_ts.diff(periods=7).dropna()
        col_names = df_diff.columns.tolist()
        d_real = len(col_names)
        X_real = standardize(df_diff.values)

        P_prior_real = np.full((d_real, d_real), 0.5)
        if os.path.exists(cfg.electricity_prior):
            P_prior_real = pd.read_csv(cfg.electricity_prior, index_col=0).values
            if P_prior_real.shape != (d_real, d_real):
                P_prior_real = np.full((d_real, d_real), 0.5)

        print(f"    d={d_real}, T={len(X_real)}")

        for seed in cfg.seeds:
            # learn tau
            try:
                W0, Wk, tau = run_prcd_map(
                    X_real, P_prior_real, d_real, K,
                    lambda1=0.001, lambda2=0.02,
                    learn_tau=True, seed=seed,
                    max_iter=35, inner_iter=250, lr=cfg.lr)
                real_rows.append(dict(
                    experiment="real_tau", seed=seed,
                    method="learn_tau", tau=tau,
                    d=d_real, T=len(X_real),
                ))
            except Exception as e:
                warnings.warn(f"Real data seed={seed}: {e}")

            # fixed tau
            try:
                W0, Wk, tau = run_prcd_map(
                    X_real, P_prior_real, d_real, K,
                    lambda1=0.001, lambda2=0.02,
                    learn_tau=False, tau0=1.0, seed=seed,
                    max_iter=35, inner_iter=250, lr=cfg.lr)
                real_rows.append(dict(
                    experiment="real_tau", seed=seed,
                    method="fixed_tau", tau=tau,
                    d=d_real, T=len(X_real),
                ))
            except Exception:
                pass

            print(f"    seed={seed} done")

        # Omega visualization for one seed
        if real_rows:
            set_seed(0)
            X_t, X_lags = make_lag_tensors(X_real, K)
            model = PRCD_MAP_Model(
                num_vars=d_real, lag_k=K, P_prior=P_prior_real,
                lambda1=0.001, lambda2=0.02,
                learn_tau=True, tau0=1.0, tau_min=0.1, tau_max=10.0,
            )
            W0, Wk, tau = train_prcd_alm(
                model, X_t, X_lags,
                max_iter=35, inner_iter=250, lr=cfg.lr,
                rho_0=1.0, gamma=2.0, tol=1e-6,
                verbose=False, postprocess=False,
            )
            with torch.no_grad():
                tau_t = model.get_tau()
                Omega = model.omega_mask(tau_t).cpu().numpy()
                P_cal = model.calibrated_prior(tau_t).cpu().numpy()

            # Save Omega and calibrated prior
            names_en = [ZH_TO_EN.get(c, c) for c in col_names]
            _plot_omega_heatmap(Omega, P_cal, P_prior_real, names_en,
                                float(tau), cfg.output_dir)

    except Exception as e:
        warnings.warn(f"Failed to load electricity data: {e}")

    df_real = pd.DataFrame(real_rows)

    # ====== Combine and save ======
    df_all = pd.concat([df_synth, df_real], ignore_index=True)
    if not df_all.empty:
        p = os.path.join(cfg.output_dir, "partC_tau_deepdive.csv")
        df_all.to_csv(p, index=False)
        print(f"\n>>> {p} ({len(df_all)} rows)")

    # ====== Generate τ figures ======
    _plot_tau_figures(df_synth, df_real, cfg)

    return df_all


def _plot_tau_figures(df_synth: pd.DataFrame, df_real: pd.DataFrame,
                     cfg: Cfg):
    """Generate all temperature τ figures."""
    out = cfg.output_dir

    if df_synth.empty:
        return

    # === Figure C1: THE KEY FIGURE — τ vs prior accuracy (dual y-axis) ===
    # Filter to main noise type (laplace) for clarity
    primary_noise = "laplace" if "laplace" in df_synth["noise"].values else \
        df_synth["noise"].iloc[0]
    sub_learn = df_synth[(df_synth["method"] == "learn_tau") &
                         (df_synth["noise"] == primary_noise)]

    if not sub_learn.empty and sub_learn["prior_acc"].nunique() >= 3:
        agg = sub_learn.groupby("prior_acc").agg(
            tau_m=("tau", "mean"), tau_s=("tau", "std"),
            f1_m=("f1_opt", "mean"), f1_s=("f1_opt", "std"),
        ).reset_index()

        fig, ax1 = plt.subplots(figsize=(9, 6))
        c_tau, c_f1 = "#E74C3C", "#2C3E50"

        # Left y-axis: learned τ
        ax1.errorbar(agg["prior_acc"], agg["tau_m"], yerr=agg["tau_s"],
                     color=c_tau, marker="o", lw=2.8, ms=9, capsize=5,
                     label="Learned $\\tau$", zorder=3)
        ax1.set_xlabel("Prior Accuracy", fontsize=14)
        ax1.set_ylabel("Learned Temperature $\\tau$", fontsize=14,
                       color=c_tau)
        ax1.tick_params(axis="y", labelcolor=c_tau)
        ax1.set_xlim(-0.05, 1.05)

        # Right y-axis: F1 for learn_tau, fixed_tau, and uniform
        ax2 = ax1.twinx()

        # learn_tau F1
        ax2.errorbar(agg["prior_acc"], agg["f1_m"], yerr=agg["f1_s"],
                     color=c_f1, marker="s", lw=2.5, ms=8, capsize=4,
                     ls="--", label="F1 (learn $\\tau$)", zorder=2)

        # fixed_tau F1
        sub_fixed = df_synth[(df_synth["method"] == "fixed_tau") &
                             (df_synth["noise"] == primary_noise)]
        if not sub_fixed.empty:
            agg_fixed = sub_fixed.groupby("prior_acc").agg(
                f1_m=("f1_opt", "mean"), f1_s=("f1_opt", "std"),
            ).reset_index()
            ax2.errorbar(agg_fixed["prior_acc"], agg_fixed["f1_m"],
                         yerr=agg_fixed["f1_s"],
                         color="#E67E22", marker="^", lw=2.0, ms=7,
                         capsize=3, ls="-.",
                         label="F1 (fixed $\\tau$=1)", zorder=1)

        # uniform F1 as horizontal line
        sub_unif = df_synth[(df_synth["method"] == "uniform") &
                            (df_synth["noise"] == primary_noise)]
        if not sub_unif.empty:
            f1_unif = sub_unif["f1_opt"].mean()
            ax2.axhline(y=f1_unif, color="#9B59B6", ls=":", lw=2.0,
                        alpha=0.8, label=f"F1 (no prior) = {f1_unif:.3f}")

        ax2.set_ylabel("F1 (optimal threshold)", fontsize=14, color=c_f1)
        ax2.tick_params(axis="y", labelcolor=c_f1)

        # Combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=10,
                   framealpha=0.9)

        ax1.set_title(
            "Temperature $\\tau$ Adaptation vs Prior Quality\n"
            "($\\tau$ increases when prior is unreliable, "
            "gracefully recovering data-driven performance)",
            fontsize=13, fontweight="bold")
        ax1.grid(True, ls=":", alpha=0.4)

        # Annotation arrows
        if len(agg) > 5:
            # Annotate low-acc region
            low_idx = agg["prior_acc"].idxmin()
            ax1.annotate("$\\tau$ large:\nprior discounted",
                         xy=(agg.loc[low_idx, "prior_acc"],
                             agg.loc[low_idx, "tau_m"]),
                         xytext=(0.25, agg["tau_m"].max() * 0.85),
                         fontsize=9, ha="center",
                         arrowprops=dict(arrowstyle="->", color=c_tau,
                                         lw=1.5),
                         color=c_tau)
            # Annotate high-acc region
            high_idx = agg["prior_acc"].idxmax()
            ax1.annotate("$\\tau$ small:\nprior trusted",
                         xy=(agg.loc[high_idx, "prior_acc"],
                             agg.loc[high_idx, "tau_m"]),
                         xytext=(0.75, agg["tau_m"].max() * 0.7),
                         fontsize=9, ha="center",
                         arrowprops=dict(arrowstyle="->", color=c_tau,
                                         lw=1.5),
                         color=c_tau)

        plt.tight_layout()
        _save(os.path.join(out, "figC1_tau_vs_prior_KEY"))

    # === Figure C2: τ across noise types ===
    if df_synth["noise"].nunique() > 1:
        sub_lt = df_synth[df_synth["method"] == "learn_tau"]
        if not sub_lt.empty:
            noises = sorted(sub_lt["noise"].unique())
            noise_colors = {"gaussian": "#3498DB", "laplace": "#E74C3C",
                            "student_t": "#27AE60",
                            "heteroscedastic": "#9B59B6"}

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            for metric, ax, yl in [("tau", axes[0], "Learned $\\tau$"),
                                    ("f1_opt", axes[1], "F1")]:
                for noise in noises:
                    sub = sub_lt[sub_lt["noise"] == noise]
                    agg_n = sub.groupby("prior_acc").agg(
                        y=(metric, "mean"), e=(metric, "std")).reset_index()
                    ax.errorbar(agg_n["prior_acc"], agg_n["y"],
                                yerr=agg_n["e"],
                                label=noise,
                                color=noise_colors.get(noise, "grey"),
                                marker="o", lw=2, capsize=3, ms=6)
                ax.set_xlabel("Prior Accuracy", fontsize=12)
                ax.set_ylabel(yl, fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, ls=":", alpha=0.5)
                ax.set_xlim(-0.05, 1.05)

            fig.suptitle("$\\tau$ Adaptation Across Noise Types",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            _save(os.path.join(out, "figC2_tau_noise_types"))

    # === Figure C3: τ on real data (box plot) ===
    if not df_real.empty:
        sub_real_lt = df_real[df_real["method"] == "learn_tau"]
        if not sub_real_lt.empty:
            tau_vals = sub_real_lt["tau"].values
            tau_mean = np.mean(tau_vals)
            tau_std = np.std(tau_vals)

            fig, ax = plt.subplots(figsize=(6, 5))
            bp = ax.boxplot([tau_vals], labels=["Electricity"],
                            widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor="#E74C3C", alpha=0.6),
                            medianprops=dict(color="black", lw=2))
            ax.scatter(np.ones(len(tau_vals)), tau_vals,
                       color="#E74C3C", s=60, alpha=0.7, zorder=3,
                       edgecolors="white", linewidth=0.8)
            ax.set_ylabel("Learned $\\tau$", fontsize=13)
            ax.set_title(
                f"Temperature $\\tau$ on Electricity Dataset\n"
                f"$\\tau$ = {tau_mean:.3f} $\\pm$ {tau_std:.3f}",
                fontsize=13, fontweight="bold")

            # Interpretation text
            if tau_mean < 2.0:
                interp = "Model places HIGH trust in domain prior"
            elif tau_mean < 5.0:
                interp = "Model places MODERATE trust in domain prior"
            else:
                interp = "Model largely IGNORES domain prior"
            ax.text(0.5, 0.02, interp, transform=ax.transAxes,
                    ha="center", fontsize=11, style="italic",
                    color="#2C3E50",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                              ec="#E67E22", alpha=0.9))

            ax.grid(True, axis="y", ls=":", alpha=0.4)
            plt.tight_layout()
            _save(os.path.join(out, "figC3_tau_real_data"))

    # === Summary table ===
    if not df_synth.empty:
        print("\n--- τ Summary (synthetic, learn_tau, laplace) ---")
        sub_lt = df_synth[(df_synth["method"] == "learn_tau") &
                          (df_synth["noise"] == primary_noise)]
        if not sub_lt.empty:
            agg_s = sub_lt.groupby("prior_acc").agg(
                tau_mean=("tau", "mean"), tau_std=("tau", "std"),
                f1_mean=("f1_opt", "mean"), f1_std=("f1_opt", "std"),
            ).reset_index()
            print(agg_s.round(4).to_string(index=False))

    if not df_real.empty:
        sub_rl = df_real[df_real["method"] == "learn_tau"]
        if not sub_rl.empty:
            print(f"\n--- τ on Electricity: "
                  f"{sub_rl['tau'].mean():.4f} +/- {sub_rl['tau'].std():.4f} ---")


def _plot_omega_heatmap(Omega: np.ndarray, P_cal: np.ndarray,
                        P_prior_raw: np.ndarray, names_en: list,
                        tau: float, out: str):
    """
    Visualize the Omega mask (prior-derived penalty weights)
    and calibrated prior P_hat(τ) on the real electricity data.
    """
    d = Omega.shape[0]
    # Use a subset if d is large
    if d > 20:
        idx = list(range(20))
        Omega = Omega[np.ix_(idx, idx)]
        P_cal = P_cal[np.ix_(idx, idx)]
        P_prior_raw = P_prior_raw[np.ix_(idx, idx)]
        names_en = [names_en[i] for i in idx]
        d = 20

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, mat, title, cmap_name in [
        (axes[0], P_prior_raw, "Raw Prior $P$", "RdYlBu_r"),
        (axes[1], P_cal, f"Calibrated Prior $\\hat{{P}}(\\tau={tau:.2f})$",
         "RdYlBu_r"),
        (axes[2], Omega, f"$\\Omega(\\tau)$ (penalty weight)",
         "YlOrRd"),
    ]:
        im = ax.imshow(mat, cmap=cmap_name, aspect="auto")
        ax.set_xticks(range(d))
        ax.set_xticklabels(names_en, rotation=90, fontsize=7)
        ax.set_yticks(range(d))
        ax.set_yticklabels(names_en, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Prior Calibration Pipeline: Raw Prior $\\to$ "
        "Calibrated Prior $\\to$ Penalty Weights\n"
        f"(learned $\\tau$ = {tau:.3f})",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "figC4_omega_visualization"))


# ====================================================================
# Part D: Convergence Analysis
# ====================================================================
def run_part_D(cfg: Cfg) -> pd.DataFrame:
    """
    Track ALM convergence: h(W₀) trajectory, loss decomposition,
    and τ evolution across outer iterations.
    """
    print("\n" + "=" * 64)
    print(" Part D: Convergence Analysis")
    print("=" * 64)

    d = cfg.conv_d
    T = cfg.conv_T
    K = cfg.conv_K
    all_logs = []

    for acc in cfg.conv_prior_accs:
        for seed in cfg.seeds[:3]:  # fewer seeds for convergence
            W0_true = make_er_dag(d, cfg.edge_prob, seed=seed)
            Wk_true = make_lag_matrices(d, K, seed=seed)
            X = simulate_svar_linear(T, W0_true, Wk_true, "laplace",
                                     seed=seed)
            if not np.all(np.isfinite(X)):
                continue
            X_std = standardize(X)
            P_prior = gen_prior(W0_true, Wk_true, acc, "random",
                                seed=seed + 999)

            # PRCD-MAP with logging
            set_seed(seed)
            X_t, X_lags = make_lag_tensors(X_std, K)
            model = PRCD_MAP_Model(
                num_vars=d, lag_k=K, P_prior=P_prior,
                lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                learn_tau=True, tau0=1.0,
                tau_min=0.1, tau_max=10.0,
            )

            try:
                W0, Wk, tau, log_df = train_prcd_alm_with_logging(
                    model, X_t, X_lags,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, rho_0=1.0, gamma=2.0, tol=1e-6,
                )
                log_df["prior_acc"] = acc
                log_df["seed"] = seed
                log_df["method"] = "PRCD-MAP(learn_tau)"
                all_logs.append(log_df)
            except Exception as e:
                warnings.warn(f"Convergence logging failed acc={acc} "
                              f"seed={seed}: {e}")

            # DYNOTEARS convergence (for comparison)
            if cfg.do_dynotears:
                set_seed(seed)
                X_t2, X_lags2 = make_lag_tensors(X_std, K)
                m_dyno = _DYNOTEARS(d, K, cfg.lambda1)
                opt_d = optim.Adam(m_dyno.parameters(), lr=cfg.lr)
                rho_d, alpha_d = 1.0, 0.0
                dyno_rows = []
                for it in range(cfg.max_iter):
                    for _ in range(cfg.inner_iter):
                        loss_d, h_d = m_dyno.loss(X_t2, X_lags2, rho_d,
                                                   alpha_d)
                        opt_d.zero_grad(set_to_none=True)
                        loss_d.backward()
                        nn.utils.clip_grad_norm_(m_dyno.parameters(), 5.0)
                        opt_d.step()
                    h_now = float(m_dyno._h().detach())
                    dyno_rows.append(dict(
                        outer_iter=it + 1, rho=rho_d, alpha=alpha_d,
                        h_val=h_now, tau=float("nan"),
                        loss_alm=float(loss_d.detach()),
                        loss_mse=float("nan"), loss_l1=float("nan"),
                        loss_prior=float("nan"),
                    ))
                    if abs(h_now) < 1e-6:
                        break
                    alpha_d += rho_d * h_now
                    rho_d *= 2.0
                df_dyno = pd.DataFrame(dyno_rows)
                df_dyno["prior_acc"] = acc
                df_dyno["seed"] = seed
                df_dyno["method"] = "DYNOTEARS"
                all_logs.append(df_dyno)

        print(f"  acc={acc} done")

    if not all_logs:
        print("  No convergence logs.")
        return pd.DataFrame()

    df_conv = pd.concat(all_logs, ignore_index=True)
    p = os.path.join(cfg.output_dir, "partD_convergence.csv")
    df_conv.to_csv(p, index=False)
    print(f">>> {p}")

    # Generate convergence figures
    _plot_convergence_figures(df_conv, cfg)

    return df_conv


def _plot_convergence_figures(df: pd.DataFrame, cfg: Cfg):
    out = cfg.output_dir

    # === Figure D1: h(W₀) trajectory across outer iterations ===
    prcd_df = df[df["method"] == "PRCD-MAP(learn_tau)"]
    accs = sorted(prcd_df["prior_acc"].unique())
    acc_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(accs)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: h(W0) vs outer iter
    ax = axes[0][0]
    for idx, acc in enumerate(accs):
        sub = prcd_df[prcd_df["prior_acc"] == acc]
        agg = sub.groupby("outer_iter")["h_val"].agg(["mean", "std"]).reset_index()
        ax.semilogy(agg["outer_iter"], np.abs(agg["mean"]) + 1e-12,
                    marker="o", lw=2, ms=5, color=acc_colors[idx],
                    label=f"acc={acc:.1f}")
    ax.axhline(y=1e-6, color="red", ls="--", lw=1.0, alpha=0.7,
               label="tol=1e-6")
    ax.set_xlabel("Outer ALM Iteration", fontsize=11)
    ax.set_ylabel("|h(W₀)| (log scale)", fontsize=11)
    ax.set_title("DAG Constraint Convergence", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 2: τ trajectory
    ax = axes[0][1]
    for idx, acc in enumerate(accs):
        sub = prcd_df[prcd_df["prior_acc"] == acc]
        agg = sub.groupby("outer_iter")["tau"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["outer_iter"], agg["mean"], yerr=agg["std"],
                    marker="s", lw=2, ms=5, color=acc_colors[idx],
                    capsize=2, label=f"acc={acc:.1f}")
    ax.set_xlabel("Outer ALM Iteration", fontsize=11)
    ax.set_ylabel("Temperature $\\tau$", fontsize=11)
    ax.set_title("$\\tau$ Evolution During Training", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 3: Total ALM loss
    ax = axes[1][0]
    for idx, acc in enumerate(accs):
        sub = prcd_df[prcd_df["prior_acc"] == acc]
        agg = sub.groupby("outer_iter")["loss_alm"].agg(["mean"]).reset_index()
        ax.plot(agg["outer_iter"], agg["mean"],
                marker="D", lw=2, ms=4, color=acc_colors[idx],
                label=f"acc={acc:.1f}")
    # Also DYNOTEARS if available
    dyno_df = df[df["method"] == "DYNOTEARS"]
    if not dyno_df.empty:
        agg_d = dyno_df.groupby("outer_iter")["loss_alm"].mean().reset_index()
        ax.plot(agg_d["outer_iter"], agg_d["loss_alm"],
                marker="x", lw=2, ms=5, color="#2C3E50", ls="--",
                label="DYNOTEARS")
    ax.set_xlabel("Outer ALM Iteration", fontsize=11)
    ax.set_ylabel("Total ALM Loss", fontsize=11)
    ax.set_title("Loss Convergence", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel 4: Loss decomposition for one representative setting
    if not accs:
        axes[1][1].set_visible(False)
    else:
        ax = axes[1][1]
        mid_acc = accs[len(accs) // 2]
        sub_mid = prcd_df[(prcd_df["prior_acc"] == mid_acc)]
        # Average across seeds
        loss_cols = ["loss_mse", "loss_l1", "loss_prior"]
        avail_cols = [c for c in loss_cols if c in sub_mid.columns]
        if avail_cols:
            agg_mid = sub_mid.groupby("outer_iter")[avail_cols].mean().reset_index()
            comp_colors = {"loss_mse": "#3498DB", "loss_l1": "#E67E22",
                           "loss_prior": "#E74C3C"}
            comp_labels = {"loss_mse": "MSE", "loss_l1": "L1 sparsity",
                           "loss_prior": "Prior penalty"}
            for col in avail_cols:
                ax.plot(agg_mid["outer_iter"], agg_mid[col],
                        marker="o", lw=2, ms=4,
                        color=comp_colors.get(col, "grey"),
                        label=comp_labels.get(col, col))
            ax.set_xlabel("Outer ALM Iteration", fontsize=11)
            ax.set_ylabel("Loss Component", fontsize=11)
            ax.set_title(f"Loss Decomposition (prior_acc={mid_acc:.1f})",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, ls=":", alpha=0.5)
        else:
            ax.set_visible(False)

    fig.suptitle("Convergence Analysis: PRCD-MAP Training Dynamics",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "figD1_convergence"))

    # === Figure D2: PRCD-MAP vs DYNOTEARS h convergence comparison ===
    if not dyno_df.empty and not prcd_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))

        # PRCD-MAP (pick middle accuracy)
        mid_acc = accs[len(accs) // 2] if accs else 0.5
        sub_p = prcd_df[prcd_df["prior_acc"] == mid_acc]
        if not sub_p.empty:
            agg_p = sub_p.groupby("outer_iter")["h_val"].agg(
                ["mean", "std"]).reset_index()
            ax.semilogy(agg_p["outer_iter"],
                        np.abs(agg_p["mean"]) + 1e-12,
                        marker="o", lw=2.5, ms=7, color="#E74C3C",
                        label=f"PRCD-MAP (acc={mid_acc:.1f})")

        # DYNOTEARS
        agg_d = dyno_df.groupby("outer_iter")["h_val"].agg(
            ["mean", "std"]).reset_index()
        ax.semilogy(agg_d["outer_iter"],
                    np.abs(agg_d["mean"]) + 1e-12,
                    marker="D", lw=2.5, ms=7, color="#2C3E50",
                    label="DYNOTEARS")

        ax.axhline(y=1e-6, color="red", ls="--", lw=1.0, alpha=0.7,
                   label="Convergence tol")
        ax.set_xlabel("Outer ALM Iteration", fontsize=12)
        ax.set_ylabel("|h(W₀)| (log scale)", fontsize=12)
        ax.set_title("DAG Constraint Convergence: PRCD-MAP vs DYNOTEARS",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, ls=":", alpha=0.5)
        plt.tight_layout()
        _save(os.path.join(out, "figD2_convergence_comparison"))


# ====================================================================
# Master Runner
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: Scalability & Hyperparameter Sensitivity")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny test run")
    parser.add_argument("--full", action="store_true",
                        help="Full NeurIPS sweep")
    parser.add_argument("--part", nargs="+", type=str, default=None,
                        choices=["A", "B", "C", "D"],
                        help="Specific part(s) to run")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="Dimensions for scalability (Part A)")
    parser.add_argument("--no-pcmci", action="store_true")
    parser.add_argument("--no-varlingam", action="store_true")
    parser.add_argument("--no-dynotears", action="store_true")
    args = parser.parse_args()

    # Select config
    if args.quick:
        cfg = cfg_quick()
    elif args.full:
        cfg = cfg_full()
    else:
        cfg = Cfg()

    # CLI overrides
    if args.output:       cfg.output_dir    = args.output
    if args.seeds:        cfg.seeds         = args.seeds
    if args.dims:         cfg.scale_dims    = args.dims
    if args.part:         cfg.parts         = args.part
    if args.no_pcmci:     cfg.do_pcmci      = False
    if args.no_varlingam: cfg.do_varlingam  = False
    if args.no_dynotears: cfg.do_dynotears  = False

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 68)
    print(" Experiment 5: Scalability & Hyperparameter Sensitivity")
    print("=" * 68)
    print(f"  parts        = {cfg.parts}")
    print(f"  seeds        = {cfg.seeds}")
    if "A" in cfg.parts:
        print(f"  scale dims   = {cfg.scale_dims}")
    if "B" in cfg.parts:
        print(f"  HP λ₁ grid   = {cfg.hp_lam1_grid}")
        print(f"  HP λ₂ grid   = {cfg.hp_lam2_grid}")
    if "C" in cfg.parts:
        print(f"  τ prior accs = {cfg.tau_prior_accs}")
        print(f"  τ noises     = {cfg.tau_noises}")
    if "D" in cfg.parts:
        print(f"  conv accs    = {cfg.conv_prior_accs}")
    print(f"  baselines    = DYNOTEARS={cfg.do_dynotears}, "
          f"PCMCI+={cfg.do_pcmci and HAS_TIGRAMITE}, "
          f"VARLiNGAM={cfg.do_varlingam and HAS_LINGAM}")
    print(f"  output       = {cfg.output_dir}")
    print("=" * 68)

    t_global = time.time()
    results = {}

    # ============================================
    # Part A: Scalability
    # ============================================
    if "A" in cfg.parts:
        results["A"] = run_part_A(cfg)

    # ============================================
    # Part B: Hyperparameter Sensitivity
    # ============================================
    if "B" in cfg.parts:
        results["B"] = run_part_B(cfg)

    # ============================================
    # Part C: Temperature τ Deep-Dive
    # ============================================
    if "C" in cfg.parts:
        results["C"] = run_part_C(cfg)

    # ============================================
    # Part D: Convergence Analysis
    # ============================================
    if "D" in cfg.parts:
        results["D"] = run_part_D(cfg)

    # ============================================
    # Final Summary
    # ============================================
    elapsed = time.time() - t_global
    print("\n" + "=" * 68)
    print(f" Experiment 5 complete in {elapsed:.1f}s")
    print("=" * 68)

    for part_name, df in results.items():
        if df is not None and not df.empty:
            print(f"  Part {part_name}: {len(df)} rows")
        else:
            print(f"  Part {part_name}: no results")

    print(f"\n>>> All results saved to: {cfg.output_dir}/")

    # List generated files
    if os.path.isdir(cfg.output_dir):
        files = sorted(os.listdir(cfg.output_dir))
        print(f"\n  Generated files ({len(files)}):")
        for f in files:
            size = os.path.getsize(os.path.join(cfg.output_dir, f))
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"    {f:50s} {size_str}")


if __name__ == "__main__":
    main()
