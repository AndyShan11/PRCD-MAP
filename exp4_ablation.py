"""
=============================================================================
Experiment 4 — Extended Ablation Study for PRCD-MAP
=============================================================================
NeurIPS-grade ablation isolating each component's contribution:

  Variant (A): No prior            — λ₂ = 0  (pure data-driven SVAR)
  Variant (B): Prior, fixed τ=1    — standard prior, no temperature learning
  Variant (C): Prior, learned τ    — FULL MODEL
  Variant (D): No L1 sparsity      — λ₁ = 0  (prior regularization only)
  Variant (E): Prior on lags only  — prior NOT applied to instantaneous W₀
  Variant (F): Hard prior mask     — binarize P_prior → 0/1 mask, force zeros

  Key insight for (F): proves soft probabilistic prior > hard masking strategy
  even with identical domain knowledge.

Evaluation:
  - Synthetic data  → graph quality metrics (AUROC, AUPRC, F1, SHD)
  - Real data       → downstream RMSE via LSTM forecasting
  - Both            → learned τ analysis

Usage:
  python exp4_ablation.py                        # default
  python exp4_ablation.py --quick                # tiny test run
  python exp4_ablation.py --full                 # full NeurIPS sweep
  python exp4_ablation.py --sub synthetic        # synthetic only
  python exp4_ablation.py --sub real             # real data only
  python exp4_ablation.py --sub hard_mask        # focus on soft vs hard
  python exp4_ablation.py --seeds 0 1 2 3 4
=============================================================================
"""
import os, sys, time, warnings, argparse, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             mean_squared_error, mean_absolute_error)
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# ---- core model ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_prcd_map import PRCD_MAP_Model, train_prcd_alm

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
    """Erdos-Renyi DAG: upper-triangular then random permutation."""
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
    cut = d // 2
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            true_edge = B_all[i, j] == 1
            if mode == "random":
                agree = rng.random() < acc
            elif mode == "systematic":
                agree = False if (i < cut and j >= cut) else (rng.random() < acc)
            elif mode == "adversarial":
                agree = rng.random() < acc
            else:
                agree = rng.random() < acc
            if agree:
                P[i, j] = rng.uniform(0.75, 0.95) if true_edge else rng.uniform(0.05, 0.25)
            else:
                P[i, j] = rng.uniform(0.05, 0.25) if true_edge else rng.uniform(0.75, 0.95)
    return P


def binarize_prior_to_mask(P_prior: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert soft prior P ∈ [0,1]^{d×d} to hard binary mask.
    Entries with P >= threshold are kept (mask=1), others forced to zero.
    This is the "standard" hard masking approach used in many existing methods.
    """
    return (P_prior >= threshold).astype(float)


# ====================================================================
# 4. Evaluation Metrics
# ====================================================================
def compute_graph_metrics(W0_true: np.ndarray,
                          W0_est_continuous: np.ndarray) -> dict:
    """
    Returns: auroc, auprc, f1_opt, prec_opt, rec_opt,
             shd, shd_norm, f1_topk, best_thr
    """
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

    # Best-F1 over threshold sweep
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

    # SHD at best threshold
    B_opt = (scores >= best_thr).astype(int)
    np.fill_diagonal(B_opt, 0)
    res["shd"] = int((B_true[mask] != B_opt[mask]).sum())
    res["shd_norm"] = res["shd"] / (d * (d - 1))

    # Top-k
    k_true = int(B_true.sum())
    res["k_true"] = k_true
    if k_true > 0 and y_score.max() > 0:
        k = min(k_true, len(y_score))
        topk_idx = np.argpartition(y_score, -k)[-k:]
        pred_topk = np.zeros_like(y_true)
        pred_topk[topk_idx] = 1
        tp = int(((y_true == 1) & (pred_topk == 1)).sum())
        fp = int(((y_true == 0) & (pred_topk == 1)).sum())
        fn = int(((y_true == 1) & (pred_topk == 0)).sum())
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        res["f1_topk"] = 2 * p * r / (p + r + 1e-12)
    else:
        res["f1_topk"] = 0.0
    return res


def combine_W0_Wk(W0: np.ndarray, Wk_list: list) -> np.ndarray:
    abs_mats = [np.abs(W0)] + [np.abs(wk) for wk in Wk_list]
    combined = np.stack(abs_mats, axis=0).max(axis=0)
    return combined


# ====================================================================
# 5. Hard-Mask PRCD-MAP Model
# ====================================================================
class PRCD_MAP_HardMask(nn.Module):
    """
    PRCD-MAP variant with HARD prior mask instead of soft probabilistic prior.
    Entries where mask=0 have their weights forced to zero (no gradient).
    This is the standard approach in literature: use domain knowledge
    to zero out impossible edges, but no soft calibration.

    Convention: mask[i,j]=1 means edge i->j is allowed by domain knowledge.
    Uses DAGMA acyclicity constraint: h(W) = -log det(sI - W⊙W) + d·log(s).
    """
    def __init__(self, num_vars: int, lag_k: int,
                 hard_mask: np.ndarray,
                 lambda1: float = 0.01,
                 dagma_s: float = 1.0):
        super().__init__()
        self.d = int(num_vars)
        self.K = int(lag_k)
        self.lambda1 = float(lambda1)
        self.dagma_s = float(dagma_s)

        init_scale = 1e-2
        self.W0 = nn.Parameter(init_scale * torch.randn(self.d, self.d))
        self.Wk = nn.ParameterList(
            [nn.Parameter(init_scale * torch.randn(self.d, self.d))
             for _ in range(self.K)]
        )
        # Hard mask: 1 = allowed, 0 = forced zero
        mask_tensor = torch.tensor(hard_mask, dtype=torch.float32)
        self.register_buffer("hard_mask", mask_tensor)
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.d))

    def get_W0_adj(self) -> torch.Tensor:
        return self.W0 * self.off_diag_mask * self.hard_mask

    def get_Wk_masked(self, k: int) -> torch.Tensor:
        return self.Wk[k] * self.hard_mask

    def _compute_h_w0(self) -> torch.Tensor:
        """DAGMA: h(W) = -log det(sI - W⊙W) + d·log(s)
        When sign <= 0, M is not positive definite (W has cycles),
        fall back to Frobenius norm as smooth surrogate.
        """
        W0_adj = self.get_W0_adj()
        s = self.dagma_s
        M = s * torch.eye(self.d, device=W0_adj.device) - W0_adj * W0_adj
        sign, logabsdet = torch.linalg.slogdet(M)
        if sign.item() <= 0:
            return torch.sum(W0_adj * W0_adj)
        h = -logabsdet + self.d * math.log(s)
        return torch.clamp(h, min=0.0)

    def forward(self, X_t: torch.Tensor, X_lags) -> torch.Tensor:
        W0_adj = self.get_W0_adj()
        pred = X_t @ W0_adj
        for k in range(self.K):
            pred = pred + X_lags[k] @ self.get_Wk_masked(k)
        return pred

    def compute_losses(self, X_t, X_lags, rho, alpha):
        T = X_t.shape[0]
        pred = self.forward(X_t, X_lags)
        loss_mse = 0.5 * torch.sum((X_t - pred) ** 2) / T
        W0_adj = self.get_W0_adj()
        loss_l1 = self.lambda1 * (
            torch.norm(W0_adj, p=1)
            + sum(torch.norm(self.get_Wk_masked(k), p=1) for k in range(self.K))
        )
        h_val = self._compute_h_w0()
        loss_alm = loss_mse + loss_l1 + alpha * h_val + 0.5 * rho * h_val ** 2
        return loss_alm, loss_mse, loss_l1, h_val


def train_hard_mask_alm(model: PRCD_MAP_HardMask,
                        X_t, X_lags,
                        max_iter=30, inner_iter=500,
                        lr=1e-2, rho_0=1.0, gamma=2.0,
                        tol=1e-6, grad_clip=5.0,
                        postprocess=False, thr_ratio=0.10):
    """Train hard-mask model with ALM + CosineAnnealingLR."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rho = float(rho_0)
    alpha = 0.0
    for it in range(max_iter):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=inner_iter, eta_min=lr * 0.01)
        for _ in range(inner_iter):
            loss_alm, loss_mse, loss_l1, h_val = model.compute_losses(
                X_t, X_lags, rho, alpha)
            optimizer.zero_grad(set_to_none=True)
            loss_alm.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
        # Reset LR for next outer iteration (warm restart)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        with torch.no_grad():
            h_now = float(model._compute_h_w0().detach().cpu().item())
        if abs(h_now) <= tol:
            break
        alpha = alpha + rho * h_now
        rho = rho * float(gamma)
    with torch.no_grad():
        W0 = model.get_W0_adj().detach().cpu().numpy()
        Wk = [model.get_Wk_masked(k).detach().cpu().numpy()
              for k in range(model.K)]
        if postprocess:
            max_weight = float(np.max(np.abs(W0))) if W0.size else 0.0
            thr = float(thr_ratio) * max_weight
            W0[np.abs(W0) < thr] = 0.0
            Wk = [np.where(np.abs(wk) < thr, 0.0, wk) for wk in Wk]
    return W0, Wk


# ====================================================================
# 6. Ablation Variant Runners
# ====================================================================
def run_variant_A(X, d, K, seed, max_iter, inner_iter, lr, lambda1):
    """
    Variant (A): No prior — λ₂ = 0.
    Pure data-driven SVAR with L1 sparsity + DAG constraint only.
    """
    set_seed(seed)
    P_unif = np.full((d, d), 0.5)
    X_t, X_lags = make_lag_tensors(X, K)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_unif,
        lambda1=lambda1, lambda2=0.0,  # key: no prior regularization
        learn_tau=False, tau0=1.0,
        tau_min=0.1, tau_max=10.0,
    )
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
        verbose=False, postprocess=False,
    )
    return W0, Wk, float(tau)


def run_variant_B(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """
    Variant (B): Prior with fixed τ = 1.
    Standard prior regularization, no temperature adaptation.
    """
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=False, tau0=1.0,  # key: fixed tau=1
        tau_min=0.1, tau_max=10.0,
    )
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
        verbose=False, postprocess=False,
    )
    return W0, Wk, float(tau)


def run_variant_C(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """
    Variant (C): FULL MODEL — prior with learned τ.
    """
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=True, tau0=1.0,  # key: learnable tau
        tau_min=0.1, tau_max=10.0,
    )
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
        verbose=False, postprocess=False,
    )
    return W0, Wk, float(tau)


def run_variant_D(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda2):
    """
    Variant (D): No L1 sparsity — λ₁ = 0.
    Only prior-based regularization + DAG constraint.
    """
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=0.0,  # key: no L1 sparsity
        lambda2=lambda2,
        learn_tau=True, tau0=1.0,
        tau_min=0.1, tau_max=10.0,
    )
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
        verbose=False, postprocess=False,
    )
    return W0, Wk, float(tau)


def run_variant_E(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """
    Variant (E): Prior on lag matrices only — NOT applied to W₀.
    Tests whether prior info on instantaneous vs lagged edges matters.
    """
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=True, tau0=1.0,
        tau_min=0.1, tau_max=10.0,
        apply_prior_to_w0=False,  # key: no prior on W0
    )
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
        verbose=False, postprocess=False,
    )
    return W0, Wk, float(tau)


def run_variant_F(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, mask_threshold=0.5):
    """
    Variant (F): Hard prior mask.
    Binarize P_prior at threshold → force masked edges to zero.
    Only L1 + DAG constraint on allowed edges.
    This is the "standard" hard-masking approach from existing literature.
    """
    set_seed(seed)
    hard_mask = binarize_prior_to_mask(P_prior, threshold=mask_threshold)
    X_t, X_lags = make_lag_tensors(X, K)
    model = PRCD_MAP_HardMask(
        num_vars=d, lag_k=K, hard_mask=hard_mask,
        lambda1=lambda1,
    )
    W0, Wk = train_hard_mask_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
    )
    return W0, Wk, float("nan")  # no tau concept


# ====================================================================
# 7. Lorenz-96 for Synthetic Benchmark
# ====================================================================
def _lorenz96_rhs(t, x, F):
    d = len(x)
    dxdt = np.zeros(d)
    for i in range(d):
        dxdt[i] = (x[(i + 1) % d] - x[(i - 2) % d]) * x[(i - 1) % d] - x[i] + F
    return dxdt


def lorenz96_ground_truth(d: int) -> np.ndarray:
    B = np.zeros((d, d), dtype=int)
    for i in range(d):
        B[(i - 2) % d, i] = 1
        B[(i - 1) % d, i] = 1
        B[(i + 1) % d, i] = 1
    return B


def generate_lorenz96(d: int = 10, T: int = 2000, F: float = 10.0,
                      dt: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = F * np.ones(d) + rng.normal(0, 0.01, d)
    burn_in = 500
    total = burn_in + T
    t_span = (0, total * dt)
    t_eval = np.arange(0, total * dt, dt)
    sol = solve_ivp(_lorenz96_rhs, t_span, x0, args=(F,),
                    t_eval=t_eval, method="RK45", max_step=dt)
    if sol.status != 0:
        return np.zeros((T, d)), lorenz96_ground_truth(d)
    X_full = sol.y.T[burn_in:][:T]
    if len(X_full) < T:
        pad = np.zeros((T - len(X_full), d))
        X_full = np.vstack([X_full, pad])
    return standardize(X_full), lorenz96_ground_truth(d)


# ====================================================================
# 8. Downstream Forecasting (for real-data ablation)
# ====================================================================
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def select_top_parents(W_combined: np.ndarray, target_idx: int,
                       top_m: int, col_names: list) -> list:
    d = W_combined.shape[0]
    w_in = np.abs(W_combined[:, target_idx]).copy()
    w_in[target_idx] = 0.0
    n_parents = max(0, top_m - 1)
    n_parents = min(n_parents, d - 1)
    if n_parents > 0:
        parent_idx = np.argsort(-w_in)[:n_parents]
    else:
        parent_idx = []
    feats = [col_names[i] for i in parent_idx] + [col_names[target_idx]]
    seen = set()
    unique = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def run_forecast(df_diff, split, feature_cols, target_col,
                 seq_len=7, max_epochs=50, lr=3e-3, hidden=32,
                 seed=0, patience=8, batch_size=64):
    """Run LSTM forecast, return (rmse, mae)."""
    set_seed(seed)
    X_raw = df_diff[feature_cols].values
    y_raw = df_diff[[target_col]].values
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
    val_split = int(len(X_train_raw) * 0.8)
    X_tr_raw, X_val_raw = X_train_raw[:val_split], X_train_raw[val_split:]
    y_tr_raw, y_val_raw = y_train_raw[:val_split], y_train_raw[val_split:]
    sx = StandardScaler()
    sy = StandardScaler()
    X_tr = sx.fit_transform(X_tr_raw)
    X_val = sx.transform(X_val_raw)
    X_test = sx.transform(X_test_raw)
    y_tr = sy.fit_transform(y_tr_raw)
    y_val = sy.transform(y_val_raw)
    y_test = sy.transform(y_test_raw)
    Xtr, ytr = make_sequences(X_tr, y_tr, seq_len)
    Xva, yva = make_sequences(X_val, y_val, seq_len)
    Xte, yte = make_sequences(X_test, y_test, seq_len)
    if len(Xtr) < 10 or len(Xte) < 5:
        return float("nan"), float("nan")
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    Xva = torch.tensor(Xva, dtype=torch.float32)
    yva = torch.tensor(yva, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.float32)
    model = LSTMForecaster(input_dim=Xtr.shape[2], hidden_dim=hidden)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                        shuffle=True)
    opt_ = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(max_epochs):
        model.train()
        for bx, by in loader:
            opt_.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt_.step()
        model.eval()
        with torch.no_grad():
            vloss = float(crit(model(Xva), yva).item())
        if vloss + 1e-9 < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(Xte).numpy()
    pred = sy.inverse_transform(pred_s).flatten()
    ytrue = sy.inverse_transform(yte.numpy()).flatten()
    rmse = float(np.sqrt(mean_squared_error(ytrue, pred)))
    mae = float(mean_absolute_error(ytrue, pred))
    return rmse, mae


# ====================================================================
# 9. Configuration
# ====================================================================
VARIANT_NAMES = {
    "A": "(A) No prior (λ₂=0)",
    "B": "(B) Fixed τ=1",
    "C": "(C) Full model (learn τ)",
    "D": "(D) No L1 (λ₁=0)",
    "E": "(E) Prior on lags only",
    "F": "(F) Hard mask",
}

VARIANT_SHORT = {
    "A": "NoPrior",
    "B": "FixedTau",
    "C": "FullModel",
    "D": "NoL1",
    "E": "LagsOnly",
    "F": "HardMask",
}


@dataclass
class Cfg:
    # Synthetic data params
    synth_dims:       List[int]   = field(default_factory=lambda: [10, 20])
    synth_Ts:         List[int]   = field(default_factory=lambda: [500, 1000])
    synth_noises:     List[str]   = field(default_factory=lambda: ["gaussian"])
    synth_K:          int         = 1
    synth_edge_prob:  float       = 0.15
    synth_prior_accs: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.7, 0.9])

    # Lorenz-96 params
    lorenz_d:         int   = 10
    lorenz_T:         int   = 2000
    lorenz_F:         float = 10.0

    # Real data
    electricity_xlsx:  str = r"E:\electricity\0227test.xlsx"
    electricity_prior: str = r"E:\electricity\Auto_Generated_Prior.csv"
    elec_targets:      List[str] = field(default_factory=lambda: [
        "大工业电量", "居民生活", "商业用电"])
    elec_top_m:        int   = 6

    # Seeds
    seeds:            List[int] = field(default_factory=lambda: list(range(5)))

    # Optimization
    lambda1:          float = 0.01
    lambda2:          float = 0.01
    max_iter:         int   = 30
    inner_iter:       int   = 500
    lr:               float = 1e-2

    # Forecasting
    seq_len:          int   = 7
    max_epochs:       int   = 50
    fc_lr:            float = 3e-3
    hidden:           int   = 32
    patience:         int   = 8

    # Hard mask threshold
    mask_threshold:   float = 0.5

    # Which ablation variants to run
    variants:         List[str] = field(default_factory=lambda: [
        "A", "B", "C", "D", "E", "F"])

    # Which experiments
    do_synthetic:     bool = True
    do_lorenz:        bool = True
    do_real:          bool = True

    # Output
    output_dir:       str = "exp4_results"


def cfg_quick():
    return Cfg(
        synth_dims=[10], synth_Ts=[500],
        synth_prior_accs=[0.3, 0.7],
        lorenz_T=500,
        seeds=list(range(3)),
        max_iter=15, inner_iter=100,
        max_epochs=20,
        elec_targets=["大工业电量"],
        do_lorenz=False,
    )


def cfg_full():
    return Cfg(
        synth_dims=[10, 20, 50],
        synth_Ts=[500, 1000, 2000],
        synth_noises=["gaussian", "laplace", "student_t"],
        synth_prior_accs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        lorenz_d=20,
        lorenz_T=2000,
        seeds=list(range(10)),
        max_epochs=80,
        elec_targets=["大工业电量", "居民生活", "商业用电", "非普工业"],
    )


# ====================================================================
# 10. Synthetic Ablation Runner
# ====================================================================
def run_synthetic_ablation(cfg: Cfg) -> pd.DataFrame:
    """
    Run all ablation variants on synthetic SVAR data.
    Reports graph quality metrics (AUROC, AUPRC, F1, SHD).
    """
    print("\n" + "=" * 60)
    print(">>> Synthetic Ablation: Graph Quality Metrics")
    print("=" * 60)

    all_rows = []
    settings = []
    for d in cfg.synth_dims:
        for T in cfg.synth_Ts:
            for nt in cfg.synth_noises:
                for acc in cfg.synth_prior_accs:
                    for seed in cfg.seeds:
                        settings.append(dict(d=d, T=T, noise=nt,
                                             prior_acc=acc, seed=seed))

    n_total = len(settings) * len(cfg.variants)
    print(f"  {len(settings)} settings x {len(cfg.variants)} variants "
          f"= {n_total} runs")
    t_global = time.time()

    for idx, st in enumerate(settings):
        d, T = st["d"], st["T"]
        nt, acc, seed = st["noise"], st["prior_acc"], st["seed"]
        K = cfg.synth_K
        base = dict(data="synthetic", d=d, T=T, noise=nt,
                    prior_acc=acc, seed=seed)

        # Generate ground truth
        W0_true = make_er_dag(d, cfg.synth_edge_prob, seed=seed)
        Wk_true = make_lag_matrices(d, K, seed=seed)

        # Simulate
        try:
            X = simulate_svar_linear(T, W0_true, Wk_true, nt, seed=seed)
        except Exception:
            continue
        if not np.all(np.isfinite(X)):
            continue

        # Generate prior
        P_prior = gen_prior(W0_true, Wk_true, acc, "random", seed=seed + 999)

        # Standardize
        X_std = standardize(X)

        # Progress
        if (idx + 1) % max(1, len(settings) // 10) == 0 or idx == 0:
            elapsed = time.time() - t_global
            print(f"  [{idx+1}/{len(settings)}] d={d} T={T} {nt} "
                  f"acc={acc} s={seed}  ({elapsed:.0f}s)")

        # Run each variant
        for var in cfg.variants:
            try:
                t0 = time.time()
                if var == "A":
                    W0_est, Wk_est, tau = run_variant_A(
                        X_std, d, K, seed, cfg.max_iter, cfg.inner_iter,
                        cfg.lr, cfg.lambda1)
                elif var == "B":
                    W0_est, Wk_est, tau = run_variant_B(
                        X_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                elif var == "C":
                    W0_est, Wk_est, tau = run_variant_C(
                        X_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                elif var == "D":
                    W0_est, Wk_est, tau = run_variant_D(
                        X_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda2)
                elif var == "E":
                    W0_est, Wk_est, tau = run_variant_E(
                        X_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                elif var == "F":
                    W0_est, Wk_est, tau = run_variant_F(
                        X_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1,
                        cfg.mask_threshold)
                else:
                    continue

                elapsed_run = time.time() - t0

                # Evaluate W0 only
                met_w0 = compute_graph_metrics(W0_true, W0_est)

                # Evaluate combined (W0 + Wk)
                W_comb_est = combine_W0_Wk(W0_est, Wk_est)
                B_comb_true = (np.abs(W0_true) > 1e-10).astype(int)
                for Wk_t in Wk_true:
                    B_comb_true = np.maximum(
                        B_comb_true, (np.abs(Wk_t) > 1e-10).astype(int))
                met_comb = compute_graph_metrics(
                    B_comb_true.astype(float) * 0.5, W_comb_est)

                all_rows.append({
                    **base,
                    "variant": var,
                    "variant_name": VARIANT_SHORT[var],
                    "tau": tau,
                    "time": elapsed_run,
                    # W0-only metrics
                    "w0_auroc": met_w0["auroc"],
                    "w0_auprc": met_w0["auprc"],
                    "w0_f1": met_w0["f1_opt"],
                    "w0_shd_norm": met_w0["shd_norm"],
                    # Combined metrics
                    "comb_auroc": met_comb["auroc"],
                    "comb_auprc": met_comb["auprc"],
                    "comb_f1": met_comb["f1_opt"],
                    "comb_shd_norm": met_comb["shd_norm"],
                })
            except Exception as e:
                warnings.warn(f"Variant {var} failed: d={d} T={T} "
                              f"acc={acc} seed={seed}: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        p = os.path.join(cfg.output_dir, "ablation_synthetic.csv")
        df.to_csv(p, index=False)
        print(f">>> {p} ({len(df)} rows)")
    return df


# ====================================================================
# 11. Lorenz-96 Ablation Runner
# ====================================================================
def run_lorenz_ablation(cfg: Cfg) -> pd.DataFrame:
    """
    Run ablation on Lorenz-96 with known ground truth.
    Tests on a realistic nonlinear dynamical system.
    """
    print("\n" + "=" * 60)
    print(">>> Lorenz-96 Ablation: Graph Quality on Nonlinear System")
    print("=" * 60)

    d = cfg.lorenz_d
    T = cfg.lorenz_T
    K = cfg.synth_K
    all_rows = []

    for seed in cfg.seeds:
        X, B_true = generate_lorenz96(d=d, T=T, F=cfg.lorenz_F, seed=seed)
        if not np.all(np.isfinite(X)):
            continue

        for acc in cfg.synth_prior_accs:
            P_prior = gen_prior(B_true.astype(float) * 0.5, [], acc,
                                "random", seed=seed + 999)
            base = dict(data="lorenz96", d=d, T=T, noise="lorenz",
                        prior_acc=acc, seed=seed)

            for var in cfg.variants:
                try:
                    t0 = time.time()
                    if var == "A":
                        W0_est, Wk_est, tau = run_variant_A(
                            X, d, K, seed, cfg.max_iter, cfg.inner_iter,
                            cfg.lr, cfg.lambda1)
                    elif var == "B":
                        W0_est, Wk_est, tau = run_variant_B(
                            X, P_prior, d, K, seed, cfg.max_iter,
                            cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                    elif var == "C":
                        W0_est, Wk_est, tau = run_variant_C(
                            X, P_prior, d, K, seed, cfg.max_iter,
                            cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                    elif var == "D":
                        W0_est, Wk_est, tau = run_variant_D(
                            X, P_prior, d, K, seed, cfg.max_iter,
                            cfg.inner_iter, cfg.lr, cfg.lambda2)
                    elif var == "E":
                        W0_est, Wk_est, tau = run_variant_E(
                            X, P_prior, d, K, seed, cfg.max_iter,
                            cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                    elif var == "F":
                        W0_est, Wk_est, tau = run_variant_F(
                            X, P_prior, d, K, seed, cfg.max_iter,
                            cfg.inner_iter, cfg.lr, cfg.lambda1,
                            cfg.mask_threshold)
                    else:
                        continue

                    W_comb = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_graph_metrics(B_true.astype(float), W_comb)
                    all_rows.append({
                        **base,
                        "variant": var,
                        "variant_name": VARIANT_SHORT[var],
                        "tau": tau,
                        "time": time.time() - t0,
                        "auroc": met["auroc"],
                        "auprc": met["auprc"],
                        "f1_opt": met["f1_opt"],
                        "shd_norm": met["shd_norm"],
                        "f1_topk": met["f1_topk"],
                    })
                except Exception as e:
                    warnings.warn(f"Lorenz variant {var} failed "
                                  f"acc={acc} seed={seed}: {e}")

        print(f"  seed={seed} done")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        p = os.path.join(cfg.output_dir, "ablation_lorenz96.csv")
        df.to_csv(p, index=False)
        print(f">>> {p} ({len(df)} rows)")
    return df


# ====================================================================
# 12. Real Data Ablation Runner (Downstream RMSE)
# ====================================================================
def run_real_ablation(cfg: Cfg) -> pd.DataFrame:
    """
    Run ablation on electricity data via downstream forecasting RMSE.
    Each variant discovers a causal graph, selects features, trains LSTM.
    """
    print("\n" + "=" * 60)
    print(">>> Real Data Ablation: Downstream Forecasting RMSE")
    print("=" * 60)

    # Load data
    try:
        df_ts = pd.read_excel(cfg.electricity_xlsx, index_col=0)
    except Exception as e:
        warnings.warn(f"Failed to load electricity data: {e}")
        return pd.DataFrame()

    df_diff = df_ts.diff(periods=7).dropna()
    col_names = df_diff.columns.tolist()
    d = len(col_names)
    split = int(len(df_diff) * 0.8)
    X_all = df_diff.values

    # Load prior
    P_prior = np.full((d, d), 0.5)
    if os.path.exists(cfg.electricity_prior):
        P_prior = pd.read_csv(cfg.electricity_prior, index_col=0).values
        if P_prior.shape != (d, d):
            P_prior = np.full((d, d), 0.5)

    # Verify targets
    valid_targets = [t for t in cfg.elec_targets if t in col_names]
    if not valid_targets:
        warnings.warn("No valid electricity targets found!")
        return pd.DataFrame()

    print(f"  d={d}, T={len(df_diff)}, split={split}")
    print(f"  targets: {[ZH_TO_EN.get(t, t) for t in valid_targets]}")

    X_train = X_all[:split]
    mu = X_train.mean(0)
    sd = X_train.std(0)
    sd[sd == 0] = 1.0
    X_train_std = (X_train - mu) / sd

    K = cfg.synth_K
    all_rows = []

    for seed in cfg.seeds:
        print(f"\n  --- Seed {seed} ---")

        # Discover graph for each variant
        variant_graphs = {}
        for var in cfg.variants:
            try:
                t0 = time.time()
                if var == "A":
                    W0, Wk, tau = run_variant_A(
                        X_train_std, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1)
                elif var == "B":
                    W0, Wk, tau = run_variant_B(
                        X_train_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                elif var == "C":
                    W0, Wk, tau = run_variant_C(
                        X_train_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                elif var == "D":
                    W0, Wk, tau = run_variant_D(
                        X_train_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda2)
                elif var == "E":
                    W0, Wk, tau = run_variant_E(
                        X_train_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                elif var == "F":
                    W0, Wk, tau = run_variant_F(
                        X_train_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1,
                        cfg.mask_threshold)
                else:
                    continue
                W_comb = combine_W0_Wk(W0, Wk)
                variant_graphs[var] = (W_comb, tau)
                cd_time = time.time() - t0
                print(f"    Variant {var} ({VARIANT_SHORT[var]}): "
                      f"tau={tau:.4f}, {cd_time:.1f}s")
            except Exception as e:
                warnings.warn(f"Variant {var} failed seed={seed}: {e}")

        # Downstream forecasting for each target
        for target in valid_targets:
            target_idx = col_names.index(target)
            target_en = ZH_TO_EN.get(target, target)

            for var, (W_comb, tau) in variant_graphs.items():
                feats = select_top_parents(
                    W_comb, target_idx, cfg.elec_top_m, col_names)
                try:
                    rmse, mae = run_forecast(
                        df_diff, split, feats, target,
                        seq_len=cfg.seq_len, max_epochs=cfg.max_epochs,
                        lr=cfg.fc_lr, hidden=cfg.hidden,
                        seed=seed, patience=cfg.patience)
                    all_rows.append(dict(
                        data="electricity",
                        variant=var,
                        variant_name=VARIANT_SHORT[var],
                        seed=seed,
                        target=target_en,
                        top_m=cfg.elec_top_m,
                        n_features=len(feats),
                        tau=tau,
                        rmse=rmse,
                        mae=mae,
                    ))
                except Exception as e:
                    warnings.warn(f"Forecast {var}/{target_en} "
                                  f"seed={seed}: {e}")

            # Also: AllFeatures baseline (no feature selection)
            try:
                rmse, mae = run_forecast(
                    df_diff, split, col_names, target,
                    seq_len=cfg.seq_len, max_epochs=cfg.max_epochs,
                    lr=cfg.fc_lr, hidden=cfg.hidden,
                    seed=seed, patience=cfg.patience)
                all_rows.append(dict(
                    data="electricity",
                    variant="ALL",
                    variant_name="AllFeatures",
                    seed=seed,
                    target=target_en,
                    top_m=d,
                    n_features=d,
                    tau=float("nan"),
                    rmse=rmse,
                    mae=mae,
                ))
            except Exception:
                pass

        print(f"  seed={seed} done ({len(all_rows)} rows)")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        p = os.path.join(cfg.output_dir, "ablation_real.csv")
        df.to_csv(p, index=False)
        print(f">>> {p} ({len(df)} rows)")
    return df


# ====================================================================
# 13. Summary Tables
# ====================================================================
def generate_summaries(df_synth: pd.DataFrame, df_lorenz: pd.DataFrame,
                       df_real: pd.DataFrame, out: str):
    print("\n" + "=" * 72)
    print("ABLATION STUDY SUMMARY")
    print("=" * 72)

    # --- Synthetic: main ablation table ---
    if not df_synth.empty:
        metric_cols = ["w0_auroc", "w0_f1", "comb_auroc", "comb_f1",
                       "comb_shd_norm"]
        avail = [c for c in metric_cols if c in df_synth.columns]
        g = df_synth.groupby(["variant", "variant_name"])
        agg = g.agg(
            **{f"{c}_mean": (c, "mean") for c in avail},
            **{f"{c}_std": (c, "std") for c in avail},
            tau_mean=("tau", "mean"),
            tau_std=("tau", "std"),
            n=("seed", "count"),
        ).reset_index()
        p = os.path.join(out, "summary_synthetic_overall.csv")
        agg.to_csv(p, index=False)
        print(f">>> {p}")

        # By prior accuracy
        g2 = df_synth.groupby(["variant", "variant_name", "prior_acc"])
        agg2 = g2.agg(
            comb_auroc_mean=("comb_auroc", "mean"),
            comb_auroc_std=("comb_auroc", "std"),
            comb_f1_mean=("comb_f1", "mean"),
            comb_f1_std=("comb_f1", "std"),
            tau_mean=("tau", "mean"),
        ).reset_index()
        p2 = os.path.join(out, "summary_synthetic_by_prior.csv")
        agg2.to_csv(p2, index=False)
        print(f">>> {p2}")

        # By dimension
        g3 = df_synth.groupby(["variant", "variant_name", "d"])
        agg3 = g3.agg(
            comb_auroc_mean=("comb_auroc", "mean"),
            comb_f1_mean=("comb_f1", "mean"),
        ).reset_index()
        p3 = os.path.join(out, "summary_synthetic_by_dim.csv")
        agg3.to_csv(p3, index=False)
        print(f">>> {p3}")

        # Console
        print("\n--- Synthetic: Overall by Variant ---")
        overall = df_synth.groupby("variant_name")[
            [c for c in avail]].mean()
        print(overall.round(4).to_string())

    # --- Lorenz-96 ---
    if not df_lorenz.empty:
        g4 = df_lorenz.groupby(["variant", "variant_name"])
        agg4 = g4.agg(
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
            f1_mean=("f1_opt", "mean"), f1_std=("f1_opt", "std"),
            shd_norm_mean=("shd_norm", "mean"),
            tau_mean=("tau", "mean"),
            n=("seed", "count"),
        ).reset_index()
        p4 = os.path.join(out, "summary_lorenz96.csv")
        agg4.to_csv(p4, index=False)
        print(f"\n>>> {p4}")

        print("\n--- Lorenz-96: Overall by Variant ---")
        overall_l = df_lorenz.groupby("variant_name")[
            ["auroc", "f1_opt", "shd_norm"]].mean()
        print(overall_l.round(4).to_string())

    # --- Real data ---
    if not df_real.empty:
        g5 = df_real.groupby(["variant", "variant_name"])
        agg5 = g5.agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            tau_mean=("tau", "mean"),
            n=("seed", "count"),
        ).reset_index()
        p5 = os.path.join(out, "summary_real_overall.csv")
        agg5.to_csv(p5, index=False)
        print(f"\n>>> {p5}")

        # Per-target
        g6 = df_real.groupby(["variant_name", "target"])
        agg6 = g6.agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        ).reset_index()
        p6 = os.path.join(out, "summary_real_per_target.csv")
        agg6.to_csv(p6, index=False)
        print(f">>> {p6}")

        print("\n--- Real Data: RMSE by Variant ---")
        overall_r = df_real.groupby("variant_name")[["rmse", "mae"]].mean()
        overall_r = overall_r.sort_values("rmse")
        print(overall_r.round(4).to_string())

    print()


# ====================================================================
# 14. Figures
# ====================================================================
VARIANT_COLORS = {
    "NoPrior":    "#95A5A6",
    "FixedTau":   "#E67E22",
    "FullModel":  "#E74C3C",
    "NoL1":       "#9B59B6",
    "LagsOnly":   "#3498DB",
    "HardMask":   "#2C3E50",
    "AllFeatures": "#BDC3C7",
}

VARIANT_MARKERS = {
    "NoPrior":    "^",
    "FixedTau":   "s",
    "FullModel":  "o",
    "NoL1":       "D",
    "LagsOnly":   "v",
    "HardMask":   "P",
    "AllFeatures": "X",
}

VARIANT_ORDER = ["NoPrior", "FixedTau", "FullModel", "NoL1",
                 "LagsOnly", "HardMask"]


def _save(prefix):
    plt.savefig(prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(prefix + ".pdf", bbox_inches="tight")
    plt.close()
    print(f">>> {prefix}.png / .pdf")


def generate_figures(df_synth: pd.DataFrame, df_lorenz: pd.DataFrame,
                     df_real: pd.DataFrame, out: str):
    if not df_synth.empty:
        _fig_synth_main_bar(df_synth, out)
        _fig_synth_prior_curves(df_synth, out)
        _fig_synth_tau_analysis(df_synth, out)
        _fig_soft_vs_hard(df_synth, out)
        if df_synth["d"].nunique() > 1:
            _fig_synth_by_dim(df_synth, out)
    if not df_lorenz.empty:
        _fig_lorenz_bar(df_lorenz, out)
        _fig_lorenz_prior_curves(df_lorenz, out)
    if not df_real.empty:
        _fig_real_bar(df_real, out)
        _fig_real_per_target(df_real, out)
    # Combined summary figure
    _fig_combined_summary(df_synth, df_lorenz, df_real, out)


def _fig_synth_main_bar(df, out):
    """
    Figure 1: Main ablation bar chart on synthetic data.
    Shows AUROC and F1 for each variant, averaged across all settings.
    """
    variants = [v for v in VARIANT_ORDER if v in df["variant_name"].values]
    n_v = len(variants)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for metric, ax, yl in [("comb_auroc", axes[0], "AUROC (combined graph)"),
                            ("comb_f1", axes[1], "F1 (combined graph)")]:
        means = [df[df["variant_name"] == v][metric].mean() for v in variants]
        stds = [df[df["variant_name"] == v][metric].std() for v in variants]
        colors = [VARIANT_COLORS.get(v, "grey") for v in variants]
        bars = ax.bar(range(n_v), means, yerr=stds, color=colors,
                      alpha=0.85, capsize=5, edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(n_v))
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(yl, fontsize=12)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Ablation Study: Graph Discovery Quality (Synthetic Data)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig1_synth_ablation_bar"))


def _fig_synth_prior_curves(df, out):
    """
    Figure 2: F1 vs prior accuracy for each variant.
    Key figure: shows how each component handles prior quality changes.
    """
    if df["prior_acc"].nunique() < 3:
        return

    variants = [v for v in VARIANT_ORDER if v in df["variant_name"].values]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for metric, ax, yl in [("comb_f1", axes[0], "F1 (combined)"),
                            ("comb_auroc", axes[1], "AUROC (combined)")]:
        for v in variants:
            sub = df[df["variant_name"] == v]
            agg = sub.groupby("prior_acc").agg(
                y=(metric, "mean"), e=(metric, "std")).reset_index()
            ax.errorbar(agg["prior_acc"], agg["y"], yerr=agg["e"],
                        label=v,
                        color=VARIANT_COLORS.get(v, "grey"),
                        marker=VARIANT_MARKERS.get(v, "x"),
                        linewidth=2, markersize=7, capsize=3)
        ax.set_xlabel("Prior Accuracy", fontsize=12)
        ax.set_ylabel(yl, fontsize=12)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Ablation: Component Contribution vs Prior Quality",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig2_synth_prior_curves"))


def _fig_synth_tau_analysis(df, out):
    """
    Figure 3: Learned τ vs prior accuracy for variants that learn τ.
    Dual y-axis with F1 for the full model.
    """
    # Only variants C (FullModel), D (NoL1), E (LagsOnly) learn tau
    tau_variants = ["FullModel", "NoL1", "LagsOnly"]
    sub_all = df[df["variant_name"].isin(tau_variants)]
    if sub_all.empty:
        return
    sub_all = sub_all.dropna(subset=["tau"])
    if sub_all["prior_acc"].nunique() < 3:
        return

    fig, ax1 = plt.subplots(figsize=(9, 6))
    c_lines = {"FullModel": "#E74C3C", "NoL1": "#9B59B6",
               "LagsOnly": "#3498DB"}

    for v in tau_variants:
        sub = sub_all[sub_all["variant_name"] == v]
        if sub.empty:
            continue
        agg = sub.groupby("prior_acc").agg(
            tau_m=("tau", "mean"), tau_s=("tau", "std")).reset_index()
        ax1.errorbar(agg["prior_acc"], agg["tau_m"], yerr=agg["tau_s"],
                     color=c_lines.get(v, "grey"),
                     marker=VARIANT_MARKERS.get(v, "o"),
                     lw=2.5, ms=8, capsize=4,
                     label=f"τ ({v})")

    ax1.set_xlabel("Prior Accuracy", fontsize=13)
    ax1.set_ylabel("Learned Temperature τ", fontsize=13, color="#E74C3C")
    ax1.tick_params(axis="y", labelcolor="#E74C3C")

    # Second y-axis: F1 for FullModel
    sub_full = df[df["variant_name"] == "FullModel"]
    if not sub_full.empty and sub_full["prior_acc"].nunique() >= 3:
        ax2 = ax1.twinx()
        agg_f1 = sub_full.groupby("prior_acc").agg(
            f1_m=("comb_f1", "mean"), f1_s=("comb_f1", "std")).reset_index()
        ax2.errorbar(agg_f1["prior_acc"], agg_f1["f1_m"], yerr=agg_f1["f1_s"],
                     color="#2C3E50", marker="D", lw=2.5, ms=8,
                     capsize=4, ls="--", label="F1 (FullModel)")
        ax2.set_ylabel("F1 (combined)", fontsize=13, color="#2C3E50")
        ax2.tick_params(axis="y", labelcolor="#2C3E50")
        h2, l2 = ax2.get_legend_handles_labels()
    else:
        h2, l2 = [], []

    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax1.set_title("Temperature τ Adaptation Across Ablation Variants",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    _save(os.path.join(out, "fig3_tau_analysis"))


def _fig_soft_vs_hard(df, out):
    """
    Figure 4: Soft probabilistic prior (FullModel) vs Hard mask (HardMask).
    Strategic figure proving the value of soft calibration.
    """
    df_cf = df[df["variant_name"].isin(["FullModel", "HardMask"])]
    if df_cf.empty or df_cf["prior_acc"].nunique() < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for metric, ax, yl in [("comb_f1", axes[0], "F1 (combined)"),
                            ("comb_auroc", axes[1], "AUROC (combined)")]:
        for v in ["FullModel", "HardMask"]:
            sub = df_cf[df_cf["variant_name"] == v]
            agg = sub.groupby("prior_acc").agg(
                y=(metric, "mean"), e=(metric, "std")).reset_index()
            lw = 3.0 if v == "FullModel" else 2.0
            ax.errorbar(agg["prior_acc"], agg["y"], yerr=agg["e"],
                        label=f"{v} (soft)" if v == "FullModel"
                        else f"{v} (binary)",
                        color=VARIANT_COLORS.get(v, "grey"),
                        marker=VARIANT_MARKERS.get(v, "x"),
                        linewidth=lw, markersize=8, capsize=4)

        # Also show NoPrior as reference
        sub_np = df[df["variant_name"] == "NoPrior"]
        if not sub_np.empty:
            np_mean = sub_np[metric].mean()
            ax.axhline(y=np_mean, color=VARIANT_COLORS["NoPrior"],
                       ls="--", alpha=0.7, lw=1.5,
                       label="NoPrior (baseline)")

        ax.set_xlabel("Prior Accuracy", fontsize=12)
        ax.set_ylabel(yl, fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Soft Probabilistic Prior vs Hard Binary Mask\n"
                 "(Same Domain Knowledge, Different Integration Strategy)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig4_soft_vs_hard"))


def _fig_synth_by_dim(df, out):
    """
    Figure 5: Performance by dimension d for each variant.
    """
    variants = [v for v in VARIANT_ORDER if v in df["variant_name"].values]
    dims = sorted(df["d"].unique())
    n_d = len(dims)
    n_v = len(variants)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_d)
    w = 0.8 / max(n_v, 1)

    for i, v in enumerate(variants):
        vals, errs = [], []
        for d_val in dims:
            s = df[(df["variant_name"] == v) & (df["d"] == d_val)]
            vals.append(s["comb_f1"].mean() if len(s) else 0)
            errs.append(s["comb_f1"].std() if len(s) else 0)
        ax.bar(x + i * w, vals, w, yerr=errs,
               label=v, color=VARIANT_COLORS.get(v, "grey"),
               alpha=0.85, capsize=2)

    ax.set_xlabel("Dimension d", fontsize=12)
    ax.set_ylabel("F1 (combined)", fontsize=12)
    ax.set_xticks(x + w * (n_v - 1) / 2)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.set_title("Ablation: Scalability Across Dimensions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig5_synth_by_dim"))


def _fig_lorenz_bar(df, out):
    """
    Figure 6: Lorenz-96 ablation bar chart.
    """
    variants = [v for v in VARIANT_ORDER if v in df["variant_name"].values]
    n_v = len(variants)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for metric, ax, yl in [("auroc", axes[0], "AUROC"),
                            ("f1_opt", axes[1], "F1")]:
        means = [df[df["variant_name"] == v][metric].mean() for v in variants]
        stds = [df[df["variant_name"] == v][metric].std() for v in variants]
        colors = [VARIANT_COLORS.get(v, "grey") for v in variants]
        bars = ax.bar(range(n_v), means, yerr=stds, color=colors,
                      alpha=0.85, capsize=5, edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(n_v))
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(yl, fontsize=12)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Ablation Study: Lorenz-96 (Nonlinear Dynamical System)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig6_lorenz_ablation"))


def _fig_lorenz_prior_curves(df, out):
    """
    Figure 7: Lorenz-96 F1 vs prior accuracy.
    """
    if df["prior_acc"].nunique() < 3:
        return

    variants = [v for v in VARIANT_ORDER if v in df["variant_name"].values]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for v in variants:
        sub = df[df["variant_name"] == v]
        agg = sub.groupby("prior_acc").agg(
            y=("f1_opt", "mean"), e=("f1_opt", "std")).reset_index()
        ax.errorbar(agg["prior_acc"], agg["y"], yerr=agg["e"],
                    label=v,
                    color=VARIANT_COLORS.get(v, "grey"),
                    marker=VARIANT_MARKERS.get(v, "x"),
                    linewidth=2, markersize=7, capsize=3)

    ax.set_xlabel("Prior Accuracy", fontsize=12)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title("Lorenz-96: Ablation vs Prior Quality",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    _save(os.path.join(out, "fig7_lorenz_prior_curves"))


def _fig_real_bar(df, out):
    """
    Figure 8: Real data ablation — RMSE bar chart.
    """
    variants = [v for v in (VARIANT_ORDER + ["AllFeatures"])
                if v in df["variant_name"].values]
    n_v = len(variants)

    fig, ax = plt.subplots(figsize=(10, 6))
    means = [df[df["variant_name"] == v]["rmse"].mean() for v in variants]
    stds = [df[df["variant_name"] == v]["rmse"].std() for v in variants]
    colors = [VARIANT_COLORS.get(v, "grey") for v in variants]

    bars = ax.bar(range(n_v), means, yerr=stds, color=colors,
                  alpha=0.85, capsize=5, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(n_v))
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("RMSE (mean +/- std)", fontsize=12)
    ax.set_title("Ablation Study: Downstream Forecasting RMSE\n"
                 "(Electricity Dataset, LSTM Predictor)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _save(os.path.join(out, "fig8_real_ablation_bar"))


def _fig_real_per_target(df, out):
    """
    Figure 9: Per-target RMSE heatmap for real data.
    """
    targets = sorted(df["target"].unique())
    variants = [v for v in (VARIANT_ORDER + ["AllFeatures"])
                if v in df["variant_name"].values]
    if len(targets) < 2 or len(variants) < 2:
        return

    piv = df.groupby(["target", "variant_name"])["rmse"].mean()
    piv = piv.unstack("variant_name").fillna(0)
    # Reorder columns
    piv = piv[[v for v in variants if v in piv.columns]]

    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 1.2),
                                    max(3, len(targets) * 0.8)))
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=10)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if val > piv.values.mean() else "black")
    ax.set_title("Per-Target RMSE: Ablation Variants",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="RMSE", shrink=0.8)
    plt.tight_layout()
    _save(os.path.join(out, "fig9_real_per_target"))


def _fig_combined_summary(df_synth, df_lorenz, df_real, out):
    """
    Figure 10: Combined summary — one row per variant,
    showing synthetic AUROC, Lorenz F1, real RMSE side by side.
    """
    variants = VARIANT_ORDER
    has_synth = not df_synth.empty
    has_lorenz = not df_lorenz.empty
    has_real = not df_real.empty

    n_panels = sum([has_synth, has_lorenz, has_real])
    if n_panels < 2:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if has_synth:
        ax = axes[panel_idx]
        panel_idx += 1
        vs = [v for v in variants if v in df_synth["variant_name"].values]
        means = [df_synth[df_synth["variant_name"] == v]["comb_auroc"].mean()
                 for v in vs]
        stds = [df_synth[df_synth["variant_name"] == v]["comb_auroc"].std()
                for v in vs]
        colors = [VARIANT_COLORS.get(v, "grey") for v in vs]
        ax.barh(range(len(vs)), means, xerr=stds, color=colors,
                alpha=0.85, capsize=3)
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels(vs, fontsize=10)
        ax.set_xlabel("AUROC")
        ax.set_title("Synthetic\n(Graph Quality)", fontsize=12,
                     fontweight="bold")
        ax.grid(True, axis="x", ls=":", alpha=0.4)
        ax.invert_yaxis()

    if has_lorenz:
        ax = axes[panel_idx]
        panel_idx += 1
        vs = [v for v in variants if v in df_lorenz["variant_name"].values]
        means = [df_lorenz[df_lorenz["variant_name"] == v]["f1_opt"].mean()
                 for v in vs]
        stds = [df_lorenz[df_lorenz["variant_name"] == v]["f1_opt"].std()
                for v in vs]
        colors = [VARIANT_COLORS.get(v, "grey") for v in vs]
        ax.barh(range(len(vs)), means, xerr=stds, color=colors,
                alpha=0.85, capsize=3)
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels(vs, fontsize=10)
        ax.set_xlabel("F1")
        ax.set_title("Lorenz-96\n(Nonlinear System)", fontsize=12,
                     fontweight="bold")
        ax.grid(True, axis="x", ls=":", alpha=0.4)
        ax.invert_yaxis()

    if has_real:
        ax = axes[panel_idx]
        panel_idx += 1
        vs = [v for v in (variants + ["AllFeatures"])
              if v in df_real["variant_name"].values]
        means = [df_real[df_real["variant_name"] == v]["rmse"].mean()
                 for v in vs]
        stds = [df_real[df_real["variant_name"] == v]["rmse"].std()
                for v in vs]
        colors = [VARIANT_COLORS.get(v, "grey") for v in vs]
        ax.barh(range(len(vs)), means, xerr=stds, color=colors,
                alpha=0.85, capsize=3)
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels(vs, fontsize=10)
        ax.set_xlabel("RMSE (lower is better)")
        ax.set_title("Electricity\n(Downstream RMSE)", fontsize=12,
                     fontweight="bold")
        ax.grid(True, axis="x", ls=":", alpha=0.4)
        ax.invert_yaxis()

    fig.suptitle("Ablation Study: Combined View Across Domains",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig10_combined_summary"))


# ====================================================================
# 15. Soft vs Hard Detailed Analysis
# ====================================================================
def run_soft_vs_hard_analysis(cfg: Cfg) -> pd.DataFrame:
    """
    Focused experiment: sweep prior accuracy densely, compare
    FullModel (soft) vs HardMask (binary) vs NoPrior (baseline).
    Multiple mask thresholds for hard mask.
    """
    print("\n" + "=" * 60)
    print(">>> Focused Analysis: Soft Prior vs Hard Mask")
    print("=" * 60)

    d = 20
    T = 1000
    K = cfg.synth_K
    accs = np.linspace(0, 1, 11).tolist()
    mask_thresholds = [0.3, 0.5, 0.7]

    all_rows = []
    for seed in cfg.seeds:
        W0_true = make_er_dag(d, cfg.synth_edge_prob, seed=seed)
        Wk_true = make_lag_matrices(d, K, seed=seed)
        X = simulate_svar_linear(T, W0_true, Wk_true, "laplace", seed=seed)
        if not np.all(np.isfinite(X)):
            continue
        X_std = standardize(X)

        for acc in accs:
            P_prior = gen_prior(W0_true, Wk_true, acc, "random",
                                seed=seed + 999)
            base = dict(d=d, T=T, prior_acc=acc, seed=seed)

            # NoPrior
            try:
                W0e, Wke, tau = run_variant_A(
                    X_std, d, K, seed, cfg.max_iter, cfg.inner_iter,
                    cfg.lr, cfg.lambda1)
                W_c = combine_W0_Wk(W0e, Wke)
                B_true_comb = (np.abs(W0_true) > 1e-10).astype(int)
                for Wk_t in Wk_true:
                    B_true_comb = np.maximum(
                        B_true_comb, (np.abs(Wk_t) > 1e-10).astype(int))
                met = compute_graph_metrics(B_true_comb.astype(float) * 0.5, W_c)
                all_rows.append({
                    **base, "method": "NoPrior", "mask_thr": float("nan"),
                    "tau": tau, **met})
            except Exception:
                pass

            # FullModel (soft)
            try:
                W0e, Wke, tau = run_variant_C(
                    X_std, P_prior, d, K, seed, cfg.max_iter,
                    cfg.inner_iter, cfg.lr, cfg.lambda1, cfg.lambda2)
                W_c = combine_W0_Wk(W0e, Wke)
                met = compute_graph_metrics(B_true_comb.astype(float) * 0.5, W_c)
                all_rows.append({
                    **base, "method": "Soft (FullModel)",
                    "mask_thr": float("nan"),
                    "tau": tau, **met})
            except Exception:
                pass

            # HardMask at multiple thresholds
            for mt in mask_thresholds:
                try:
                    W0e, Wke, tau = run_variant_F(
                        X_std, P_prior, d, K, seed, cfg.max_iter,
                        cfg.inner_iter, cfg.lr, cfg.lambda1, mt)
                    W_c = combine_W0_Wk(W0e, Wke)
                    met = compute_graph_metrics(
                        B_true_comb.astype(float) * 0.5, W_c)
                    all_rows.append({
                        **base, "method": f"Hard (thr={mt})",
                        "mask_thr": mt,
                        "tau": tau, **met})
                except Exception:
                    pass

        print(f"  seed={seed} done")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        p = os.path.join(cfg.output_dir, "soft_vs_hard_detailed.csv")
        df.to_csv(p, index=False)
        print(f">>> {p} ({len(df)} rows)")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        methods_ordered = (["NoPrior", "Soft (FullModel)"]
                           + [f"Hard (thr={mt})" for mt in mask_thresholds])
        method_colors = {
            "NoPrior": "#95A5A6",
            "Soft (FullModel)": "#E74C3C",
            "Hard (thr=0.3)": "#2C3E50",
            "Hard (thr=0.5)": "#3498DB",
            "Hard (thr=0.7)": "#27AE60",
        }
        method_markers = {
            "NoPrior": "^",
            "Soft (FullModel)": "o",
            "Hard (thr=0.3)": "s",
            "Hard (thr=0.5)": "D",
            "Hard (thr=0.7)": "v",
        }

        for metric, ax, yl in [("f1_opt", axes[0], "F1"),
                                ("auroc", axes[1], "AUROC")]:
            for m in methods_ordered:
                sub = df[df["method"] == m]
                if sub.empty:
                    continue
                agg = sub.groupby("prior_acc").agg(
                    y=(metric, "mean"), e=(metric, "std")).reset_index()
                lw = 3.0 if "Soft" in m else (1.5 if "NoPrior" in m else 2.0)
                ax.errorbar(agg["prior_acc"], agg["y"], yerr=agg["e"],
                            label=m,
                            color=method_colors.get(m, "grey"),
                            marker=method_markers.get(m, "x"),
                            linewidth=lw, markersize=7, capsize=3)
            ax.set_xlabel("Prior Accuracy", fontsize=12)
            ax.set_ylabel(yl, fontsize=12)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, ls=":", alpha=0.5)
            ax.set_xlim(-0.05, 1.05)

        fig.suptitle("Soft Probabilistic vs Hard Mask: Detailed Comparison\n"
                     "(d=20, T=1000, multiple mask thresholds)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(os.path.join(out, "fig_soft_vs_hard_detailed"))

    return df


# ====================================================================
# 16. Entry Point
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Extended Ablation Study")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny test run")
    parser.add_argument("--full", action="store_true",
                        help="Full NeurIPS sweep")
    parser.add_argument("--sub", type=str, default=None,
                        choices=["synthetic", "real", "lorenz",
                                 "hard_mask"],
                        help="Run a specific sub-experiment")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--variants", nargs="+", type=str, default=None,
                        choices=["A", "B", "C", "D", "E", "F"],
                        help="Specific variants to test")
    parser.add_argument("--dims", nargs="+", type=int, default=None)
    parser.add_argument("--Ts", nargs="+", type=int, default=None)
    parser.add_argument("--prior-accs", nargs="+", type=float, default=None)
    args = parser.parse_args()

    # Select config
    if args.quick:
        cfg = cfg_quick()
    elif args.full:
        cfg = cfg_full()
    else:
        cfg = Cfg()

    # Sub-experiment overrides
    if args.sub == "synthetic":
        cfg.do_lorenz = False
        cfg.do_real = False
    elif args.sub == "real":
        cfg.do_synthetic = False
        cfg.do_lorenz = False
    elif args.sub == "lorenz":
        cfg.do_synthetic = False
        cfg.do_real = False
    elif args.sub == "hard_mask":
        pass  # handled separately below

    # CLI overrides
    if args.output:     cfg.output_dir     = args.output
    if args.seeds:      cfg.seeds          = args.seeds
    if args.variants:   cfg.variants       = args.variants
    if args.dims:       cfg.synth_dims     = args.dims
    if args.Ts:         cfg.synth_Ts       = args.Ts
    if args.prior_accs: cfg.synth_prior_accs = args.prior_accs

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 64)
    print(" Experiment 4: Extended Ablation Study")
    print("=" * 64)
    print(f"  variants     = {cfg.variants}")
    print(f"  seeds        = {cfg.seeds}")
    if cfg.do_synthetic:
        print(f"  synth dims   = {cfg.synth_dims}")
        print(f"  synth Ts     = {cfg.synth_Ts}")
        print(f"  synth noise  = {cfg.synth_noises}")
        print(f"  prior_accs   = {cfg.synth_prior_accs}")
    if cfg.do_lorenz:
        print(f"  lorenz       = d={cfg.lorenz_d}, T={cfg.lorenz_T}")
    if cfg.do_real:
        print(f"  elec targets = {cfg.elec_targets}")
    print(f"  output       = {cfg.output_dir}")
    print("  Variant descriptions:")
    for v in cfg.variants:
        print(f"    {v}: {VARIANT_NAMES[v]}")
    print("=" * 64)

    t_global = time.time()
    df_synth = pd.DataFrame()
    df_lorenz = pd.DataFrame()
    df_real = pd.DataFrame()

    # Special sub-experiment: detailed soft vs hard analysis
    if args.sub == "hard_mask":
        run_soft_vs_hard_analysis(cfg)
        elapsed = time.time() - t_global
        print(f"\n>>> Soft vs Hard analysis done in {elapsed:.1f}s")
        print(f">>> Results in: {cfg.output_dir}/")
        return

    # Main ablation experiments
    if cfg.do_synthetic:
        df_synth = run_synthetic_ablation(cfg)

    if cfg.do_lorenz:
        df_lorenz = run_lorenz_ablation(cfg)

    if cfg.do_real:
        df_real = run_real_ablation(cfg)

    # Summaries and figures
    generate_summaries(df_synth, df_lorenz, df_real, cfg.output_dir)
    generate_figures(df_synth, df_lorenz, df_real, cfg.output_dir)

    # Optionally run soft vs hard detailed analysis
    if "F" in cfg.variants and cfg.do_synthetic:
        run_soft_vs_hard_analysis(cfg)

    # Save combined results
    dfs = []
    if not df_synth.empty:
        dfs.append(("synthetic", df_synth))
    if not df_lorenz.empty:
        dfs.append(("lorenz96", df_lorenz))
    if not df_real.empty:
        dfs.append(("real", df_real))
    if dfs:
        combined_rows = []
        for name, df in dfs:
            df2 = df.copy()
            df2["experiment"] = name
            combined_rows.append(df2)
        df_combined = pd.concat(combined_rows, ignore_index=True)
        p = os.path.join(cfg.output_dir, "exp4_full_results.csv")
        df_combined.to_csv(p, index=False)
        print(f"\n>>> Combined results: {p} ({len(df_combined)} rows)")

    elapsed = time.time() - t_global
    print(f"\n>>> Experiment 4 complete in {elapsed:.1f}s")
    print(f">>> Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
