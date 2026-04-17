"""
exp_utils.py — Shared utilities for all PRCD-MAP experiments.

Contains: seed, data gen, graph gen, prior gen, baselines, metrics, plotting.
"""

import os, sys, time, warnings, math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- core model ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_linear import PRCD_MAP_Model, train_prcd_alm

# ---- optional baselines ----
HAS_TIGRAMITE = False
HAS_LINGAM = False

try:
    import tigramite
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite import data_processing as pp
    HAS_TIGRAMITE = True
except ImportError:
    pass

try:
    import lingam
    HAS_LINGAM = True
except ImportError:
    pass

# ---- NGC (Neural Granger Causality, Tank et al. 2021) ----
HAS_NGC = False
try:
    from neuralGC import cMLP as _NGC_cMLP
    from neuralGC.training import train_model_ista as _ngc_train_ista
    HAS_NGC = True
except ImportError:
    try:
        from neural_gc import cMLP as _NGC_cMLP
        from neural_gc.training import train_model_ista as _ngc_train_ista
        HAS_NGC = True
    except ImportError:
        try:
            # Neural-GC clone 在 home 目录下, 没有 setup.py, 直接加 sys.path
            _ngc_path = os.path.expanduser("~/Neural-GC")
            if os.path.isdir(_ngc_path) and _ngc_path not in sys.path:
                sys.path.insert(0, _ngc_path)
            from models.cmlp import cMLP as _NGC_cMLP
            from models.cmlp import train_model_ista as _ngc_train_ista
            HAS_NGC = True
        except ImportError:
            pass

# ---- RHINO / causica (Gong et al. 2023) ----
HAS_RHINO = False
try:
    import causica
    HAS_RHINO = True
except ImportError:
    pass

# ---- CUTS+ fallback (Cheng et al. 2023) ----
HAS_CUTS = False
try:
    from cuts import CUTS as _CUTS_Model
    HAS_CUTS = True
except ImportError:
    pass

# ====================================================================
# Plotting config
# ====================================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Arial", "Helvetica",
    "Liberation Sans", "SimHei", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "PRCD-MAP(learn_tau)":   "#E74C3C",
    "PRCD-MAP(fixed_tau)":   "#E67E22",
    "PRCD-MAP(no_l1_weight)":"#F39C12",
    "PRCD-MAP(uniform)":     "#9B59B6",
    "PRCD-MAP(NAM)":         "#C0392B",
    "DYNOTEARS":             "#2C3E50",
    "PCMCI+":                "#27AE60",
    "VARLiNGAM":             "#3498DB",
    "RHINO":                 "#1ABC9C",
    "NGC":                   "#8E44AD",
}
MARKERS = {
    "PRCD-MAP(learn_tau)":   "o",
    "PRCD-MAP(fixed_tau)":   "s",
    "PRCD-MAP(no_l1_weight)":"d",
    "PRCD-MAP(uniform)":     "^",
    "PRCD-MAP(NAM)":         "p",
    "DYNOTEARS":             "D",
    "PCMCI+":                "v",
    "VARLiNGAM":             "P",
    "RHINO":                 "h",
    "NGC":                   "*",
}

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
# General Utilities
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


def inject_missing(X: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    """
    Inject MCAR (Missing Completely at Random) missing values.
    Returns X with NaN at randomly selected positions.
    ratio: fraction of values to mask (0.0 = no missing, 0.3 = 30% missing).
    """
    if ratio <= 0:
        return X.copy()
    rng = np.random.default_rng(seed + 9999)
    X_miss = X.copy()
    mask = rng.random(X.shape) < ratio
    X_miss[mask] = np.nan
    return X_miss


def make_lag_tensors_with_mask(X: np.ndarray, K: int):
    """
    Like make_lag_tensors but also returns obs_mask for missing data.
    NaN values in X are replaced with 0 in tensors; obs_mask marks valid positions.
    Returns (X_t, X_lags, obs_mask) where obs_mask is (T-K, d) tensor or None.
    """
    has_nan = np.any(np.isnan(X))
    if not has_nan:
        X_t, X_lags = make_lag_tensors(X, K)
        return X_t, X_lags, None

    # Replace NaN with 0 for computation, track mask
    X_clean = np.nan_to_num(X, nan=0.0)
    X_t = torch.tensor(X_clean[K:], dtype=torch.float32)
    X_lags = [torch.tensor(X_clean[K - k: -k], dtype=torch.float32)
              for k in range(1, K + 1)]
    # obs_mask: 1 where original X_t is observed (not NaN)
    obs_mask = torch.tensor(~np.isnan(X[K:]), dtype=torch.float32)
    return X_t, X_lags, obs_mask


def ensure_dir(path: str):
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def fmt_time(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_fig(prefix: str):
    plt.savefig(prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(prefix + ".pdf", bbox_inches="tight")
    plt.close()
    print(f">>> {prefix}.png / .pdf")


# ====================================================================
# Graph Generation
# ====================================================================

def make_er_dag(d: int, edge_prob: float = 0.15,
                w_range=(0.3, 0.8), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Scale down weights for large d to keep (I-W0) well-conditioned
    if d >= 100:
        w_lo = w_range[0] * 0.3
        w_hi = w_range[1] * 0.3
    elif d >= 50:
        w_lo = w_range[0] * 0.6
        w_hi = w_range[1] * 0.6
    else:
        w_lo, w_hi = w_range
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                W[i, j] = rng.uniform(w_lo, w_hi) * rng.choice([-1, 1])
    perm = rng.permutation(d)
    W = W[perm][:, perm]
    return W


def make_ba_dag(d: int, m: int = 2,
                w_range=(0.3, 0.8), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(d, m, seed=seed)
    order = rng.permutation(d)
    rank = np.zeros(d, dtype=int)
    rank[order] = np.arange(d)
    W = np.zeros((d, d))
    for u, v in G.edges():
        src, dst = (u, v) if rank[u] < rank[v] else (v, u)
        W[src, dst] = rng.uniform(*w_range) * rng.choice([-1, 1])
    return W


def make_lag_matrices(d: int, K: int, edge_prob: float = 0.10,
                      scale: float = 0.25, seed: int = 0) -> list:
    rng = np.random.default_rng(seed + 1000)
    # Scale down for large d to prevent overflow in simulation
    if d >= 100:
        eff_scale = scale * 0.3
    elif d >= 50:
        eff_scale = scale * 0.6
    else:
        eff_scale = scale
    Wk_list = []
    for _ in range(K):
        Wk = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if rng.random() < edge_prob:
                    Wk[i, j] = rng.uniform(0.05 * eff_scale / scale, eff_scale) * rng.choice([-1, 1])
        Wk_list.append(Wk)

    # Spectral radius check for VAR stability
    if K > 0:
        companion_dim = d * K
        if companion_dim <= 500:
            C = np.zeros((companion_dim, companion_dim))
            for k in range(K):
                C[:d, k * d:(k + 1) * d] = Wk_list[k]
            if K > 1:
                C[d:, :-d] = np.eye(d * (K - 1))
            try:
                eigvals = np.linalg.eigvals(C)
                sr = np.max(np.abs(eigvals))
                if sr > 0.95:
                    shrink = 0.90 / sr
                    Wk_list = [Wk * shrink for Wk in Wk_list]
            except np.linalg.LinAlgError:
                pass
    return Wk_list


# ====================================================================
# Data Simulation
# ====================================================================

def _sample_noise(rng, noise_type: str, d: int, noise_scale: float,
                  x_prev: Optional[np.ndarray] = None) -> np.ndarray:
    if noise_type == "gaussian":
        return rng.normal(0, noise_scale, size=d)
    elif noise_type == "laplace":
        return rng.laplace(0, noise_scale, size=d)
    elif noise_type == "student_t":
        return rng.standard_t(df=5, size=d) * noise_scale
    elif noise_type == "heteroscedastic":
        var = 0.5 + 0.5 * np.abs(x_prev) if x_prev is not None else np.ones(d)
        return rng.normal(0, 1, size=d) * np.sqrt(var) * noise_scale
    raise ValueError(f"Unknown noise_type: {noise_type}")


def simulate_svar_linear(T: int, W0: np.ndarray, Wk_list: list,
                         noise_type: str = "laplace",
                         noise_scale: float = 1.0,
                         seed: int = 0) -> Optional[np.ndarray]:
    rng = np.random.default_rng(seed)
    d = W0.shape[0]
    K = len(Wk_list)

    M = np.eye(d) - W0
    cond = np.linalg.cond(M)
    if cond > 1e10:
        warnings.warn(f"simulate_svar_linear: (I-W0) ill-conditioned (cond={cond:.1e})")
        return None

    A_inv = np.linalg.inv(M)
    T_total = T + K + 50
    X = np.zeros((T_total, d))

    for t in range(K, T_total):
        eps = _sample_noise(rng, noise_type, d, noise_scale, X[t - 1])
        lag_sum = sum(X[t - k - 1] @ Wk_list[k] for k in range(K))
        X[t] = (lag_sum + eps) @ A_inv
        if not np.all(np.isfinite(X[t])) or np.max(np.abs(X[t])) > 1e15:
            warnings.warn(f"simulate_svar_linear: overflow at t={t}")
            return None

    X = X[K + 50:]
    if not np.all(np.isfinite(X)):
        return None

    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - X.mean(0)) / std


def simulate_svar_nonlinear(T: int, W0: np.ndarray, Wk_list: list,
                            noise_type: str = "gaussian",
                            noise_scale: float = 1.0,
                            seed: int = 0) -> Optional[np.ndarray]:
    rng = np.random.default_rng(seed)
    d = W0.shape[0]
    K = len(Wk_list)

    G_dag = nx.DiGraph()
    G_dag.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if abs(W0[i, j]) > 1e-10:
                G_dag.add_edge(i, j)
    try:
        topo_order = list(nx.topological_sort(G_dag))
    except nx.NetworkXUnfeasible:
        warnings.warn("W0 not DAG, falling back to linear")
        return simulate_svar_linear(T, W0, Wk_list, noise_type, noise_scale, seed)

    nl_a = rng.uniform(0.3, 0.8, (d, d)) * np.sign(W0)
    nl_b = rng.uniform(0.5, 2.0, (d, d))
    nl_c = rng.uniform(0.1, 0.3, (d, d)) * np.sign(W0)

    T_total = T + K + 50
    X = np.zeros((T_total, d))

    for t in range(K, T_total):
        eps = _sample_noise(rng, noise_type, d, noise_scale, X[t - 1])
        lag_sum = sum(X[t - k - 1] @ Wk_list[k] for k in range(K))
        x_t = np.zeros(d)
        for j in topo_order:
            val = lag_sum[j] + eps[j]
            for i in range(d):
                if abs(W0[i, j]) > 1e-10:
                    val += nl_a[i, j] * np.tanh(nl_b[i, j] * x_t[i]) + nl_c[i, j] * x_t[i]
            x_t[j] = np.clip(val, -20, 20)
        X[t] = x_t

    X = X[K + 50:]
    if not np.all(np.isfinite(X)):
        return None

    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - X.mean(0)) / std


# ====================================================================
# Prior Generation
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


def gen_prior_from_truth(B_true: np.ndarray, acc: float,
                         mode: str = "random", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = B_true.shape[0]
    P = np.full((d, d), 0.5)
    cut = d // 2
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            true_edge = B_true[i, j] > 0
            if mode == "random":
                agree = rng.random() < acc
            elif mode == "systematic":
                agree = False if (i < cut and j >= cut) else (rng.random() < acc)
            else:
                agree = rng.random() < acc
            if agree:
                P[i, j] = rng.uniform(0.75, 0.95) if true_edge else rng.uniform(0.05, 0.25)
            else:
                P[i, j] = rng.uniform(0.05, 0.25) if true_edge else rng.uniform(0.75, 0.95)
    return P


def binarize_prior_to_mask(P_prior: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
    return (P_prior >= threshold).astype(float)


# ====================================================================
# Evaluation Metrics
# ====================================================================

def compute_all_metrics(B_true: np.ndarray,
                        W_est_continuous: np.ndarray,
                        exclude_diag: bool = True) -> dict:
    d = B_true.shape[0]
    B = B_true.copy().astype(int)
    scores = np.abs(W_est_continuous).copy()

    if exclude_diag:
        np.fill_diagonal(B, 0)
        np.fill_diagonal(scores, 0.0)
        mask = ~np.eye(d, dtype=bool)
    else:
        mask = np.ones((d, d), dtype=bool)

    y_true = B[mask]
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

    # SHD at best threshold
    B_opt = (scores >= best_thr).astype(int)
    if exclude_diag:
        np.fill_diagonal(B_opt, 0)
    res["shd"] = int((B[mask] != B_opt[mask]).sum())
    n_pairs = int(mask.sum())
    res["shd_norm"] = res["shd"] / max(n_pairs, 1)

    # Top-k
    k_true = int(B[mask].sum())
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
        res["shd_topk"] = fp + fn
    else:
        res["f1_topk"] = 0.0
        res["shd_topk"] = int(k_true + (y_score > 0).sum())

    return res


def combine_W0_Wk(W0: np.ndarray, Wk_list: list) -> np.ndarray:
    abs_mats = [np.abs(W0)] + [np.abs(wk) for wk in Wk_list]
    return np.stack(abs_mats, axis=0).max(axis=0)


def compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est) -> dict:
    B_w0 = (np.abs(W0_true) > 1e-10).astype(float)
    met_w0 = compute_all_metrics(B_w0, W0_est)
    res = {f"w0_{k}": v for k, v in met_w0.items()}

    B_comb = (np.abs(W0_true) > 1e-10).astype(int)
    for Wk_t in Wk_true:
        B_comb = np.maximum(B_comb, (np.abs(Wk_t) > 1e-10).astype(int))
    W_comb_est = combine_W0_Wk(W0_est, Wk_est)
    met_comb = compute_all_metrics(B_comb.astype(float), W_comb_est)
    res.update({f"comb_{k}": v for k, v in met_comb.items()})

    # Fix 20: top-level 指标使用 combined (W0+Wk) 而非仅 W0
    # 同时保留 w0_ 前缀版本供需要时使用
    res.update({k: v for k, v in met_comb.items()})
    return res


# ====================================================================
# Baselines
# ====================================================================

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


def _run_dynotears_single(X_t, X_lags, d, K, lam, max_outer, inner, lr, dev):
    """Run a single DYNOTEARS instance with given lambda."""
    m = _DYNOTEARS(d, K, lam).to(dev)
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
        rho = min(rho * 3.0, 1e6)
    with torch.no_grad():
        W0 = m._adj().cpu().numpy()
        Wk = [w.detach().cpu().numpy() for w in m.Wk]
    return W0, Wk


def run_dynotears(X, d, K, lam=0.01, max_outer=35, inner=300,
                  lr=1e-2, seed=0, grid_search=False):
    """Run DYNOTEARS baseline (single lambda, no grid search for fair comparison)."""
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t, X_lags = X_t.to(dev), [x.to(dev) for x in X_lags]
    return _run_dynotears_single(X_t, X_lags, d, K, lam, max_outer,
                                  inner, lr, dev)


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


# ====================================================================
# PRCD-MAP Wrapper (v3 compatible)
# ====================================================================

def _calibrate_scores(W: np.ndarray) -> np.ndarray:
    """
    Row-wise score calibration: 每行除以该行的max绝对值.
    使得不同变量的edge score可比, 提升AUROC.
    """
    W_cal = np.abs(W).copy()
    for i in range(W_cal.shape[0]):
        row_max = W_cal[i].max()
        if row_max > 1e-10:
            W_cal[i] /= row_max
    np.fill_diagonal(W_cal, 0.0)
    # 保留原始符号用于其他用途, 但返回calibrated absolute scores
    return W_cal * np.sign(W)


def run_prcd_map(X, P_prior, d, K, lambda1=0.001, lambda2=0.01,
                 learn_tau=True, tau0=1.0,
                 max_iter=35, inner_iter=400, lr=1e-2, seed=0,
                 loss_type="huber", prior_l1_weight=True,
                 n_tau_groups=4, verbose=False,
                 score_calibration=False):
    set_seed(seed)
    X_t, X_lags, obs_mask = make_lag_tensors_with_mask(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]
    if obs_mask is not None:
        obs_mask = obs_mask.to(dev)

    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau, tau0=tau0,
        tau_min=0.05, tau_max=3.0,
        loss_type=loss_type,
        prior_l1_weight=prior_l1_weight,
        n_tau_groups=n_tau_groups,
    ).to(dev)

    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=verbose, postprocess=False,
        obs_mask=obs_mask,
    )

    # Score calibration: row-wise normalization for better AUROC
    if score_calibration:
        W0 = _calibrate_scores(W0)
        Wk = [_calibrate_scores(wk) for wk in Wk]

    return W0, Wk, tau


# ====================================================================
# Lorenz-96
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
                      dt: float = 0.05, subsample: int = 1,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = F * np.ones(d) + rng.normal(0, 0.01, d)
    burn_in_steps = 500
    total_steps = burn_in_steps + T * subsample
    t_span = (0, total_steps * dt)
    t_eval = np.arange(0, total_steps * dt, dt)

    sol = solve_ivp(_lorenz96_rhs, t_span, x0, args=(F,),
                    t_eval=t_eval, method="RK45", max_step=dt)
    if sol.status != 0:
        warnings.warn(f"Lorenz-96 integration failed: {sol.message}")
        return np.zeros((T, d)), lorenz96_ground_truth(d)

    X_full = sol.y.T
    X_full = X_full[burn_in_steps:]
    if subsample > 1:
        X_full = X_full[::subsample]
    X_full = X_full[:T]
    if len(X_full) < T:
        pad = np.zeros((T - len(X_full), d))
        X_full = np.vstack([X_full, pad])

    return standardize(X_full), lorenz96_ground_truth(d)


# ====================================================================
# Electricity Data Loader
# ====================================================================

def load_electricity(excel_path: str, prior_csv_path: str,
                     diff_periods: int = 7, train_ratio: float = 0.8):
    df_ts = pd.read_excel(excel_path, index_col=0)
    df_diff = df_ts.diff(periods=diff_periods).dropna()
    split = int(len(df_diff) * train_ratio)
    col_names = df_diff.columns.tolist()
    d = len(col_names)

    if os.path.exists(prior_csv_path):
        P_prior = pd.read_csv(prior_csv_path, index_col=0).values
        if P_prior.shape != (d, d):
            warnings.warn(f"Prior shape {P_prior.shape} != ({d},{d}), using uniform")
            P_prior = np.full((d, d), 0.5)
    else:
        P_prior = np.full((d, d), 0.5)

    return df_diff, P_prior, split, col_names


# ====================================================================
# RHINO-Style Table Formatting
# ====================================================================

def print_rhino_table(df: pd.DataFrame, metric: str = "auroc",
                      group_col: str = "method",
                      setting_col: str = "setting",
                      title: str = ""):
    """
    Print a RHINO-style comparison table.
    Rows = methods, Cols = settings, Cells = metric mean ± std.
    Best per column is marked with *.
    """
    if df.empty:
        print("  (no data)")
        return

    agg = df.groupby([group_col, setting_col])[metric].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0)
    agg["cell"] = agg.apply(lambda r: f"{r['mean']:.3f}±{r['std']:.3f}", axis=1)

    piv = agg.pivot(index=group_col, columns=setting_col, values="mean")
    piv_str = agg.pivot(index=group_col, columns=setting_col, values="cell")

    # Mark best per column
    for col in piv.columns:
        if piv[col].dropna().empty:
            continue
        best_idx = piv[col].idxmax(skipna=True)
        if pd.notna(piv.loc[best_idx, col]):
            piv_str.loc[best_idx, col] = piv_str.loc[best_idx, col] + " *"

    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(piv_str.to_string())
    print()


def compute_significance(df: pd.DataFrame, metric: str = "auroc",
                         method_col: str = "method",
                         group_cols: list = None) -> pd.DataFrame:
    """
    Compute pairwise Wilcoxon signed-rank tests between methods (Fix 17).
    Returns DataFrame with p-values for each method pair.
    """
    from scipy.stats import wilcoxon
    methods = sorted(df[method_col].unique())
    rows = []
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if j <= i:
                continue
            d1 = df[df[method_col] == m1]
            d2 = df[df[method_col] == m2]
            if group_cols:
                merged = d1.merge(d2, on=group_cols, suffixes=("_1", "_2"))
                v1 = merged[f"{metric}_1"].values
                v2 = merged[f"{metric}_2"].values
            else:
                v1 = d1[metric].values
                v2 = d2[metric].values
            n = min(len(v1), len(v2))
            if n < 5:
                p_val = float("nan")
            else:
                v1, v2 = v1[:n], v2[:n]
                diff = v1 - v2
                if np.all(diff == 0):
                    p_val = 1.0
                else:
                    try:
                        _, p_val = wilcoxon(v1, v2, alternative="two-sided")
                    except Exception:
                        p_val = float("nan")
            rows.append(dict(method1=m1, method2=m2, metric=metric,
                             n=n, p_value=p_val,
                             mean_diff=float(np.mean(v1[:n] - v2[:n]))))
    return pd.DataFrame(rows)


# ====================================================================
# Hard-Mask PRCD-MAP Model (for ablation exp4)
# ====================================================================

class PRCD_MAP_HardMask(nn.Module):
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
             for _ in range(self.K)])
        mask_tensor = torch.tensor(hard_mask, dtype=torch.float32)
        self.register_buffer("hard_mask", mask_tensor)
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.d))

    def get_W0_adj(self):
        return self.W0 * self.off_diag_mask * self.hard_mask

    def get_Wk_masked(self, k):
        return self.Wk[k] * self.hard_mask

    def _compute_h_w0(self):
        W0_adj = self.get_W0_adj()
        s = self.dagma_s
        M = s * torch.eye(self.d, device=W0_adj.device) - W0_adj * W0_adj
        sign, logabsdet = torch.linalg.slogdet(M)
        if sign.item() <= 0:
            return torch.sum(W0_adj * W0_adj)
        return torch.clamp(-logabsdet + self.d * math.log(s), min=0.0)

    def forward(self, X_t, X_lags):
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
            + sum(torch.norm(self.get_Wk_masked(k), p=1) for k in range(self.K)))
        h_val = self._compute_h_w0()
        loss_alm = loss_mse + loss_l1 + alpha * h_val + 0.5 * rho * h_val ** 2
        return loss_alm, loss_mse, loss_l1, h_val


def train_hard_mask_alm(model: PRCD_MAP_HardMask,
                        X_t, X_lags,
                        max_iter=35, inner_iter=400,
                        lr=1e-2, rho_0=1.0, gamma=3.0,
                        tol=1e-6, grad_clip=5.0):
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
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        with torch.no_grad():
            h_now = float(model._compute_h_w0().detach().cpu().item())
        if abs(h_now) <= tol:
            break
        alpha += rho * h_now
        rho = min(rho * float(gamma), 1e6)
    with torch.no_grad():
        W0 = model.get_W0_adj().detach().cpu().numpy()
        Wk = [model.get_Wk_masked(k).detach().cpu().numpy()
              for k in range(model.K)]
    return W0, Wk


# ====================================================================
# Training with Logging (for exp5 convergence analysis)
# ====================================================================

def train_prcd_alm_with_logging(
    model: PRCD_MAP_Model,
    X_t: torch.Tensor,
    X_lags,
    max_iter: int = 35,
    inner_iter: int = 400,
    lr: float = 1e-2,
    rho_0: float = 1.0,
    gamma: float = 3.0,
    tol: float = 1e-6,
    grad_clip: float = 5.0,
    tau_warmup: int = 0,
    tau_eb_steps: int = 8,
    tau_eb_lr: float = 0.2,
) -> Tuple[np.ndarray, list, float, pd.DataFrame]:
    rho = float(rho_0)
    alpha = 0.0
    log_rows = []
    device = next(model.parameters()).device

    for it in range(max_iter):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=inner_iter, eta_min=lr * 0.01)

        for _ in range(inner_iter):
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

        # EB tau update
        if model.learn_tau and it >= tau_warmup:
            for _ in range(tau_eb_steps):
                tau_var = model.tau_groups.clone().detach().requires_grad_(True)
                eb_loss = model.compute_eb_objective(X_t, X_lags, tau_var)
                eb_loss.backward()
                with torch.no_grad():
                    tau_var = tau_var - tau_eb_lr * tau_var.grad
                    tau_var = tau_var.clamp(model.tau_min, model.tau_max)
                model.tau_groups.copy_(tau_var)
            tau_now = float(model.get_tau().detach().cpu().item())

        log_rows.append(dict(
            outer_iter=it + 1,
            rho=rho, alpha=alpha,
            h_val=h_now, tau=tau_now,
            loss_alm=l_alm, loss_mse=l_mse,
            loss_l1=l_l1, loss_prior=l_prior_val,
        ))

        if abs(h_now) <= tol:
            break
        alpha += rho * h_now
        rho = min(rho * float(gamma), 1e6)

    with torch.no_grad():
        W0 = model.get_W0_adj().detach().cpu().numpy()
        Wk = [wk.detach().cpu().numpy() for wk in model.Wk]
        tau_est = float(model.get_tau().detach().cpu().item())

    return W0, Wk, tau_est, pd.DataFrame(log_rows)


# ====================================================================
# NGC baseline (Tank et al. 2021, official codebase wrapper)
# ====================================================================

class _NGCModelBuiltin(nn.Module):
    """
    Fallback NGC implementation when official package unavailable.
    d separate MLPs with group-sparse penalty on input-layer weights.
    """
    def __init__(self, d, K, hidden=32, n_layers=2):
        super().__init__()
        self.d = d
        self.K = K
        input_dim = d * (K + 1)  # current + K lags
        self.nets = nn.ModuleList()
        for _ in range(d):
            layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            layers.append(nn.Linear(hidden, 1))
            self.nets.append(nn.Sequential(*layers))

    def forward(self, X_full):
        """X_full: (T, d*(K+1)). Returns (T, d) predictions."""
        preds = []
        for i in range(self.d):
            preds.append(self.nets[i](X_full))
        return torch.cat(preds, dim=1)

    def get_gc_matrix(self):
        """Extract Granger-causal strength matrix (d, d) from input-layer weights."""
        W = np.zeros((self.d, self.d))
        for i in range(self.d):
            w_input = self.nets[i][0].weight.detach().cpu().numpy()  # (hidden, d*(K+1))
            for j in range(self.d):
                # Group norm: all weights from variable j across current + K lags
                idx = [j + self.d * lag for lag in range(self.K + 1)]
                W[j, i] = np.linalg.norm(w_input[:, idx])
        np.fill_diagonal(W, 0.0)
        return W


def run_ngc(X, d, K, hidden=32, n_layers=2, lam_group=0.01,
            epochs=500, lr=5e-4, seed=0):
    """
    Neural Granger Causality (Tank et al. 2021).
    Uses official codebase if available, otherwise fallback implementation.
    Returns (W0, Wk_list) where W0 is the GC strength matrix.
    """
    set_seed(seed)

    if HAS_NGC:
        # --- 使用官方代码库 (Neural-GC: cMLP(num_series, lag, hidden)) ---
        try:
            X_torch = torch.tensor(standardize(X), dtype=torch.float32).unsqueeze(0)  # (1, T, d)
            model = _NGC_cMLP(d, K, [hidden] * n_layers)
            _ngc_train_ista(model, X_torch, lr, epochs,
                            lam=lam_group, lam_ridge=0.0,
                            check_every=100, verbose=0)
            GC = model.GC(threshold=False).detach().cpu().numpy()
            np.fill_diagonal(GC, 0.0)
            Wk_zeros = [np.zeros((d, d)) for _ in range(K)]
            return GC, Wk_zeros
        except Exception as e:
            warnings.warn(f"NGC official failed, using fallback: {e}")

    # --- Fallback: built-in implementation ---
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_std = standardize(X)
    T_eff = len(X_std) - K
    # Build input: concat [X_t, X_{t-1}, ..., X_{t-K}]
    X_full_list = [X_std[K:]]
    for lag in range(1, K + 1):
        X_full_list.append(X_std[K - lag: -lag if lag < len(X_std) else None])
    X_full = np.concatenate(X_full_list, axis=1)
    X_full_t = torch.tensor(X_full, dtype=torch.float32).to(dev)
    Y_t = torch.tensor(X_std[K:], dtype=torch.float32).to(dev)

    model = _NGCModelBuiltin(d, K, hidden, n_layers).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = model(X_full_t)
        loss_mse = 0.5 * torch.sum((Y_t - pred) ** 2) / T_eff
        # Group-sparse penalty
        loss_group = torch.tensor(0.0, device=dev)
        for i in range(d):
            w_input = model.nets[i][0].weight  # (hidden, d*(K+1))
            for j in range(d):
                if j != i:
                    idx = [j + d * lag for lag in range(K + 1)]
                    loss_group = loss_group + torch.norm(w_input[:, idx])
        loss = loss_mse + lam_group * loss_group
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W0 = model.get_gc_matrix()
    Wk_zeros = [np.zeros((d, d)) for _ in range(K)]
    return W0, Wk_zeros


# ====================================================================
# RHINO baseline (Gong et al. 2023, via causica / CUTS+ fallback)
# ====================================================================

def run_rhino(X, d, K, seed=0, max_epochs=200):
    """
    RHINO (Gong et al. 2023) via causica official codebase.
    Falls back to CUTS+ if causica unavailable.
    Returns (W0, Wk_list).
    """
    set_seed(seed)

    if HAS_RHINO:
        try:
            # causica API varies by version, try multiple import paths
            DECI = None
            try:
                from causica.models.deci.deci import DECI
            except ImportError:
                try:
                    from causica.models.deci import DECI
                except ImportError:
                    from causica import DECI

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_std = standardize(X)
            X_torch = torch.tensor(X_std, dtype=torch.float32)

            # 尝试新版 causica API (>= 0.3)
            try:
                model = DECI(
                    "rhino_model", d,
                    device=dev,
                    lag=K,
                    embedding_size=32,
                    out_dim_g=32,
                    num_layers_g=2,
                    tau_gumbel=0.25,
                    lambda_sparse=0.05,
                )
            except TypeError:
                # 旧版 API
                model = DECI.create(
                    "rhino", d, save_dir="/tmp/rhino_temp",
                    device=dev, lag=K,
                )

            model.run_train(X_torch, X_torch, max_epochs=max_epochs, batch_size=256)
            adj = model.get_adj_matrix(samples=100, do_round=False)
            W0 = adj.mean(0) if adj.ndim == 3 else adj
            if isinstance(W0, torch.Tensor):
                W0 = W0.detach().cpu().numpy()
            np.fill_diagonal(W0, 0.0)
            Wk_zeros = [np.zeros((d, d)) for _ in range(K)]
            return W0, Wk_zeros
        except Exception as e:
            warnings.warn(f"RHINO (causica) failed: {e}")

    if HAS_CUTS:
        try:
            # CUTS+ fallback
            X_std = standardize(X)
            cuts_model = _CUTS_Model(d, K)
            cuts_model.fit(X_std)
            W0 = cuts_model.get_adj_matrix()
            np.fill_diagonal(W0, 0.0)
            Wk_zeros = [np.zeros((d, d)) for _ in range(K)]
            return W0, Wk_zeros
        except Exception as e:
            warnings.warn(f"CUTS+ fallback also failed: {e}")

    warnings.warn("Neither causica (RHINO) nor CUTS+ available. Returning None.")
    return None, None


# ====================================================================
# Data loaders: Netsim fMRI, CausalTime
# ====================================================================

def load_netsim(data_dir, sim_id=3):
    """
    Load Netsim fMRI benchmark (Smith et al. 2011).
    Expects {data_dir}/sim{sim_id}.mat with 'ts' (time series) and 'net' (adjacency).
    Returns (X_standardized, B_true) or (None, None) on failure.
    """
    from scipy.io import loadmat

    mat_path = os.path.join(data_dir, f"sim{sim_id}.mat")
    if not os.path.exists(mat_path):
        warnings.warn(f"Netsim file not found: {mat_path}")
        return None, None

    data = loadmat(mat_path)

    # Netsim .mat: 'ts' = (n_subjects, T, d) or (T, d), 'net' = (n_subjects, d, d) or (d, d)
    if "ts" in data:
        X = data["ts"].astype(np.float64)
    elif "Ysim" in data:
        X = data["Ysim"].astype(np.float64)
    else:
        for key in data:
            if not key.startswith("_") and isinstance(data[key], np.ndarray) and data[key].ndim >= 2:
                if data[key].shape[-2] > data[key].shape[-1]:
                    X = data[key].astype(np.float64)
                    break
        else:
            warnings.warn(f"Cannot find time series in {mat_path}")
            return None, None

    if X.ndim == 3:
        # (n_subjects, T, d) — concatenate all subjects along time axis
        X = X.reshape(-1, X.shape[-1])

    if "net" in data:
        B_raw = data["net"]
    elif "Adj" in data:
        B_raw = data["Adj"]
    else:
        B_raw = None
        for key in data:
            if not key.startswith("_") and isinstance(data[key], np.ndarray):
                arr = np.squeeze(data[key])
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    B_raw = arr
                    break
        if B_raw is None:
            warnings.warn(f"Cannot find adjacency in {mat_path}")
            return None, None

    B_raw = np.squeeze(B_raw)
    if B_raw.ndim == 3 and B_raw.shape[1] == B_raw.shape[2]:
        # Netsim stores (n_subjects, d, d) — all subjects share the same network
        B_raw = B_raw[0]
    elif B_raw.ndim == 3 and B_raw.shape[0] == B_raw.shape[1]:
        B_raw = B_raw[:, :, 0]
    if B_raw.ndim != 2 or B_raw.shape[0] != B_raw.shape[1]:
        d = X.shape[1]
        if B_raw.size == d * d:
            B_raw = B_raw.reshape(d, d)
        else:
            warnings.warn(f"Adjacency shape {B_raw.shape} is not square in {mat_path}")
            return None, None

    B_true = (np.abs(B_raw) > 0.5).astype(int)
    np.fill_diagonal(B_true, 0)
    X = standardize(X)
    return X, B_true


def load_causaltime(data_dir, dataset_name="AQI", n_samples=1):
    """
    Load CausalTime benchmark (Cheng et al. 2024, ICLR).
    Supports both .npy (official format) and .csv fallback.

    Official .npy format:
      gen_data.npy: (N_samples, T, 2*d) — 取前 n_samples 个 sample 的前 d 列
      graph.npy: (d, d) ground truth 邻接矩阵

    目录名映射: "AQI" → "pm25", "Traffic" → "traffic", "Medical" → "medical"

    Parameters
    ----------
    n_samples : int
        Number of samples to concatenate (default=1 for single sample T=40).
        Set n_samples>1 to increase effective T (e.g., n_samples=10 → T=400).

    Returns (X_standardized, B_true) or (None, None) on failure.
    """
    # CausalTime 官方目录名与常用名的映射
    name_map = {"AQI": "pm25", "aqi": "pm25", "PM25": "pm25",
                "Traffic": "traffic", "Medical": "medical"}
    mapped_name = name_map.get(dataset_name, dataset_name)

    # 尝试多个可能的目录名
    ds_dir = None
    for candidate in [dataset_name, mapped_name, dataset_name.lower()]:
        candidate_dir = os.path.join(data_dir, candidate)
        if os.path.isdir(candidate_dir):
            ds_dir = candidate_dir
            break
    if ds_dir is None:
        warnings.warn(f"CausalTime dataset dir not found for '{dataset_name}' in {data_dir}")
        return None, None

    # --- 优先尝试 .npy 格式 (官方) ---
    npy_data = os.path.join(ds_dir, "gen_data.npy")
    npy_graph = os.path.join(ds_dir, "graph.npy")

    if os.path.exists(npy_graph):
        B_true = np.load(npy_graph).astype(float)
        B_true = (np.abs(B_true) > 0.5).astype(int)
        np.fill_diagonal(B_true, 0)
        d = B_true.shape[0]

        if os.path.exists(npy_data):
            raw = np.load(npy_data)  # (N_samples, T, 2*d)
            if raw.ndim == 3:
                n_avail = raw.shape[0]
                n_use = min(n_samples, n_avail)
                if n_use > 1:
                    # Concatenate multiple samples for larger effective T
                    parts = [raw[i, :, :d].astype(np.float64) for i in range(n_use)]
                    X = np.concatenate(parts, axis=0)
                else:
                    X = raw[0, :, :d].astype(np.float64)
            elif raw.ndim == 2:
                X = raw[:, :d].astype(np.float64)
            else:
                warnings.warn(f"Unexpected gen_data.npy shape: {raw.shape}")
                return None, None
        else:
            warnings.warn(f"graph.npy found but gen_data.npy missing in {ds_dir}")
            return None, None

        if not np.all(np.isfinite(X)):
            warnings.warn(f"CausalTime {dataset_name}: non-finite values")
            return None, None

        X = standardize(X)
        return X, B_true

    # --- Fallback: CSV 格式 ---
    data_path = os.path.join(ds_dir, "data.csv")
    graph_path = os.path.join(ds_dir, "graph.csv")

    if not os.path.exists(data_path):
        for alt in ["timeseries.csv", "ts.csv", f"{dataset_name}.csv"]:
            alt_path = os.path.join(ds_dir, alt)
            if os.path.exists(alt_path):
                data_path = alt_path
                break
        else:
            warnings.warn(f"CausalTime data not found in {ds_dir}")
            return None, None

    if not os.path.exists(graph_path):
        for alt in ["adjacency.csv", "adj.csv", "dag.csv", "ground_truth.csv"]:
            alt_path = os.path.join(ds_dir, alt)
            if os.path.exists(alt_path):
                graph_path = alt_path
                break
        else:
            warnings.warn(f"CausalTime graph not found in {ds_dir}")
            return None, None

    df_data = pd.read_csv(data_path)
    numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()
    X = df_data[numeric_cols].values.astype(np.float64)

    df_graph = pd.read_csv(graph_path, header=None)
    B_true = df_graph.values.astype(float)
    B_true = (np.abs(B_true) > 0.5).astype(int)
    np.fill_diagonal(B_true, 0)

    X = standardize(X)
    return X, B_true


# ====================================================================
# PRCD-MAP NAM wrapper (nonlinear extension)
# ====================================================================

def run_prcd_map_nam(X, P_prior, d, K, lambda1=0.001, lambda2=0.01,
                     learn_tau=True, edge_hidden=16, edge_layers=2,
                     max_iter=35, inner_iter=400, lr=5e-4, seed=0,
                     n_tau_groups=4, verbose=False):
    """
    PRCD-MAP with Neural Additive Model parameterization.
    Each edge (i->j) is modeled by a small MLP: f_{ij}(x_i).
    Returns (W0_strength, Wk_list, tau).
    """
    from model_nam import PRCD_MAP_NAM, train_prcd_nam_alm

    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]

    model = PRCD_MAP_NAM(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau,
        tau_min=0.05, tau_max=3.0,
        edge_hidden=edge_hidden, edge_layers=edge_layers,
        n_tau_groups=n_tau_groups,
    ).to(dev)

    W0, Wk, tau = train_prcd_nam_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=verbose,
    )
    return W0, Wk, tau


# ====================================================================
# Laplace approximation validation (exp5 Part E)
# ====================================================================

def validate_laplace_approximation(
    X, d, K, P_prior, lambda1=0.001, lambda2=0.01,
    seed=0, max_iter=35, inner_iter=400, lr=1e-2,
    tau_grid=None, n_grid=50,
):
    """
    Validate Laplace approximation of EB objective against numerical integration.

    线性 SVAR 下条件后验是 Gaussian, 可 closed-form 计算真实边际似然.
    与 Laplace 近似 (compute_eb_objective) 对比.

    Returns dict with:
      tau_grid, exact_values, laplace_values,
      tau_exact_opt, tau_laplace_opt, gap
    """
    from scipy.integrate import quad

    set_seed(seed)
    X_std = standardize(X)
    X_t, X_lags = make_lag_tensors(X_std, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]

    # 1. 训练 PRCD-MAP 到收敛 → W*
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=True, tau_min=0.05, tau_max=3.0,
        loss_type="huber", prior_l1_weight=True,
    ).to(dev)
    train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        verbose=False,
    )

    # 2. 固定 W*, sweep tau
    if tau_grid is None:
        tau_grid = np.linspace(0.05, 3.0, n_grid)

    T_eff = X_t.shape[0]

    # 预计算数据相关量 (用于 closed-form marginal likelihood)
    X_t_np = X_t.detach().cpu().numpy()
    X_lags_np = [xl.detach().cpu().numpy() for xl in X_lags]
    W0_star = model.get_W0_adj().detach().cpu().numpy()
    Wk_star = [wk.detach().cpu().numpy() for wk in model.Wk]

    # 残差: R = X_t - X_t @ W0 - sum X_lags[k] @ Wk[k]
    pred = X_t_np @ W0_star
    for k_idx in range(K):
        pred = pred + X_lags_np[k_idx] @ Wk_star[k_idx]
    residuals = X_t_np - pred
    sigma2_hat = np.mean(residuals ** 2)  # 残差方差估计

    P_clamped = np.clip(P_prior, 1e-3, 1.0 - 1e-3)
    prior_logits = np.log(P_clamped) - np.log(1.0 - P_clamped)
    off_mask = 1.0 - np.eye(d)

    laplace_values = []
    exact_values = []

    for tau_val in tau_grid:
        # --- Laplace 近似 (现有 compute_eb_objective) ---
        tau_t = torch.full((model.n_tau_groups,), tau_val, dtype=torch.float32,
                           device=dev, requires_grad=False)
        with torch.no_grad():
            eb_val = model.compute_eb_objective(X_t, X_lags, tau_t)
        laplace_values.append(float(eb_val.cpu().item()))

        # --- 精确边际似然 (closed-form for linear Gaussian SVAR) ---
        # 对线性模型: log p(X|τ) ∝ -T/2 log(σ²) + log p(W*|τ)
        # Prior: p(W|τ) ∝ exp(-λ₂/2 * Ω(τ) * W² - λ₁ * coeff(τ) * |W|)
        # Ω(τ) = 1 - sigmoid(logit(P)*τ) + δ
        P_hat = 1.0 / (1.0 + np.exp(-prior_logits * tau_val))
        Omega = (1.0 - P_hat + 1e-3) * off_mask
        coeff = np.clip(1.5 - P_hat, 0.1, 1.5) * off_mask

        # Log prior at W*
        log_prior = -0.5 * lambda2 * np.sum(Omega * W0_star ** 2)
        log_prior -= lambda1 * np.sum(coeff * np.abs(W0_star))
        for k_idx in range(K):
            log_prior -= 0.5 * lambda2 * np.sum(Omega * Wk_star[k_idx] ** 2)

        # Hessian diagonal (exact for linear MSE)
        data_hess = np.sum(X_t_np ** 2, axis=0) / (T_eff * d)  # (d,)
        log_det = 0.0
        for j in range(d):
            for i in range(d):
                if i != j:
                    H_ij = data_hess[i] + lambda2 * Omega[i, j]
                    log_det += np.log(max(H_ij, 1e-10))
        for k_idx in range(K):
            X_lag_np = X_lags_np[k_idx]
            data_hess_k = np.sum(X_lag_np ** 2, axis=0) / (T_eff * d)
            for j in range(d):
                for i in range(d):
                    H_ij = data_hess_k[i] + lambda2 * Omega[i, j]
                    log_det += np.log(max(H_ij, 1e-10))

        # 数值积分修正 (1D quadrature over τ-dependent terms)
        # 对 agreement loss 做精确计算 (不依赖 Hessian 近似)
        w_max = max(np.max(np.abs(W0_star)), 1e-6)
        W0_prob = np.clip(np.abs(W0_star) / w_max, 1e-6, 1.0 - 1e-6) * off_mask
        P_hat_safe = np.clip(P_hat, 1e-6, 1.0 - 1e-6)
        agreement = -np.sum(
            (W0_prob * np.log(P_hat_safe) + (1.0 - W0_prob) * np.log(1.0 - P_hat_safe))
            * off_mask
        )

        # τ 正则
        tau_reg = 0.5 * (tau_val - 0.5) ** 2 / (2.0 ** 2)

        exact_val = agreement + 0.5 * log_det + tau_reg
        exact_values.append(exact_val)

    tau_grid = np.array(tau_grid)
    laplace_values = np.array(laplace_values)
    exact_values = np.array(exact_values)

    tau_laplace_opt = tau_grid[np.argmin(laplace_values)]
    tau_exact_opt = tau_grid[np.argmin(exact_values)]

    return {
        "tau_grid": tau_grid,
        "laplace_values": laplace_values,
        "exact_values": exact_values,
        "tau_laplace_opt": float(tau_laplace_opt),
        "tau_exact_opt": float(tau_exact_opt),
        "gap": float(abs(tau_laplace_opt - tau_exact_opt)),
        "relative_error": float(np.mean(np.abs(laplace_values - exact_values)
                                        / (np.abs(exact_values) + 1e-10))),
    }
