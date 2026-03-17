"""
=============================================================================
Experiment 1 — Comprehensive Synthetic Benchmark for PRCD-MAP
=============================================================================
NeurIPS-grade synthetic evaluation covering:
  - Graph structures: Erdos-Renyi (ER) and Scale-Free (Barabasi-Albert)
  - Noise types: Gaussian, Laplace, Student-t(df=5), Heteroscedastic
  - Nonlinear instantaneous effects (tanh + linear mixing)
  - Prior corruption modes: Random, Systematic, Adversarial
  - Baselines: DYNOTEARS (from scratch), PCMCI+, VARLiNGAM
  - PRCD-MAP variants: learn_tau, fixed_tau, uniform prior
  - Metrics: AUROC, AUPRC, Best-F1, Directed SHD, Normalized SHD, Top-k F1

Usage:
  python exp1_full_benchmark.py                  # default (moderate scope)
  python exp1_full_benchmark.py --quick          # tiny test run
  python exp1_full_benchmark.py --full           # full NeurIPS sweep
  python exp1_full_benchmark.py --sub noise      # noise-robustness sub-exp
  python exp1_full_benchmark.py --sub graph      # graph-structure sub-exp
  python exp1_full_benchmark.py --sub nonlinear  # nonlinearity sub-exp
  python exp1_full_benchmark.py --sub prior      # prior-corruption sub-exp
  python exp1_full_benchmark.py --sub scale      # scalability sub-exp
  python exp1_full_benchmark.py --dims 10 20 --Ts 500 --seeds 0 1 2
=============================================================================
"""

import os, sys, time, warnings, argparse, traceback
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

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

# ====================================================================
# 2. Graph Generation
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

def make_ba_dag(d: int, m: int = 2,
                w_range=(0.3, 0.8), seed: int = 0) -> np.ndarray:
    """Barabasi-Albert scale-free DAG."""
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
    """K stable lag coefficient matrices."""
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

# ====================================================================
# 3. Data Simulation
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
                         seed: int = 0) -> np.ndarray:
    """
    Linear SVAR:
      x_t (I - W0) = sum_k x_{t-k} Wk + eps_t
      => x_t = (sum_k x_{t-k} Wk + eps_t) (I - W0)^{-1}
    Convention: W0[i,j] means i -> j.
    """
    rng = np.random.default_rng(seed)
    d = W0.shape[0]
    K = len(Wk_list)
    A_inv = np.linalg.inv(np.eye(d) - W0)
    T_total = T + K + 50  # extra burn-in
    X = np.zeros((T_total, d))

    for t in range(K, T_total):
        eps = _sample_noise(rng, noise_type, d, noise_scale, X[t - 1])
        lag_sum = sum(X[t - k - 1] @ Wk_list[k] for k in range(K))
        X[t] = (lag_sum + eps) @ A_inv

    X = X[K + 50:]  # remove burn-in
    if not np.all(np.isfinite(X)):
        warnings.warn("simulate_svar_linear: non-finite values, returning zeros")
        return np.zeros((T, d))

    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - X.mean(0)) / std

def simulate_svar_nonlinear(T: int, W0: np.ndarray, Wk_list: list,
                            noise_type: str = "gaussian",
                            noise_scale: float = 1.0,
                            seed: int = 0) -> np.ndarray:
    """
    Nonlinear instantaneous effects:
      x_t,j = sum_{i in pa(j)} [a_ij tanh(b_ij x_t,i) + c_ij x_t,i]
               + sum_k x_{t-k} Wk_{:,j} + eps_t,j
    Processed in topological order of W0 DAG.
    """
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

    # Random nonlinear weights (fixed per (i,j))
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
        warnings.warn("simulate_svar_nonlinear: non-finite values")
        return np.zeros((T, d))

    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - X.mean(0)) / std

# ====================================================================
# 4. Prior Generation
# ====================================================================

def gen_prior(W0_true: np.ndarray, Wk_true: list,
              acc: float, mode: str = "random",
              seed: int = 0) -> np.ndarray:
    """
    Prior P in [0,1]^{d x d}.
    Modes:
      random      — each entry independently agrees with truth with prob `acc`
      systematic  — cross-block (first half -> second half) edges always wrong
      adversarial — acc < 0.5 used directly; structured negative correlation
    """
    rng = np.random.default_rng(seed)
    d = W0_true.shape[0]
    # Combined ground truth (W0 + all Wk)
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

# ====================================================================
# 5. Evaluation Metrics
# ====================================================================

def compute_all_metrics(W0_true: np.ndarray,
                        W0_est_continuous: np.ndarray) -> dict:
    """
    Returns: auroc, auprc, f1_opt, prec_opt, rec_opt,
             shd, shd_norm, f1_topk, shd_topk, k_true, best_thr
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
    n_pos, n_neg = y_true.sum(), (y_true == 0).sum()

    # AUROC / AUPRC
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

    # Top-k (k = #true edges)
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
        res["shd_topk"] = fp + fn
    else:
        res["f1_topk"] = 0.0
        res["shd_topk"] = int(k_true + (y_score > 0).sum())

    return res


def combine_W0_Wk(W0: np.ndarray, Wk_list: list) -> np.ndarray:
    """Element-wise max of |W0|, |W1|, ..., |WK|."""
    abs_mats = [np.abs(W0)] + [np.abs(wk) for wk in Wk_list]
    return np.stack(abs_mats, axis=0).max(axis=0)


def compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est) -> dict:
    """
    Compute both W0-only and combined (W0+Wk) metrics.
    Returns dict with 'w0_' and 'comb_' prefixed keys.
    """
    # W0-only
    met_w0 = compute_all_metrics(W0_true, W0_est)
    res = {f"w0_{k}": v for k, v in met_w0.items()}

    # Combined: ground truth union, estimated max
    B_comb_true = (np.abs(W0_true) > 1e-10).astype(int)
    for Wk_t in Wk_true:
        B_comb_true = np.maximum(B_comb_true, (np.abs(Wk_t) > 1e-10).astype(int))
    W_comb_est = combine_W0_Wk(W0_est, Wk_est)
    met_comb = compute_all_metrics(B_comb_true.astype(float) * 0.5, W_comb_est)
    res.update({f"comb_{k}": v for k, v in met_comb.items()})

    # Keep backward-compat: unqualified keys = W0-only (original behavior)
    res.update(met_w0)

    return res


# ====================================================================
# 6. Baseline: DYNOTEARS (from scratch — no causalnex required)
# ====================================================================

class _DYNOTEARS(nn.Module):
    """
    DYNOTEARS (Pamfil et al., 2020) re-implemented in PyTorch ALM.
    min  0.5/T ||X_t - X_t W0 - sum_k X_{t-k} Wk||^2 + lam ||.||_1
    s.t. h(W0) = tr(exp(W0 o W0)) - d = 0
    """
    def __init__(self, d: int, K: int, lam: float = 0.01):
        super().__init__()
        self.d, self.K, self.lam = d, K, lam
        s = 1e-2
        self.W0 = nn.Parameter(s * torch.randn(d, d))
        self.Wk = nn.ParameterList(
            [nn.Parameter(s * torch.randn(d, d)) for _ in range(K)]
        )
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

# ====================================================================
# 7. Baseline: PCMCI+  (tigramite wrapper)
# ====================================================================

def run_pcmci_plus(X, d, K, alpha_level=0.05, seed=0):
    """Returns (W0_continuous_scores, Wk_list) or (None, None)."""
    if not HAS_TIGRAMITE:
        return None, None
    try:
        df = pp.DataFrame(X, var_names=[f"V{i}" for i in range(d)])
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=df, cond_ind_test=parcorr, verbosity=0)
        res = pcmci.run_pcmciplus(tau_min=0, tau_max=K, pc_alpha=alpha_level)
        val = res["val_matrix"]            # (d, d, K+1)
        graph = res["graph"]               # (d, d, K+1)  string

        # Instantaneous: use |val| as continuous score
        # For direction, only keep i->j if graph[i,j,0] contains '-->'
        W0 = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                g = str(graph[i, j, 0])
                if "-->" in g:
                    W0[i, j] = abs(val[i, j, 0])
                elif "o-o" in g or "x-x" in g:
                    # Undirected / unresolved contemporaneous
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

# ====================================================================
# 8. Baseline: VARLiNGAM  (lingam wrapper)
# ====================================================================

def run_varlingam(X, d, K, seed=0):
    """Returns (W0, Wk_list) or (None, None).
    lingam convention: B[i,j] means j->i. Our convention: W[i,j] means i->j.
    So W = B.T
    """
    if not HAS_LINGAM:
        return None, None
    try:
        model = lingam.VARLiNGAM(lags=K, random_state=seed)
        model.fit(X)
        B0 = model.adjacency_matrices_[0]     # (d,d), B[i,j] = j->i
        W0 = B0.T                              # now W[i,j] = i->j
        np.fill_diagonal(W0, 0.0)
        Wk = []
        for k in range(1, K + 1):
            Bk = model.adjacency_matrices_[k]
            Wk.append(Bk.T)
        return W0, Wk
    except Exception as e:
        warnings.warn(f"VARLiNGAM failed: {e}")
        return None, None

# ====================================================================
# 9. PRCD-MAP wrapper
# ====================================================================

def run_prcd_map(X, P_prior, d, K, lambda1=0.01, lambda2=0.01,
                 learn_tau=True, tau0=1.0,
                 max_iter=30, inner_iter=500, lr=1e-2, seed=0):
    """Returns (W0, Wk, tau)."""
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau, tau0=tau0,
        tau_min=0.1, tau_max=10.0
    )
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
        verbose=False, postprocess=False,
    )
    return W0, Wk, tau

# ====================================================================
# 10. Configuration
# ====================================================================

@dataclass
class Cfg:
    dims:          List[int]   = field(default_factory=lambda: [10, 20, 50])
    sample_sizes:  List[int]   = field(default_factory=lambda: [500, 1000])
    graph_types:   List[str]   = field(default_factory=lambda: ["ER"])
    noise_types:   List[str]   = field(default_factory=lambda: ["gaussian"])
    lag_orders:    List[int]   = field(default_factory=lambda: [1])
    nonlinear:     bool        = False
    prior_accs:    List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior_modes:   List[str]   = field(default_factory=lambda: ["random"])
    seeds:         List[int]   = field(default_factory=lambda: list(range(10)))
    # graph params
    edge_prob_w0:  float = 0.15
    ba_m:          int   = 2
    edge_prob_wk:  float = 0.10
    # method hyper-params
    lambda1:       float = 0.01
    lambda2:       float = 0.01
    max_iter:      int   = 30
    inner_iter:    int   = 500
    lr:            float = 1e-2
    # which baselines to run
    do_dynotears:  bool = True
    do_pcmci:      bool = True
    do_varlingam:  bool = True
    # output
    output_dir:    str  = "exp1_results"


def cfg_quick():
    return Cfg(dims=[10], sample_sizes=[500], seeds=list(range(3)),
               prior_accs=[0.2, 0.6, 1.0], max_iter=15, inner_iter=100,
               do_pcmci=False, do_varlingam=False)

def cfg_full():
    return Cfg(
        dims=[10, 20, 50, 100],
        sample_sizes=[200, 500, 1000, 2000],
        graph_types=["ER", "BA"],
        noise_types=["gaussian", "laplace", "student_t", "heteroscedastic"],
        lag_orders=[1, 2, 3],
        nonlinear=True,
        prior_accs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        prior_modes=["random", "systematic", "adversarial"],
        seeds=list(range(10)),
    )

def cfg_sub(name: str):
    """Pre-built configs for individual sub-experiments."""
    if name == "noise":
        return Cfg(
            dims=[20], sample_sizes=[500],
            noise_types=["gaussian", "laplace", "student_t", "heteroscedastic"],
            prior_accs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            output_dir="exp1_noise",
        )
    elif name == "graph":
        return Cfg(
            dims=[20], sample_sizes=[500],
            graph_types=["ER", "BA"],
            prior_accs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            output_dir="exp1_graph",
        )
    elif name == "nonlinear":
        return Cfg(
            dims=[10, 20], sample_sizes=[500, 1000],
            nonlinear=True,
            noise_types=["gaussian"],
            prior_accs=[0.2, 0.6, 1.0],
            output_dir="exp1_nonlinear",
        )
    elif name == "prior":
        return Cfg(
            dims=[20], sample_sizes=[500],
            prior_accs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            prior_modes=["random", "systematic", "adversarial"],
            output_dir="exp1_prior",
        )
    elif name == "scale":
        return Cfg(
            dims=[10, 20, 50, 100, 200],
            sample_sizes=[500],
            prior_accs=[0.6],
            seeds=list(range(3)),
            do_pcmci=False,   # too slow for d=200
            output_dir="exp1_scale",
        )
    else:
        raise ValueError(f"Unknown sub-experiment: {name}")

# ====================================================================
# 11. Main Experiment Loop
# ====================================================================

def run_experiment(cfg: Cfg) -> pd.DataFrame:
    os.makedirs(cfg.output_dir, exist_ok=True)
    all_rows = []
    run_id = 0

    # Enumerate linear settings
    settings = []
    for d in cfg.dims:
        for T in cfg.sample_sizes:
            for gt in cfg.graph_types:
                for nt in cfg.noise_types:
                    for K in cfg.lag_orders:
                        for acc in cfg.prior_accs:
                            for pm in cfg.prior_modes:
                                for s in cfg.seeds:
                                    settings.append(dict(
                                        d=d, T=T, graph=gt, noise=nt, K=K,
                                        prior_acc=acc, prior_mode=pm,
                                        seed=s, nonlinear=False))
    # Nonlinear settings
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
                                    seed=s, nonlinear=True))

    n_total = len(settings)
    method_names = ["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)", "PRCD-MAP(uniform)"]
    if cfg.do_dynotears:
        method_names.append("DYNOTEARS")
    if cfg.do_pcmci and HAS_TIGRAMITE:
        method_names.append("PCMCI+")
    if cfg.do_varlingam and HAS_LINGAM:
        method_names.append("VARLiNGAM")

    print(f">>> {n_total} settings x {len(method_names)} methods")
    print(f">>> Methods: {', '.join(method_names)}")
    t_global = time.time()

    for idx, st in enumerate(settings):
        d, T, K, seed = st["d"], st["T"], st["K"], st["seed"]
        gt, nt = st["graph"], st["noise"]
        acc, pm = st["prior_acc"], st["prior_mode"]
        nl = st["nonlinear"]

        if (idx + 1) % max(1, n_total // 20) == 0 or idx == 0:
            elapsed = time.time() - t_global
            print(f"  [{idx+1}/{n_total}] d={d} T={T} {gt} {nt} K={K} "
                  f"acc={acc} {pm} nl={nl} s={seed}  ({elapsed:.0f}s)")

        base = dict(d=d, T=T, graph=gt, noise=nt, K=K,
                    prior_acc=acc, prior_mode=pm, seed=seed, nonlinear=nl)

        # --- Ground truth ---
        if gt == "ER":
            W0_true = make_er_dag(d, cfg.edge_prob_w0, seed=seed)
        else:
            W0_true = make_ba_dag(d, cfg.ba_m, seed=seed)
        Wk_true = make_lag_matrices(d, K, cfg.edge_prob_wk, seed=seed)

        # --- Simulate data ---
        try:
            if nl:
                X = simulate_svar_nonlinear(T, W0_true, Wk_true, nt, seed=seed)
            else:
                X = simulate_svar_linear(T, W0_true, Wk_true, nt, seed=seed)
        except Exception as e:
            warnings.warn(f"Sim failed {st}: {e}")
            continue
        if not np.all(np.isfinite(X)):
            continue

        # --- Prior ---
        P_prior = gen_prior(W0_true, Wk_true, acc, pm, seed=seed + 999)

        # ======= Method 1: PRCD-MAP (learn tau) =======
        try:
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                learn_tau=True, seed=seed,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr)
            met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
            all_rows.append({**base, "method": "PRCD-MAP(learn_tau)",
                             "tau": float(tau), "time": time.time()-t0, **met})
        except Exception:
            pass

        # ======= Method 2: PRCD-MAP (fixed tau=1) =======
        try:
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_prior, d, K, cfg.lambda1, cfg.lambda2,
                learn_tau=False, tau0=1.0, seed=seed,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr)
            met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
            all_rows.append({**base, "method": "PRCD-MAP(fixed_tau)",
                             "tau": float(tau), "time": time.time()-t0, **met})
        except Exception:
            pass

        # ======= Method 3: PRCD-MAP (uniform prior P=0.5) =======
        try:
            P_unif = np.full((d, d), 0.5)
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_unif, d, K, cfg.lambda1, cfg.lambda2,
                learn_tau=True, seed=seed,
                max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr)
            met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
            all_rows.append({**base, "method": "PRCD-MAP(uniform)",
                             "tau": float(tau), "time": time.time()-t0, **met})
        except Exception:
            pass

        # ======= Method 4: DYNOTEARS =======
        if cfg.do_dynotears:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_dynotears(
                    X, d, K, lam=cfg.lambda1, max_outer=cfg.max_iter,
                    inner=cfg.inner_iter, lr=cfg.lr, seed=seed)
                met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
                all_rows.append({**base, "method": "DYNOTEARS",
                                 "tau": np.nan, "time": time.time()-t0, **met})
            except Exception:
                pass

        # ======= Method 5: PCMCI+ =======
        if cfg.do_pcmci and HAS_TIGRAMITE and d <= 80:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_pcmci_plus(X, d, K, seed=seed)
                if W0_est is not None:
                    met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est if Wk_est else [])
                    all_rows.append({**base, "method": "PCMCI+",
                                     "tau": np.nan, "time": time.time()-t0, **met})
            except Exception:
                pass

        # ======= Method 6: VARLiNGAM =======
        if cfg.do_varlingam and HAS_LINGAM:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_varlingam(X, d, K, seed=seed)
                if W0_est is not None:
                    met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est if Wk_est else [])
                    all_rows.append({**base, "method": "VARLiNGAM",
                                     "tau": np.nan, "time": time.time()-t0, **met})
            except Exception:
                pass

        # Periodic save
        if (idx + 1) % 100 == 0:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(cfg.output_dir, "_intermediate.csv"), index=False)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(cfg.output_dir, "exp1_full_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {csv_path}")

    # Post-processing
    generate_summaries(df, cfg.output_dir)
    generate_figures(df, cfg.output_dir)
    return df

# ====================================================================
# 12. Summary Tables
# ====================================================================

def generate_summaries(df: pd.DataFrame, out: str):
    if df.empty:
        return
    metric_cols = ["auroc", "auprc", "f1_opt", "shd_norm", "f1_topk",
                   "w0_auroc", "w0_f1_opt", "comb_auroc", "comb_f1_opt"]

    # --- Table A: by method x (d, T) ---
    g = df.groupby(["method", "d", "T"])
    agg = g.agg(**{f"{c}_mean": (c, "mean") for c in metric_cols},
                **{f"{c}_std":  (c, "std")  for c in metric_cols},
                tau_mean=("tau", "mean"), tau_std=("tau", "std"),
                time_mean=("time", "mean"), n=("seed", "count")).reset_index()
    p = os.path.join(out, "summary_by_dT.csv")
    agg.to_csv(p, index=False); print(f">>> {p}")

    # --- Table B: prior degradation ---
    if "prior_acc" in df.columns:
        g2 = df.groupby(["method", "prior_acc", "prior_mode"])
        agg2 = g2.agg(
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
            f1_mean=("f1_opt", "mean"), f1_std=("f1_opt", "std"),
            tau_mean=("tau", "mean"), tau_std=("tau", "std"),
        ).reset_index()
        p2 = os.path.join(out, "summary_prior_degradation.csv")
        agg2.to_csv(p2, index=False); print(f">>> {p2}")

    # --- Table C: noise ---
    if df["noise"].nunique() > 1:
        g3 = df.groupby(["method", "noise"])
        agg3 = g3.agg(
            auroc_mean=("auroc", "mean"), f1_mean=("f1_opt", "mean"),
        ).reset_index()
        p3 = os.path.join(out, "summary_noise.csv")
        agg3.to_csv(p3, index=False); print(f">>> {p3}")

    # --- Table D: graph structure ---
    if df["graph"].nunique() > 1:
        g4 = df.groupby(["method", "graph"])
        agg4 = g4.agg(
            auroc_mean=("auroc", "mean"), f1_mean=("f1_opt", "mean"),
        ).reset_index()
        p4 = os.path.join(out, "summary_graph.csv")
        agg4.to_csv(p4, index=False); print(f">>> {p4}")

    # --- Console summary ---
    print("\n" + "=" * 72)
    print("Overall mean (method)")
    print("=" * 72)
    overall = df.groupby("method")[metric_cols].mean()
    print(overall.round(4).to_string())
    print()

# ====================================================================
# 13. Figures
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

def generate_figures(df: pd.DataFrame, out: str):
    if df.empty:
        return
    _fig_prior_degradation(df, out)
    _fig_tau_vs_prior(df, out)
    _fig_method_comparison(df, out)
    if df["noise"].nunique() > 1:
        _fig_noise_robustness(df, out)


def _fig_prior_degradation(df, out):
    """Key figure: F1/AUROC vs prior accuracy, one line per method."""
    df_r = df[df["prior_mode"] == "random"] if "prior_mode" in df.columns else df
    if df_r.empty or df_r["prior_acc"].nunique() < 3:
        return
    methods = sorted(df_r["method"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for metric, ax, yl in [("f1_opt", axes[0], "F1 (optimal threshold)"),
                            ("auroc",  axes[1], "AUROC")]:
        for m in methods:
            sub = df_r[df_r["method"] == m]
            agg = sub.groupby("prior_acc").agg(
                y=(metric, "mean"), e=(metric, "std")).reset_index()
            ax.errorbar(agg["prior_acc"], agg["y"], yerr=agg["e"],
                        label=m, color=COLORS.get(m, "grey"),
                        marker=MARKERS.get(m, "x"),
                        linewidth=2, markersize=7, capsize=3)
        ax.set_xlabel("Prior Accuracy", fontsize=12)
        ax.set_ylabel(yl, fontsize=12)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Prior Accuracy Degradation (Random Corruption)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig1_prior_degradation"))

    # Additional plots for other prior modes
    for pm in ["systematic", "adversarial"]:
        sub_pm = df[df["prior_mode"] == pm] if "prior_mode" in df.columns else pd.DataFrame()
        if sub_pm.empty or sub_pm["prior_acc"].nunique() < 3:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for m in sorted(sub_pm["method"].unique()):
            sub = sub_pm[sub_pm["method"] == m]
            agg = sub.groupby("prior_acc").agg(
                y=("f1_opt", "mean"), e=("f1_opt", "std")).reset_index()
            ax.errorbar(agg["prior_acc"], agg["y"], yerr=agg["e"],
                        label=m, color=COLORS.get(m, "grey"),
                        marker=MARKERS.get(m, "x"),
                        linewidth=2, markersize=7, capsize=3)
        ax.set_xlabel("Prior Accuracy"); ax.set_ylabel("F1")
        ax.set_title(f"Prior Degradation ({pm.capitalize()})", fontsize=13)
        ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
        plt.tight_layout()
        _save(os.path.join(out, f"fig1_{pm}_degradation"))


def _fig_tau_vs_prior(df, out):
    """Figure 2: learned tau vs prior accuracy (dual y-axis with F1)."""
    sub = df[df["method"] == "PRCD-MAP(learn_tau)"]
    if "prior_mode" in sub.columns:
        sub = sub[sub["prior_mode"] == "random"]
    if sub.empty or sub["prior_acc"].nunique() < 3:
        return

    agg = sub.groupby("prior_acc").agg(
        tau_m=("tau", "mean"), tau_s=("tau", "std"),
        f1_m=("f1_opt", "mean"), f1_s=("f1_opt", "std")).reset_index()

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    c1, c2 = "#E74C3C", "#2C3E50"
    ax1.errorbar(agg["prior_acc"], agg["tau_m"], yerr=agg["tau_s"],
                 color=c1, marker="o", lw=2.5, ms=8, capsize=4, label="Learned τ")
    ax1.set_xlabel("Prior Accuracy", fontsize=13)
    ax1.set_ylabel("Learned Temperature τ", fontsize=13, color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2 = ax1.twinx()
    ax2.errorbar(agg["prior_acc"], agg["f1_m"], yerr=agg["f1_s"],
                 color=c2, marker="s", lw=2.5, ms=8, capsize=4, ls="--", label="F1")
    ax2.set_ylabel("F1 (optimal threshold)", fontsize=13, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=11)
    ax1.set_title("Temperature τ Adaptation vs Prior Quality",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    _save(os.path.join(out, "fig2_tau_vs_prior"))


def _fig_method_comparison(df, out):
    """Figure 3: grouped bar chart by d."""
    methods = sorted(df["method"].unique())
    dims = sorted(df["d"].unique())
    n_m, n_d = len(methods), len(dims)
    if n_m == 0 or n_d == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for metric, ax, yl in [("auroc", axes[0], "AUROC"),
                            ("f1_opt", axes[1], "F1")]:
        x = np.arange(n_d)
        w = 0.8 / n_m
        for i, m in enumerate(methods):
            vals, errs = [], []
            for d in dims:
                s = df[(df["method"] == m) & (df["d"] == d)]
                vals.append(s[metric].mean() if len(s) else 0)
                errs.append(s[metric].std() if len(s) else 0)
            ax.bar(x + i * w, vals, w, yerr=errs,
                   label=m, color=COLORS.get(m, "grey"), alpha=0.85, capsize=2)
        ax.set_xlabel("d"); ax.set_ylabel(yl)
        ax.set_xticks(x + w * (n_m - 1) / 2)
        ax.set_xticklabels([str(d) for d in dims])
        ax.legend(fontsize=7); ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.suptitle("Method Comparison Across Dimensions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig3_method_comparison"))


def _fig_noise_robustness(df, out):
    """Figure 4: bar chart by noise type."""
    methods = sorted(df["method"].unique())
    noises = sorted(df["noise"].unique())
    n_m, n_n = len(methods), len(noises)
    if n_m == 0 or n_n == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_n)
    w = 0.8 / n_m
    for i, m in enumerate(methods):
        vals, errs = [], []
        for nt in noises:
            s = df[(df["method"] == m) & (df["noise"] == nt)]
            vals.append(s["f1_opt"].mean() if len(s) else 0)
            errs.append(s["f1_opt"].std() if len(s) else 0)
        ax.bar(x + i * w, vals, w, yerr=errs,
               label=m, color=COLORS.get(m, "grey"), alpha=0.85, capsize=2)
    ax.set_xlabel("Noise Type"); ax.set_ylabel("F1")
    ax.set_xticks(x + w * (n_m - 1) / 2)
    ax.set_xticklabels(noises)
    ax.legend(fontsize=8); ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.set_title("Noise Robustness Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig4_noise_robustness"))

# ====================================================================
# 14. Scalability timing
# ====================================================================

def run_scalability(cfg: Cfg) -> pd.DataFrame:
    """Wall-clock time vs d for each method."""
    print("\n>>> Scalability timing ...")
    rows = []
    T, K, acc = 500, 1, 0.6
    for d in cfg.dims:
        for seed in cfg.seeds:
            W0 = make_er_dag(d, cfg.edge_prob_w0, seed=seed)
            Wk = make_lag_matrices(d, K, cfg.edge_prob_wk, seed=seed)
            X = simulate_svar_linear(T, W0, Wk, "laplace", seed=seed)
            P = gen_prior(W0, Wk, acc, "random", seed=seed + 999)
            base = dict(d=d, seed=seed)

            for name, fn in [
                ("PRCD-MAP(learn_tau)", lambda: run_prcd_map(
                    X, P, d, K, cfg.lambda1, cfg.lambda2, True, seed=seed,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr)),
                ("DYNOTEARS", lambda: run_dynotears(
                    X, d, K, cfg.lambda1, cfg.max_iter, cfg.inner_iter,
                    cfg.lr, seed)),
            ]:
                try:
                    t0 = time.time()
                    fn()
                    rows.append({**base, "method": name, "time": time.time() - t0})
                except Exception:
                    pass

            if cfg.do_pcmci and HAS_TIGRAMITE and d <= 80:
                try:
                    t0 = time.time()
                    run_pcmci_plus(X, d, K, seed=seed)
                    rows.append({**base, "method": "PCMCI+", "time": time.time() - t0})
                except Exception:
                    pass

            if cfg.do_varlingam and HAS_LINGAM:
                try:
                    t0 = time.time()
                    run_varlingam(X, d, K, seed=seed)
                    rows.append({**base, "method": "VARLiNGAM", "time": time.time() - t0})
                except Exception:
                    pass

            print(f"  d={d}, seed={seed} done")

    df_t = pd.DataFrame(rows)
    if df_t.empty:
        return df_t
    p = os.path.join(cfg.output_dir, "scalability_timing.csv")
    df_t.to_csv(p, index=False); print(f">>> {p}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in sorted(df_t["method"].unique()):
        sub = df_t[df_t["method"] == m]
        agg = sub.groupby("d").agg(t_m=("time", "mean"), t_s=("time", "std")).reset_index()
        ax.errorbar(agg["d"], agg["t_m"], yerr=agg["t_s"],
                    label=m, marker=MARKERS.get(m, "x"),
                    color=COLORS.get(m, "grey"), lw=2, capsize=3)
    ax.set_xlabel("d"); ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Scalability: Runtime vs Dimension", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    _save(os.path.join(cfg.output_dir, "fig5_scalability"))
    return df_t

# ====================================================================
# 15. Hyperparameter sensitivity
# ====================================================================

def run_hyperparam_sensitivity(cfg: Cfg) -> pd.DataFrame:
    """lambda1 x lambda2 grid on a representative synthetic setting."""
    print("\n>>> Hyperparameter sensitivity ...")
    d, T, K = 20, 500, 1
    lam1_grid = [0.001, 0.01, 0.1, 0.5]
    lam2_grid = [0.001, 0.01, 0.1, 0.5]
    rows = []
    for seed in cfg.seeds[:3]:
        W0 = make_er_dag(d, cfg.edge_prob_w0, seed=seed)
        Wk = make_lag_matrices(d, K, cfg.edge_prob_wk, seed=seed)
        X = simulate_svar_linear(T, W0, Wk, "laplace", seed=seed)
        P = gen_prior(W0, Wk, 0.6, "random", seed=seed + 999)
        for l1 in lam1_grid:
            for l2 in lam2_grid:
                try:
                    W0_est, _, tau = run_prcd_map(
                        X, P, d, K, l1, l2, learn_tau=True, seed=seed,
                        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                        lr=cfg.lr)
                    met = compute_all_metrics(W0, W0_est)
                    rows.append(dict(lambda1=l1, lambda2=l2, seed=seed,
                                     tau=tau, **met))
                except Exception:
                    pass
    df_hp = pd.DataFrame(rows)
    if df_hp.empty:
        return df_hp
    p = os.path.join(cfg.output_dir, "hyperparam_sensitivity.csv")
    df_hp.to_csv(p, index=False); print(f">>> {p}")

    # Heatmap
    agg = df_hp.groupby(["lambda1", "lambda2"])["f1_opt"].mean().reset_index()
    piv = agg.pivot(index="lambda1", columns="lambda2", values="f1_opt")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto",
                   origin="lower", vmin=0, vmax=max(0.01, piv.values.max()))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v:.3f}" for v in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in piv.index])
    ax.set_xlabel("λ₂ (prior weight)"); ax.set_ylabel("λ₁ (sparsity)")
    ax.set_title("F1 Sensitivity to (λ₁, λ₂)", fontsize=14, fontweight="bold")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f"{piv.values[i,j]:.3f}",
                    ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="F1")
    plt.tight_layout()
    _save(os.path.join(cfg.output_dir, "fig6_hyperparam_heatmap"))
    return df_hp

# ====================================================================
# 16. Entry Point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Synthetic Benchmark")
    parser.add_argument("--quick", action="store_true", help="Tiny test run")
    parser.add_argument("--full",  action="store_true", help="Full NeurIPS sweep")
    parser.add_argument("--sub",   type=str, default=None,
                        choices=["noise","graph","nonlinear","prior","scale"],
                        help="Run a specific sub-experiment")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dims",  nargs="+", type=int, default=None)
    parser.add_argument("--Ts",    nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--no-pcmci",    action="store_true")
    parser.add_argument("--no-varlingam",action="store_true")
    parser.add_argument("--no-dynotears",action="store_true")
    parser.add_argument("--scalability", action="store_true",
                        help="Also run scalability timing")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Also run hyperparam sensitivity")
    args = parser.parse_args()

    # Select config
    if args.quick:
        cfg = cfg_quick()
    elif args.full:
        cfg = cfg_full()
    elif args.sub:
        cfg = cfg_sub(args.sub)
    else:
        cfg = Cfg()

    # Override from CLI
    if args.output:    cfg.output_dir    = args.output
    if args.dims:      cfg.dims          = args.dims
    if args.Ts:        cfg.sample_sizes  = args.Ts
    if args.seeds:     cfg.seeds         = args.seeds
    if args.no_pcmci:     cfg.do_pcmci     = False
    if args.no_varlingam: cfg.do_varlingam = False
    if args.no_dynotears: cfg.do_dynotears = False

    print("=" * 64)
    print(" Experiment 1: Comprehensive Synthetic Benchmark")
    print("=" * 64)
    print(f"  dims       = {cfg.dims}")
    print(f"  Ts         = {cfg.sample_sizes}")
    print(f"  graphs     = {cfg.graph_types}")
    print(f"  noise      = {cfg.noise_types}")
    print(f"  K          = {cfg.lag_orders}")
    print(f"  nonlinear  = {cfg.nonlinear}")
    print(f"  prior_accs = {cfg.prior_accs}")
    print(f"  prior_modes= {cfg.prior_modes}")
    print(f"  seeds      = {cfg.seeds}")
    print(f"  baselines  = DYNOTEARS={cfg.do_dynotears}, "
          f"PCMCI+={cfg.do_pcmci and HAS_TIGRAMITE}, "
          f"VARLiNGAM={cfg.do_varlingam and HAS_LINGAM}")
    print(f"  output     = {cfg.output_dir}")
    print("=" * 64)

    t0 = time.time()
    df = run_experiment(cfg)
    print(f"\n>>> Main experiment done in {time.time()-t0:.1f}s ({len(df)} rows)")

    if args.scalability or args.sub == "scale":
        run_scalability(cfg)

    if args.sensitivity:
        run_hyperparam_sensitivity(cfg)

    print(f"\n>>> All done. Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
