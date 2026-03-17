"""
=============================================================================
Experiment 2 — Standard Public Real-World Benchmark Datasets for PRCD-MAP
=============================================================================
NeurIPS-grade evaluation on public benchmarks with known ground truth:

  Benchmark 1: Lorenz-96 dynamical system (self-generated, always available)
  Benchmark 2: CausalTime benchmark (Cheng et al., ICLR 2024)
  Benchmark 3: DREAM4 gene regulatory networks (classic benchmark)
  Benchmark 4: Netsim fMRI (synthetic fMRI with known connectivity)
  Case Study:  Electricity consumption (private, interpretability focus)

Baselines: DYNOTEARS, PCMCI+, VARLiNGAM, PRCD-MAP variants
Metrics:   AUROC, AUPRC, Best-F1, Directed SHD, Normalized SHD

Usage:
  python exp2_real_benchmarks.py                       # all available benchmarks
  python exp2_real_benchmarks.py --quick               # Lorenz-96 only, small
  python exp2_real_benchmarks.py --bench lorenz96      # specific benchmark
  python exp2_real_benchmarks.py --bench causaltime    # CausalTime only
  python exp2_real_benchmarks.py --bench dream4        # DREAM4 only
  python exp2_real_benchmarks.py --bench netsim        # Netsim fMRI only
  python exp2_real_benchmarks.py --bench electricity   # Electricity case study
  python exp2_real_benchmarks.py --seeds 0 1 2 3 4
  python exp2_real_benchmarks.py --prior-accs 0.0 0.2 0.4 0.6 0.8 1.0

Data directories (create & populate before running):
  ./data/causaltime/       — clone https://github.com/causaltime/causaltime
  ./data/dream4/           — download from https://gnw.sourceforge.net/
  ./data/netsim/           — download Netsim dataset or use CDT package
=============================================================================
"""

import os, sys, time, warnings, argparse, json, traceback
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.integrate import solve_ivp

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
    """Zero-mean unit-variance standardization (per column)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd

# ====================================================================
# 2. Evaluation Metrics (shared with exp1)
# ====================================================================
def compute_all_metrics(B_true: np.ndarray,
                        W_est_continuous: np.ndarray,
                        exclude_diag: bool = True) -> dict:
    """
    Evaluate estimated continuous weight matrix against binary ground truth.

    Args:
        B_true: binary ground truth adjacency (d x d), B[i,j]=1 means i->j
        W_est_continuous: continuous estimated weights (d x d)
        exclude_diag: whether to exclude diagonal (self-loops) from eval

    Returns dict with: auroc, auprc, f1_opt, prec_opt, rec_opt,
                       shd, shd_norm, f1_topk, shd_topk, k_true, best_thr
    """
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
    if exclude_diag:
        np.fill_diagonal(B_opt, 0)
    res["shd"] = int((B[mask] != B_opt[mask]).sum())
    n_pairs = int(mask.sum())
    res["shd_norm"] = res["shd"] / max(n_pairs, 1)

    # Top-k (k = #true edges)
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
    """
    Combine instantaneous + lagged weights into a summary score matrix.
    For each (i,j), take max absolute value across all lags.
    """
    abs_mats = [np.abs(W0)] + [np.abs(wk) for wk in Wk_list]
    combined = np.stack(abs_mats, axis=0).max(axis=0)
    return combined


# ====================================================================
# 3. Prior Generation for Benchmarks
# ====================================================================
def gen_prior_from_truth(B_true: np.ndarray, acc: float,
                         mode: str = "random", seed: int = 0) -> np.ndarray:
    """
    Generate a prior P in [0,1]^{dxd} from binary ground truth B_true.

    Args:
        B_true: binary adjacency (d x d)
        acc: probability that prior agrees with ground truth
        mode: "random" | "systematic" | "adversarial"
        seed: random seed
    """
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
# 4. Benchmark Data Generators / Loaders
# ====================================================================

# ---- 4a. Lorenz-96 Dynamical System ----

def _lorenz96_rhs(t, x, F):
    """Right-hand side of Lorenz-96 ODE."""
    d = len(x)
    dxdt = np.zeros(d)
    for i in range(d):
        dxdt[i] = (x[(i + 1) % d] - x[(i - 2) % d]) * x[(i - 1) % d] - x[i] + F
    return dxdt


def lorenz96_ground_truth(d: int) -> np.ndarray:
    """
    Ground truth summary causal graph for Lorenz-96.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    Variable i has causal parents: {i-2, i-1, i+1} (mod d), plus self.

    Returns B (d x d) where B[j,i]=1 means j causally affects i.
    Diagonal (self-loops) excluded from evaluation.
    """
    B = np.zeros((d, d), dtype=int)
    for i in range(d):
        B[(i - 2) % d, i] = 1
        B[(i - 1) % d, i] = 1
        B[(i + 1) % d, i] = 1
        # self-loop B[i, i] = 1 excluded (diagonal not evaluated)
    return B


def generate_lorenz96(d: int = 10, T: int = 2000, F: float = 10.0,
                      dt: float = 0.05, subsample: int = 1,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lorenz-96 time series with known ground truth causal graph.

    Args:
        d: number of variables
        T: number of output time steps (after subsampling + burn-in removal)
        F: forcing constant (10 = chaotic regime)
        dt: integration time step
        subsample: keep every `subsample`-th time step
        seed: random seed for initial conditions

    Returns:
        X: (T, d) standardized time series
        B_true: (d, d) binary ground truth adjacency
    """
    rng = np.random.default_rng(seed)

    # Initial condition: small perturbation around F
    x0 = F * np.ones(d) + rng.normal(0, 0.01, d)

    # Total integration time: burn-in + required samples
    burn_in_steps = 500
    total_steps = burn_in_steps + T * subsample
    t_span = (0, total_steps * dt)
    t_eval = np.arange(0, total_steps * dt, dt)

    sol = solve_ivp(_lorenz96_rhs, t_span, x0, args=(F,),
                    t_eval=t_eval, method="RK45", max_step=dt)

    if sol.status != 0:
        warnings.warn(f"Lorenz-96 integration failed: {sol.message}")
        return np.zeros((T, d)), lorenz96_ground_truth(d)

    X_full = sol.y.T  # (n_steps, d)

    # Remove burn-in and subsample
    X_full = X_full[burn_in_steps:]
    if subsample > 1:
        X_full = X_full[::subsample]

    # Truncate to exactly T
    X_full = X_full[:T]
    if len(X_full) < T:
        warnings.warn(f"Lorenz-96: got {len(X_full)} steps, requested {T}, padding")
        pad = np.zeros((T - len(X_full), d))
        X_full = np.vstack([X_full, pad])

    X_std = standardize(X_full)
    B_true = lorenz96_ground_truth(d)

    return X_std, B_true


# ---- 4b. CausalTime Benchmark Loader ----

def discover_causaltime_datasets(base_dir: str) -> List[Dict[str, Any]]:
    """
    Auto-discover CausalTime datasets under base_dir.

    Expected structure (common patterns from the CausalTime repo):
      base_dir/
        <dataset_name>/
          data.npy  OR  data.csv  OR  timeseries.npy
          dag.npy   OR  adj.npy   OR  graph.npy  OR  W.npy

    Returns list of dicts with keys: name, data_path, graph_path
    """
    datasets = []
    if not os.path.isdir(base_dir):
        return datasets

    data_patterns = ["data.npy", "timeseries.npy", "X.npy", "data.csv"]
    graph_patterns = ["dag.npy", "adj.npy", "graph.npy", "W.npy",
                      "DAG.npy", "Adj.npy", "true_graph.npy"]

    for entry in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, entry)
        if not os.path.isdir(sub):
            continue
        data_path, graph_path = None, None
        for dp in data_patterns:
            p = os.path.join(sub, dp)
            if os.path.exists(p):
                data_path = p
                break
        for gp in graph_patterns:
            p = os.path.join(sub, gp)
            if os.path.exists(p):
                graph_path = p
                break
        if data_path and graph_path:
            datasets.append(dict(name=f"CausalTime_{entry}",
                                 data_path=data_path,
                                 graph_path=graph_path))

    return datasets


def load_causaltime_dataset(data_path: str, graph_path: str,
                            max_T: int = 3000) -> Tuple[Optional[np.ndarray],
                                                         Optional[np.ndarray]]:
    """
    Load a single CausalTime dataset.

    Returns (X, B_true) or (None, None) on failure.
    """
    try:
        if data_path.endswith(".npy"):
            X = np.load(data_path)
        elif data_path.endswith(".csv"):
            X = pd.read_csv(data_path).values
        else:
            return None, None

        B_true = np.load(graph_path)

        # Handle potential shapes
        if X.ndim == 3:
            # (num_samples, T, d) — take first sample
            X = X[0]
        if X.ndim != 2:
            warnings.warn(f"CausalTime data shape {X.shape} not supported")
            return None, None

        # Binarize ground truth
        B_true = (np.abs(B_true) > 1e-10).astype(int)
        if B_true.shape[0] != X.shape[1]:
            warnings.warn(f"CausalTime shape mismatch: X={X.shape}, B={B_true.shape}")
            return None, None

        # Truncate if too long
        if len(X) > max_T:
            X = X[:max_T]

        X = standardize(X.astype(np.float64))
        return X, B_true

    except Exception as e:
        warnings.warn(f"Failed to load CausalTime dataset: {e}")
        return None, None


# ---- 4c. DREAM4 Gene Regulatory Network Loader ----

def discover_dream4_datasets(base_dir: str) -> List[Dict[str, Any]]:
    """
    Auto-discover DREAM4 in-silico network challenge datasets.

    Expected file naming (from GNW):
      base_dir/
        insilico_size100_1_timeseries.tsv
        insilico_size100_1_goldstandard.tsv
        insilico_size100_2_timeseries.tsv
        ...
    OR nested directories:
      base_dir/
        network1/
          timeseries.tsv
          goldstandard.tsv
    """
    datasets = []
    if not os.path.isdir(base_dir):
        return datasets

    # Pattern 1: flat files
    import glob
    ts_files = sorted(glob.glob(os.path.join(base_dir, "*timeseries*")))
    for ts_path in ts_files:
        gs_path = ts_path.replace("timeseries", "goldstandard")
        if os.path.exists(gs_path):
            name = os.path.basename(ts_path).replace("_timeseries", "")
            name = name.replace(".tsv", "").replace(".csv", "")
            datasets.append(dict(name=f"DREAM4_{name}",
                                 data_path=ts_path,
                                 graph_path=gs_path))

    # Pattern 2: nested dirs
    for entry in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, entry)
        if not os.path.isdir(sub):
            continue
        ts_candidates = [os.path.join(sub, f) for f in os.listdir(sub)
                         if "timeseries" in f.lower()]
        gs_candidates = [os.path.join(sub, f) for f in os.listdir(sub)
                         if "goldstandard" in f.lower() or "gold_standard" in f.lower()]
        if ts_candidates and gs_candidates:
            datasets.append(dict(name=f"DREAM4_{entry}",
                                 data_path=ts_candidates[0],
                                 graph_path=gs_candidates[0]))

    return datasets


def load_dream4_dataset(data_path: str, graph_path: str,
                        max_T: int = 3000) -> Tuple[Optional[np.ndarray],
                                                     Optional[np.ndarray]]:
    """
    Load DREAM4 time series + gold standard.

    Time series format: tab-separated, first row = gene names,
                        first col might be time index.
    Gold standard format: "G1\tG2\t1" (edge) or "G1\tG2\t0" (no edge).

    Returns (X, B_true) or (None, None).
    """
    try:
        # Load time series
        sep = "\t" if data_path.endswith(".tsv") else ","
        df = pd.read_csv(data_path, sep=sep)

        # Drop time/index column if present
        if df.columns[0].lower() in ("time", "t", "index", ""):
            df = df.iloc[:, 1:]

        gene_names = list(df.columns)
        d = len(gene_names)
        X = df.values.astype(np.float64)

        # DREAM4 often has multiple perturbation experiments concatenated
        # Separated by NaN rows
        nan_rows = np.any(np.isnan(X), axis=1)
        if nan_rows.any():
            # Split by NaN rows and concatenate non-NaN segments
            segments = []
            current = []
            for i in range(len(X)):
                if nan_rows[i]:
                    if current:
                        segments.append(np.array(current))
                        current = []
                else:
                    current.append(X[i])
            if current:
                segments.append(np.array(current))
            # Use longest segment or concatenate all
            X = max(segments, key=len) if segments else X[~nan_rows]

        if len(X) > max_T:
            X = X[:max_T]

        # Load gold standard
        sep_gs = "\t" if graph_path.endswith(".tsv") else ","
        gs = pd.read_csv(graph_path, sep=sep_gs, header=None)
        # Format: src, dst, edge_flag (1/0)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        B_true = np.zeros((d, d), dtype=int)
        for _, row in gs.iterrows():
            src = str(row.iloc[0]).strip()
            dst = str(row.iloc[1]).strip()
            flag = int(row.iloc[2]) if len(row) > 2 else 1
            if src in gene_to_idx and dst in gene_to_idx and flag == 1:
                B_true[gene_to_idx[src], gene_to_idx[dst]] = 1

        X = standardize(X)
        return X, B_true

    except Exception as e:
        warnings.warn(f"Failed to load DREAM4 dataset: {e}")
        return None, None


# ---- 4d. Netsim fMRI Loader ----

def discover_netsim_datasets(base_dir: str) -> List[Dict[str, Any]]:
    """
    Auto-discover Netsim fMRI datasets.

    Expected structure:
      base_dir/
        sim1.mat  OR  sim1/  (with ts.npy, net.npy)
    OR:
      base_dir/
        sim1_timeseries.npy
        sim1_network.npy
    """
    datasets = []
    if not os.path.isdir(base_dir):
        return datasets

    # Pattern: paired .npy files
    import glob
    for ts_path in sorted(glob.glob(os.path.join(base_dir, "*timeseries*"))):
        net_path = ts_path.replace("timeseries", "network")
        if os.path.exists(net_path):
            name = os.path.basename(ts_path).split("_")[0]
            datasets.append(dict(name=f"Netsim_{name}",
                                 data_path=ts_path,
                                 graph_path=net_path))

    # Pattern: .mat files (scipy required)
    for mat_path in sorted(glob.glob(os.path.join(base_dir, "sim*.mat"))):
        name = os.path.splitext(os.path.basename(mat_path))[0]
        datasets.append(dict(name=f"Netsim_{name}",
                             data_path=mat_path,
                             graph_path=mat_path))  # both in same .mat

    # Pattern: nested directories
    for entry in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, entry)
        if not os.path.isdir(sub):
            continue
        ts_p = os.path.join(sub, "ts.npy")
        net_p = os.path.join(sub, "net.npy")
        if os.path.exists(ts_p) and os.path.exists(net_p):
            datasets.append(dict(name=f"Netsim_{entry}",
                                 data_path=ts_p, graph_path=net_p))

    return datasets


def load_netsim_dataset(data_path: str, graph_path: str,
                        max_T: int = 3000,
                        max_d: int = 50) -> Tuple[Optional[np.ndarray],
                                                   Optional[np.ndarray]]:
    """
    Load a Netsim fMRI dataset.

    Returns (X, B_true) or (None, None).
    """
    try:
        if data_path.endswith(".mat"):
            try:
                from scipy.io import loadmat
                mat = loadmat(data_path)
                # Common keys: 'ts' for time series, 'net' for network
                X = None
                B = None
                for key in ["ts", "timeseries", "data", "X"]:
                    if key in mat:
                        X = np.array(mat[key], dtype=np.float64)
                        break
                for key in ["net", "network", "adj", "Adj", "W"]:
                    if key in mat:
                        B = np.array(mat[key])
                        break
                if X is None or B is None:
                    warnings.warn(f"Netsim .mat keys not found: {list(mat.keys())}")
                    return None, None
            except ImportError:
                warnings.warn("scipy not installed — cannot load .mat files")
                return None, None
        else:
            X = np.load(data_path).astype(np.float64)
            B = np.load(graph_path)

        if X.ndim == 3:
            X = X[0]  # take first subject/sample

        B_true = (np.abs(B) > 1e-10).astype(int)

        # Limit dimensions
        d = X.shape[1]
        if d > max_d:
            X = X[:, :max_d]
            B_true = B_true[:max_d, :max_d]
            d = max_d

        if len(X) > max_T:
            X = X[:max_T]

        X = standardize(X)
        return X, B_true

    except Exception as e:
        warnings.warn(f"Failed to load Netsim dataset: {e}")
        return None, None


# ---- 4e. Electricity Case Study ----

def load_electricity_case_study(
    excel_path: str,
    prior_csv_path: str,
    diff_periods: int = 7,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[list]]:
    """
    Load electricity consumption data (no ground truth — case study only).

    Returns (X, P_prior, None, col_names) — B_true is None since no ground truth.
    """
    try:
        df_ts = pd.read_excel(excel_path, index_col=0)
        df_diff = df_ts.diff(periods=diff_periods).dropna()
        col_names = df_diff.columns.tolist()
        X = standardize(df_diff.values.astype(np.float64))

        P_prior = None
        if os.path.exists(prior_csv_path):
            P_prior = pd.read_csv(prior_csv_path, index_col=0).values

        return X, P_prior, None, col_names

    except Exception as e:
        warnings.warn(f"Failed to load electricity data: {e}")
        return None, None, None, None


# ====================================================================
# 5. Baseline Implementations
# ====================================================================

# ---- 5a. DYNOTEARS (PyTorch ALM re-implementation) ----

class _DYNOTEARS(nn.Module):
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


# ---- 5b. PCMCI+ wrapper ----

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


# ---- 5c. VARLiNGAM wrapper ----

def run_varlingam(X, d, K, seed=0):
    if not HAS_LINGAM:
        return None, None
    try:
        model = lingam.VARLiNGAM(lags=K, random_state=seed)
        model.fit(X)
        B0 = model.adjacency_matrices_[0]
        W0 = B0.T
        np.fill_diagonal(W0, 0.0)
        Wk = []
        for k in range(1, K + 1):
            Bk = model.adjacency_matrices_[k]
            Wk.append(Bk.T)
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
# 6. Single-Benchmark Runner
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
    max_iter: int = 30,
    inner_iter: int = 500,
    lr: float = 1e-2,
    do_dynotears: bool = True,
    do_pcmci: bool = True,
    do_varlingam: bool = True,
    eval_combined: bool = True,
) -> pd.DataFrame:
    """
    Run all methods on a single benchmark dataset.

    Args:
        bench_name: name of the benchmark
        X: (T, d) time series data
        B_true: (d, d) binary ground truth adjacency
        K: lag order
        prior_accs: list of prior accuracy values for PRCD-MAP
        seeds: list of random seeds
        eval_combined: if True, evaluate combined (W0+Wk) summary graph

    Returns:
        DataFrame with rows = (method, seed, prior_acc, metrics...)
    """
    T, d = X.shape
    rows = []
    n_edges = int(B_true.sum() - np.trace(B_true))  # off-diagonal edges
    print(f"\n  [{bench_name}] d={d}, T={T}, K={K}, "
          f"true_edges={n_edges}, density={n_edges/(d*(d-1)+1e-12):.3f}")

    for seed in seeds:
        # === Baselines (prior-independent, run once per seed) ===

        # DYNOTEARS
        if do_dynotears:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_dynotears(X, d, K, lam=lambda1,
                                                max_outer=max_iter,
                                                inner=inner_iter,
                                                lr=lr, seed=seed)
                W_eval = combine_W0_Wk(W0_est, Wk_est) if eval_combined else W0_est
                met = compute_all_metrics(B_true, W_eval)
                rows.append(dict(bench=bench_name, method="DYNOTEARS",
                                 seed=seed, prior_acc=np.nan, prior_mode="none",
                                 tau=np.nan, time=time.time()-t0, **met))
            except Exception as e:
                warnings.warn(f"DYNOTEARS failed on {bench_name}: {e}")

        # PCMCI+
        if do_pcmci and HAS_TIGRAMITE and d <= 80:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_pcmci_plus(X, d, K, seed=seed)
                if W0_est is not None:
                    W_eval = combine_W0_Wk(W0_est, Wk_est) if eval_combined else W0_est
                    met = compute_all_metrics(B_true, W_eval)
                    rows.append(dict(bench=bench_name, method="PCMCI+",
                                     seed=seed, prior_acc=np.nan, prior_mode="none",
                                     tau=np.nan, time=time.time()-t0, **met))
            except Exception as e:
                warnings.warn(f"PCMCI+ failed on {bench_name}: {e}")

        # VARLiNGAM
        if do_varlingam and HAS_LINGAM:
            try:
                t0 = time.time()
                W0_est, Wk_est = run_varlingam(X, d, K, seed=seed)
                if W0_est is not None:
                    W_eval = combine_W0_Wk(W0_est, Wk_est) if eval_combined else W0_est
                    met = compute_all_metrics(B_true, W_eval)
                    rows.append(dict(bench=bench_name, method="VARLiNGAM",
                                     seed=seed, prior_acc=np.nan, prior_mode="none",
                                     tau=np.nan, time=time.time()-t0, **met))
            except Exception as e:
                warnings.warn(f"VARLiNGAM failed on {bench_name}: {e}")

        # === PRCD-MAP variants (sweep prior accuracy) ===
        for acc in prior_accs:
            P_prior = gen_prior_from_truth(B_true, acc, mode="random",
                                           seed=seed + 999)

            # PRCD-MAP (learn tau)
            try:
                t0 = time.time()
                W0_est, Wk_est, tau = run_prcd_map(
                    X, P_prior, d, K, lambda1, lambda2,
                    learn_tau=True, seed=seed,
                    max_iter=max_iter, inner_iter=inner_iter, lr=lr)
                W_eval = combine_W0_Wk(W0_est, Wk_est) if eval_combined else W0_est
                met = compute_all_metrics(B_true, W_eval)
                rows.append(dict(bench=bench_name, method="PRCD-MAP(learn_tau)",
                                 seed=seed, prior_acc=acc, prior_mode="random",
                                 tau=float(tau), time=time.time()-t0, **met))
            except Exception as e:
                warnings.warn(f"PRCD-MAP(learn_tau) failed: {e}")

            # PRCD-MAP (fixed tau=1)
            try:
                t0 = time.time()
                W0_est, Wk_est, tau = run_prcd_map(
                    X, P_prior, d, K, lambda1, lambda2,
                    learn_tau=False, tau0=1.0, seed=seed,
                    max_iter=max_iter, inner_iter=inner_iter, lr=lr)
                W_eval = combine_W0_Wk(W0_est, Wk_est) if eval_combined else W0_est
                met = compute_all_metrics(B_true, W_eval)
                rows.append(dict(bench=bench_name, method="PRCD-MAP(fixed_tau)",
                                 seed=seed, prior_acc=acc, prior_mode="random",
                                 tau=float(tau), time=time.time()-t0, **met))
            except Exception as e:
                warnings.warn(f"PRCD-MAP(fixed_tau) failed: {e}")

        # PRCD-MAP (uniform prior) — only once per seed
        try:
            P_unif = np.full((d, d), 0.5)
            t0 = time.time()
            W0_est, Wk_est, tau = run_prcd_map(
                X, P_unif, d, K, lambda1, lambda2,
                learn_tau=True, seed=seed,
                max_iter=max_iter, inner_iter=inner_iter, lr=lr)
            W_eval = combine_W0_Wk(W0_est, Wk_est) if eval_combined else W0_est
            met = compute_all_metrics(B_true, W_eval)
            rows.append(dict(bench=bench_name, method="PRCD-MAP(uniform)",
                             seed=seed, prior_acc=np.nan, prior_mode="uniform",
                             tau=float(tau), time=time.time()-t0, **met))
        except Exception as e:
            warnings.warn(f"PRCD-MAP(uniform) failed: {e}")

        print(f"    seed={seed} done ({len(rows)} rows so far)")

    return pd.DataFrame(rows)


# ====================================================================
# 7. Electricity Case Study Runner (no ground truth)
# ====================================================================

def run_electricity_case_study(
    X: np.ndarray,
    P_prior: np.ndarray,
    col_names: list,
    K: int,
    seeds: List[int],
    lambda1: float = 0.001,
    lambda2: float = 0.02,
    max_iter: int = 35,
    inner_iter: int = 250,
    lr: float = 1e-2,
    topk_per_seed: int = 120,
    output_dir: str = "exp2_results",
    do_dynotears: bool = True,
    do_pcmci: bool = True,
    do_varlingam: bool = True,
) -> pd.DataFrame:
    """
    Run all methods on electricity data WITHOUT ground truth.
    Compare discovered graphs via stability analysis and inter-method agreement.

    Returns DataFrame with per-method edge lists.
    """
    d = X.shape[1]
    print(f"\n  [Electricity Case Study] d={d}, T={X.shape[0]}, K={K}")

    results = {}  # method_name -> list of (W0, Wk) per seed

    # --- PRCD-MAP (with domain prior) ---
    print("    Running PRCD-MAP (domain prior)...")
    prcd_W0s = []
    prcd_taus = []
    for seed in seeds:
        try:
            W0, Wk, tau = run_prcd_map(
                X, P_prior, d, K, lambda1, lambda2,
                learn_tau=True, seed=seed,
                max_iter=max_iter, inner_iter=inner_iter, lr=lr)
            combined = combine_W0_Wk(W0, Wk)
            prcd_W0s.append(combined)
            prcd_taus.append(tau)
        except Exception as e:
            warnings.warn(f"PRCD-MAP seed={seed} failed: {e}")
    results["PRCD-MAP"] = prcd_W0s
    if prcd_taus:
        print(f"    PRCD-MAP tau: {np.mean(prcd_taus):.4f} +/- {np.std(prcd_taus):.4f}")

    # --- DYNOTEARS ---
    if do_dynotears:
        print("    Running DYNOTEARS...")
        dyno_W0s = []
        for seed in seeds:
            try:
                W0, Wk = run_dynotears(X, d, K, lam=lambda1,
                                        max_outer=max_iter, inner=inner_iter,
                                        lr=lr, seed=seed)
                combined = combine_W0_Wk(W0, Wk)
                dyno_W0s.append(combined)
            except Exception:
                pass
        results["DYNOTEARS"] = dyno_W0s

    # --- PCMCI+ ---
    if do_pcmci and HAS_TIGRAMITE and d <= 80:
        print("    Running PCMCI+...")
        pcmci_W0s = []
        for seed in seeds:
            try:
                W0, Wk = run_pcmci_plus(X, d, K, seed=seed)
                if W0 is not None:
                    combined = combine_W0_Wk(W0, Wk)
                    pcmci_W0s.append(combined)
            except Exception:
                pass
        results["PCMCI+"] = pcmci_W0s

    # --- VARLiNGAM ---
    if do_varlingam and HAS_LINGAM:
        print("    Running VARLiNGAM...")
        vl_W0s = []
        for seed in seeds:
            try:
                W0, Wk = run_varlingam(X, d, K, seed=seed)
                if W0 is not None:
                    combined = combine_W0_Wk(W0, Wk)
                    vl_W0s.append(combined)
            except Exception:
                pass
        results["VARLiNGAM"] = vl_W0s

    # --- Compute stability metrics per method ---
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
    names_en = [ZH_TO_EN.get(c, c) for c in col_names]

    summary_rows = []
    for mname, W_list in results.items():
        if not W_list:
            continue
        n_runs = len(W_list)
        # Stability frequency
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

        # Stability score = freq * |w_mean|
        score = freq * np.abs(w_mean)
        np.fill_diagonal(score, 0.0)

        # Top edges
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
        df_edges = df_edges.sort_values(["method", "score"], ascending=[True, False])
        p = os.path.join(output_dir, "electricity_case_study_edges.csv")
        df_edges.to_csv(p, index=False)
        print(f"    Saved: {p}")

    # --- Inter-method agreement ---
    method_names = list(results.keys())
    if len(method_names) >= 2:
        print("\n    Inter-method edge agreement (top-60 edges):")
        for i, m1 in enumerate(method_names):
            for j, m2 in enumerate(method_names):
                if j <= i:
                    continue
                edges1 = set()
                edges2 = set()
                sub1 = df_edges[df_edges["method"] == m1]
                sub2 = df_edges[df_edges["method"] == m2]
                for _, r in sub1.iterrows():
                    edges1.add((r["src"], r["dst"]))
                for _, r in sub2.iterrows():
                    edges2.add((r["src"], r["dst"]))
                overlap = len(edges1 & edges2)
                union = len(edges1 | edges2)
                jaccard = overlap / max(union, 1)
                print(f"      {m1} vs {m2}: overlap={overlap}, "
                      f"Jaccard={jaccard:.3f}")

    return df_edges


# ====================================================================
# 8. Lorenz-96 Multi-Configuration Runner
# ====================================================================

def run_lorenz96_benchmark(
    dims: List[int],
    Ts: List[int],
    Fs: List[float],
    K: int,
    prior_accs: List[float],
    seeds: List[int],
    output_dir: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Run Lorenz-96 benchmark across multiple configurations.
    """
    all_dfs = []
    for d in dims:
        for T in Ts:
            for F_val in Fs:
                for seed in seeds:
                    X, B_true = generate_lorenz96(d=d, T=T, F=F_val, seed=seed)
                    if not np.all(np.isfinite(X)):
                        continue
                    name = f"Lorenz96_d{d}_T{T}_F{F_val}"
                    df = run_single_benchmark(
                        bench_name=name, X=X, B_true=B_true, K=K,
                        prior_accs=prior_accs, seeds=[seed], **kwargs)
                    all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


# ====================================================================
# 9. Configuration
# ====================================================================

@dataclass
class Cfg:
    # Lorenz-96 params
    lorenz_dims:     List[int]   = field(default_factory=lambda: [10, 20])
    lorenz_Ts:       List[int]   = field(default_factory=lambda: [1000, 2000])
    lorenz_Fs:       List[float] = field(default_factory=lambda: [10.0])

    # General params
    K:               int   = 1
    prior_accs:      List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    seeds:           List[int]   = field(default_factory=lambda: list(range(5)))

    # Optimization
    lambda1:         float = 0.01
    lambda2:         float = 0.01
    max_iter:        int   = 30
    inner_iter:      int   = 500
    lr:              float = 1e-2

    # Baselines
    do_dynotears:    bool = True
    do_pcmci:        bool = True
    do_varlingam:    bool = True

    # Data directories
    causaltime_dir:  str = "./data/causaltime"
    dream4_dir:      str = "./data/dream4"
    netsim_dir:      str = "./data/netsim"
    electricity_xlsx: str = r"E:\electricity\0227test.xlsx"
    electricity_prior: str = r"E:\electricity\Auto_Generated_Prior.csv"

    # Benchmarks to run
    benchmarks:      List[str] = field(default_factory=lambda: [
        "lorenz96", "causaltime", "dream4", "netsim", "electricity"
    ])

    # Output
    output_dir:      str = "exp2_results"


def cfg_quick():
    return Cfg(
        lorenz_dims=[10], lorenz_Ts=[500],
        prior_accs=[0.2, 0.6, 1.0],
        seeds=list(range(3)),
        max_iter=15, inner_iter=100,
        do_pcmci=False, do_varlingam=False,
        benchmarks=["lorenz96"],
    )


def cfg_full():
    return Cfg(
        lorenz_dims=[10, 20, 40],
        lorenz_Ts=[500, 1000, 2000],
        lorenz_Fs=[8.0, 10.0],
        prior_accs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        seeds=list(range(10)),
        benchmarks=["lorenz96", "causaltime", "dream4", "netsim", "electricity"],
    )


# ====================================================================
# 10. Summary Tables
# ====================================================================

def generate_summaries(df: pd.DataFrame, out: str):
    if df.empty:
        return
    metric_cols = [c for c in ["auroc", "auprc", "f1_opt", "shd_norm", "f1_topk"]
                   if c in df.columns]
    if not metric_cols:
        return

    # --- Table A: Overall by method x benchmark ---
    g = df.groupby(["bench", "method"])
    agg = g.agg(**{f"{c}_mean": (c, "mean") for c in metric_cols},
                **{f"{c}_std":  (c, "std")  for c in metric_cols},
                tau_mean=("tau", "mean"), tau_std=("tau", "std"),
                time_mean=("time", "mean"),
                n=("seed", "count")).reset_index()
    p = os.path.join(out, "summary_by_bench_method.csv")
    agg.to_csv(p, index=False)
    print(f">>> {p}")

    # --- Table B: Aggregated across benchmarks (method comparison) ---
    # For baselines, use all rows; for PRCD-MAP, use best prior_acc
    baseline_methods = ["DYNOTEARS", "PCMCI+", "VARLiNGAM", "PRCD-MAP(uniform)"]
    prcd_methods = ["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)"]

    df_base = df[df["method"].isin(baseline_methods)]
    # For PRCD-MAP with prior, take acc=0.7 as representative
    df_prcd = df[df["method"].isin(prcd_methods)]
    if "prior_acc" in df_prcd.columns:
        # Use acc closest to 0.7
        if not df_prcd.empty:
            avail_accs = df_prcd["prior_acc"].dropna().unique()
            if len(avail_accs) > 0:
                best_acc = avail_accs[np.argmin(np.abs(avail_accs - 0.7))]
                df_prcd = df_prcd[df_prcd["prior_acc"] == best_acc]

    df_combined = pd.concat([df_base, df_prcd], ignore_index=True)
    if not df_combined.empty:
        g2 = df_combined.groupby("method")
        agg2 = g2.agg(**{f"{c}_mean": (c, "mean") for c in metric_cols},
                       **{f"{c}_std":  (c, "std")  for c in metric_cols},
                       ).reset_index()
        p2 = os.path.join(out, "summary_method_overall.csv")
        agg2.to_csv(p2, index=False)
        print(f">>> {p2}")

    # --- Table C: Prior degradation (PRCD-MAP methods only) ---
    df_prior = df[df["method"].isin(prcd_methods + ["PRCD-MAP(uniform)"])]
    if not df_prior.empty and "prior_acc" in df_prior.columns:
        g3 = df_prior.groupby(["method", "prior_acc"])
        agg3 = g3.agg(
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
            f1_mean=("f1_opt", "mean"), f1_std=("f1_opt", "std"),
            tau_mean=("tau", "mean"), tau_std=("tau", "std"),
        ).reset_index()
        p3 = os.path.join(out, "summary_prior_degradation.csv")
        agg3.to_csv(p3, index=False)
        print(f">>> {p3}")

    # --- Console summary ---
    print("\n" + "=" * 72)
    print("Overall mean by method (across all benchmarks)")
    print("=" * 72)
    overall = df.groupby("method")[metric_cols].mean()
    print(overall.round(4).to_string())
    print()

    # Per-benchmark summary
    for bench in df["bench"].unique():
        sub = df[df["bench"] == bench]
        print(f"\n--- {bench} ---")
        overall_b = sub.groupby("method")[metric_cols].mean()
        print(overall_b.round(4).to_string())


# ====================================================================
# 11. Figures
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
    _fig_benchmark_comparison(df, out)
    _fig_prior_degradation_real(df, out)
    _fig_tau_vs_prior_real(df, out)
    _fig_per_benchmark_bars(df, out)


def _fig_benchmark_comparison(df, out):
    """
    Figure 1: Grouped bar chart comparing methods across benchmarks.
    For PRCD-MAP with prior, use acc=0.7 (or closest available).
    """
    # Select representative rows
    baseline_methods = ["DYNOTEARS", "PCMCI+", "VARLiNGAM"]
    prcd_methods = ["PRCD-MAP(learn_tau)", "PRCD-MAP(uniform)"]

    df_base = df[df["method"].isin(baseline_methods)]
    df_prcd_u = df[df["method"] == "PRCD-MAP(uniform)"]
    df_prcd_l = df[df["method"] == "PRCD-MAP(learn_tau)"]
    if not df_prcd_l.empty and "prior_acc" in df_prcd_l.columns:
        avail = df_prcd_l["prior_acc"].dropna().unique()
        if len(avail) > 0:
            best = avail[np.argmin(np.abs(avail - 0.7))]
            df_prcd_l = df_prcd_l[df_prcd_l["prior_acc"] == best]

    df_plot = pd.concat([df_base, df_prcd_u, df_prcd_l], ignore_index=True)

    benchmarks = sorted(df_plot["bench"].unique())
    methods = sorted(df_plot["method"].unique())
    n_b, n_m = len(benchmarks), len(methods)
    if n_b == 0 or n_m == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for metric, ax, yl in [("auroc", axes[0], "AUROC"),
                            ("f1_opt", axes[1], "F1")]:
        x = np.arange(n_b)
        w = 0.8 / max(n_m, 1)
        for i, m in enumerate(methods):
            vals, errs = [], []
            for b in benchmarks:
                s = df_plot[(df_plot["method"] == m) & (df_plot["bench"] == b)]
                vals.append(s[metric].mean() if len(s) else 0)
                errs.append(s[metric].std() if len(s) else 0)
            ax.bar(x + i * w, vals, w, yerr=errs,
                   label=m, color=COLORS.get(m, "grey"), alpha=0.85, capsize=2)
        ax.set_xlabel("Benchmark")
        ax.set_ylabel(yl)
        ax.set_xticks(x + w * (n_m - 1) / 2)
        ax.set_xticklabels([b.replace("Lorenz96_", "L96_") for b in benchmarks],
                           rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle("Method Comparison Across Real-World Benchmarks",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig1_benchmark_comparison"))


def _fig_prior_degradation_real(df, out):
    """
    Figure 2: F1 and AUROC vs prior accuracy on real benchmarks.
    Shows all methods, baselines as horizontal lines.
    """
    prcd_df = df[df["method"].isin(["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)"])]
    if prcd_df.empty or "prior_acc" not in prcd_df.columns:
        return
    if prcd_df["prior_acc"].dropna().nunique() < 3:
        return

    # Baselines for reference
    baseline_df = df[df["method"].isin(["DYNOTEARS", "PCMCI+", "VARLiNGAM",
                                         "PRCD-MAP(uniform)"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for metric, ax, yl in [("f1_opt", axes[0], "F1 (optimal threshold)"),
                            ("auroc",  axes[1], "AUROC")]:
        # PRCD-MAP curves
        for m in ["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)"]:
            sub = prcd_df[prcd_df["method"] == m]
            agg = sub.groupby("prior_acc").agg(
                y=(metric, "mean"), e=(metric, "std")).reset_index()
            ax.errorbar(agg["prior_acc"], agg["y"], yerr=agg["e"],
                        label=m, color=COLORS.get(m, "grey"),
                        marker=MARKERS.get(m, "x"),
                        linewidth=2, markersize=7, capsize=3)

        # Baselines as horizontal lines
        for m in baseline_df["method"].unique():
            sub = baseline_df[baseline_df["method"] == m]
            mean_val = sub[metric].mean()
            ax.axhline(y=mean_val, color=COLORS.get(m, "grey"),
                       linestyle="--", alpha=0.7, linewidth=1.5,
                       label=f"{m} (avg)")

        ax.set_xlabel("Prior Accuracy", fontsize=12)
        ax.set_ylabel(yl, fontsize=12)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Prior Degradation on Real-World Benchmarks",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig2_prior_degradation_real"))


def _fig_tau_vs_prior_real(df, out):
    """
    Figure 3: Learned tau vs prior accuracy (dual y-axis with F1).
    """
    sub = df[df["method"] == "PRCD-MAP(learn_tau)"]
    if sub.empty or "prior_acc" not in sub.columns:
        return
    sub = sub.dropna(subset=["prior_acc"])
    if sub["prior_acc"].nunique() < 3:
        return

    agg = sub.groupby("prior_acc").agg(
        tau_m=("tau", "mean"), tau_s=("tau", "std"),
        f1_m=("f1_opt", "mean"), f1_s=("f1_opt", "std")).reset_index()

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    c1, c2 = "#E74C3C", "#2C3E50"

    ax1.errorbar(agg["prior_acc"], agg["tau_m"], yerr=agg["tau_s"],
                 color=c1, marker="o", lw=2.5, ms=8, capsize=4,
                 label="Learned $\\tau$")
    ax1.set_xlabel("Prior Accuracy", fontsize=13)
    ax1.set_ylabel("Learned Temperature $\\tau$", fontsize=13, color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2 = ax1.twinx()
    ax2.errorbar(agg["prior_acc"], agg["f1_m"], yerr=agg["f1_s"],
                 color=c2, marker="s", lw=2.5, ms=8, capsize=4, ls="--",
                 label="F1")
    ax2.set_ylabel("F1 (optimal threshold)", fontsize=13, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=11)
    ax1.set_title("Temperature $\\tau$ Adaptation vs Prior Quality\n"
                  "(Real-World Benchmarks)", fontsize=14, fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    _save(os.path.join(out, "fig3_tau_vs_prior_real"))


def _fig_per_benchmark_bars(df, out):
    """
    Figure 4: Per-benchmark grouped bar chart (AUROC).
    One subplot per benchmark.
    """
    benchmarks = sorted(df["bench"].unique())
    if len(benchmarks) < 2:
        return

    # For PRCD-MAP, use acc=0.7
    prcd_l = df[df["method"] == "PRCD-MAP(learn_tau)"]
    if not prcd_l.empty and "prior_acc" in prcd_l.columns:
        avail = prcd_l["prior_acc"].dropna().unique()
        if len(avail) > 0:
            best = avail[np.argmin(np.abs(avail - 0.7))]
            prcd_l = prcd_l[prcd_l["prior_acc"] == best]

    others = df[~df["method"].isin(["PRCD-MAP(learn_tau)", "PRCD-MAP(fixed_tau)"])]
    df_plot = pd.concat([others, prcd_l], ignore_index=True)

    methods = sorted(df_plot["method"].unique())
    n_m = len(methods)
    n_b = len(benchmarks)

    ncols = min(3, n_b)
    nrows = (n_b + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx // ncols][idx % ncols]
        sub = df_plot[df_plot["bench"] == bench]
        x = np.arange(1)
        w = 0.8 / max(n_m, 1)
        for i, m in enumerate(methods):
            s = sub[sub["method"] == m]
            v = s["auroc"].mean() if len(s) else 0
            e = s["auroc"].std() if len(s) else 0
            ax.bar(i * w, v, w, yerr=e,
                   color=COLORS.get(m, "grey"), alpha=0.85, capsize=2)
        ax.set_xticks([i * w for i in range(n_m)])
        ax.set_xticklabels([m.replace("PRCD-MAP", "PRCD") for m in methods],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("AUROC")
        ax.set_title(bench.replace("Lorenz96_", "L96_"), fontsize=10)
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    # Hide empty subplots
    for idx in range(n_b, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("AUROC per Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig4_per_benchmark"))


# ====================================================================
# 12. Main Entry Point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Real-World Benchmark Datasets")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (Lorenz-96 only, small)")
    parser.add_argument("--full", action="store_true",
                        help="Full NeurIPS sweep")
    parser.add_argument("--bench", nargs="+", type=str, default=None,
                        choices=["lorenz96", "causaltime", "dream4",
                                 "netsim", "electricity"],
                        help="Specific benchmark(s) to run")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--prior-accs", nargs="+", type=float, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--no-pcmci", action="store_true")
    parser.add_argument("--no-varlingam", action="store_true")
    parser.add_argument("--no-dynotears", action="store_true")
    parser.add_argument("--causaltime-dir", type=str, default=None)
    parser.add_argument("--dream4-dir", type=str, default=None)
    parser.add_argument("--netsim-dir", type=str, default=None)
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
    if args.prior_accs:   cfg.prior_accs    = args.prior_accs
    if args.K is not None: cfg.K            = args.K
    if args.bench:        cfg.benchmarks    = args.bench
    if args.no_pcmci:     cfg.do_pcmci      = False
    if args.no_varlingam: cfg.do_varlingam  = False
    if args.no_dynotears: cfg.do_dynotears  = False
    if args.causaltime_dir: cfg.causaltime_dir = args.causaltime_dir
    if args.dream4_dir:   cfg.dream4_dir    = args.dream4_dir
    if args.netsim_dir:   cfg.netsim_dir    = args.netsim_dir

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 68)
    print(" Experiment 2: Standard Public Real-World Benchmarks")
    print("=" * 68)
    print(f"  benchmarks   = {cfg.benchmarks}")
    print(f"  K            = {cfg.K}")
    print(f"  prior_accs   = {cfg.prior_accs}")
    print(f"  seeds        = {cfg.seeds}")
    print(f"  baselines    = DYNOTEARS={cfg.do_dynotears}, "
          f"PCMCI+={cfg.do_pcmci and HAS_TIGRAMITE}, "
          f"VARLiNGAM={cfg.do_varlingam and HAS_LINGAM}")
    print(f"  output       = {cfg.output_dir}")
    print("=" * 68)

    t_global = time.time()
    all_dfs = []
    method_kwargs = dict(
        lambda1=cfg.lambda1, lambda2=cfg.lambda2,
        max_iter=cfg.max_iter, inner_iter=cfg.inner_iter, lr=cfg.lr,
        do_dynotears=cfg.do_dynotears,
        do_pcmci=cfg.do_pcmci,
        do_varlingam=cfg.do_varlingam,
    )

    # ============================================
    # Benchmark 1: Lorenz-96
    # ============================================
    if "lorenz96" in cfg.benchmarks:
        print("\n" + "=" * 50)
        print(">>> Benchmark 1: Lorenz-96 Dynamical System")
        print("=" * 50)
        for d in cfg.lorenz_dims:
            for T in cfg.lorenz_Ts:
                for F_val in cfg.lorenz_Fs:
                    print(f"\n  Generating Lorenz-96: d={d}, T={T}, F={F_val}")
                    for seed in cfg.seeds:
                        X, B_true = generate_lorenz96(d=d, T=T, F=F_val,
                                                       seed=seed)
                        if not np.all(np.isfinite(X)):
                            warnings.warn(f"Lorenz96 d={d} T={T} seed={seed}: "
                                          f"non-finite, skipping")
                            continue
                        name = f"Lorenz96_d{d}_T{T}_F{F_val}"
                        df = run_single_benchmark(
                            bench_name=name, X=X, B_true=B_true,
                            K=cfg.K, prior_accs=cfg.prior_accs,
                            seeds=[seed], **method_kwargs)
                        all_dfs.append(df)

    # ============================================
    # Benchmark 2: CausalTime
    # ============================================
    if "causaltime" in cfg.benchmarks:
        print("\n" + "=" * 50)
        print(">>> Benchmark 2: CausalTime (Cheng et al., ICLR 2024)")
        print("=" * 50)
        datasets = discover_causaltime_datasets(cfg.causaltime_dir)
        if datasets:
            print(f"  Found {len(datasets)} CausalTime dataset(s)")
            for ds in datasets:
                X, B_true = load_causaltime_dataset(ds["data_path"],
                                                     ds["graph_path"])
                if X is None:
                    continue
                df = run_single_benchmark(
                    bench_name=ds["name"], X=X, B_true=B_true,
                    K=cfg.K, prior_accs=cfg.prior_accs,
                    seeds=cfg.seeds, **method_kwargs)
                all_dfs.append(df)
        else:
            print(f"  CausalTime data not found at {cfg.causaltime_dir}")
            print("  To use CausalTime benchmark:")
            print("    1. git clone https://github.com/causaltime/causaltime")
            print("    2. Place dataset folders in ./data/causaltime/")
            print("       Each subfolder should contain data.npy + dag.npy")
            print("    3. Or use --causaltime-dir to point to your data")

    # ============================================
    # Benchmark 3: DREAM4
    # ============================================
    if "dream4" in cfg.benchmarks:
        print("\n" + "=" * 50)
        print(">>> Benchmark 3: DREAM4 Gene Regulatory Networks")
        print("=" * 50)
        datasets = discover_dream4_datasets(cfg.dream4_dir)
        if datasets:
            print(f"  Found {len(datasets)} DREAM4 dataset(s)")
            for ds in datasets:
                X, B_true = load_dream4_dataset(ds["data_path"],
                                                 ds["graph_path"])
                if X is None:
                    continue
                df = run_single_benchmark(
                    bench_name=ds["name"], X=X, B_true=B_true,
                    K=cfg.K, prior_accs=cfg.prior_accs,
                    seeds=cfg.seeds, **method_kwargs)
                all_dfs.append(df)
        else:
            print(f"  DREAM4 data not found at {cfg.dream4_dir}")
            print("  To use DREAM4 benchmark:")
            print("    1. Download from https://gnw.sourceforge.net/")
            print("       (DREAM4 In-Silico Network Challenge)")
            print("    2. Place files in ./data/dream4/")
            print("       Expected: insilico_size100_*_timeseries.tsv")
            print("                 insilico_size100_*_goldstandard.tsv")
            print("    3. Or use --dream4-dir to point to your data")

    # ============================================
    # Benchmark 4: Netsim fMRI
    # ============================================
    if "netsim" in cfg.benchmarks:
        print("\n" + "=" * 50)
        print(">>> Benchmark 4: Netsim fMRI")
        print("=" * 50)
        datasets = discover_netsim_datasets(cfg.netsim_dir)
        if datasets:
            print(f"  Found {len(datasets)} Netsim dataset(s)")
            for ds in datasets:
                X, B_true = load_netsim_dataset(ds["data_path"],
                                                 ds["graph_path"])
                if X is None:
                    continue
                df = run_single_benchmark(
                    bench_name=ds["name"], X=X, B_true=B_true,
                    K=cfg.K, prior_accs=cfg.prior_accs,
                    seeds=cfg.seeds, **method_kwargs)
                all_dfs.append(df)
        else:
            print(f"  Netsim data not found at {cfg.netsim_dir}")
            print("  To use Netsim benchmark:")
            print("    1. Download Netsim dataset (Smith et al., 2011)")
            print("    2. Place files in ./data/netsim/")
            print("       Expected: sim*_timeseries.npy + sim*_network.npy")
            print("       Or: sim*.mat files with 'ts' and 'net' keys")
            print("    3. Or use --netsim-dir to point to your data")

    # ============================================
    # Case Study: Electricity Consumption
    # ============================================
    if "electricity" in cfg.benchmarks:
        print("\n" + "=" * 50)
        print(">>> Case Study: Electricity Consumption")
        print("=" * 50)
        X_elec, P_prior, _, col_names = load_electricity_case_study(
            cfg.electricity_xlsx, cfg.electricity_prior)
        if X_elec is not None and P_prior is not None:
            run_electricity_case_study(
                X_elec, P_prior, col_names, K=cfg.K,
                seeds=cfg.seeds, output_dir=cfg.output_dir,
                do_dynotears=cfg.do_dynotears,
                do_pcmci=cfg.do_pcmci,
                do_varlingam=cfg.do_varlingam,
            )
        else:
            print("  Electricity data not found or failed to load.")
            print(f"  Expected: {cfg.electricity_xlsx}")
            print(f"  Prior:    {cfg.electricity_prior}")

    # ============================================
    # Aggregate Results
    # ============================================
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        csv_path = os.path.join(cfg.output_dir, "exp2_full_results.csv")
        df_all.to_csv(csv_path, index=False)
        print(f"\n>>> Saved {len(df_all)} rows -> {csv_path}")

        generate_summaries(df_all, cfg.output_dir)
        generate_figures(df_all, cfg.output_dir)
    else:
        print("\n>>> No benchmark results to aggregate.")

    elapsed = time.time() - t_global
    print(f"\n>>> Experiment 2 complete in {elapsed:.1f}s")
    print(f">>> Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
