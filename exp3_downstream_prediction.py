"""
=============================================================================
Experiment 3 — Downstream Prediction via Causal Feature Selection
=============================================================================
NeurIPS-grade evaluation proving that PRCD-MAP's discovered causal graph
yields better features for downstream forecasting than competing methods.

Key upgrades over the original exp2_forecasting.py:
  1. Feature selection from EVERY baseline's graph (DYNOTEARS, PCMCI+,
     VARLiNGAM), not just Pearson correlation.
  2. Multiple prediction targets (not just one).
  3. Multiple predictor architectures: LSTM, linear VAR, Transformer.
  4. Results on both the electricity dataset AND Lorenz-96 benchmark.
  5. Comprehensive reporting with error bars across seeds.

Baselines for feature selection:
  - All Features (no selection)
  - Pearson correlation top-M
  - PRCD-MAP (learn_tau, with domain prior) top-M
  - PRCD-MAP (uniform prior) top-M
  - DYNOTEARS top-M
  - PCMCI+ top-M
  - VARLiNGAM top-M

Predictor architectures:
  - LSTM (1-layer, hidden=32)
  - Linear VAR (autoregressive)
  - Transformer (lightweight PatchTST-style)

Usage:
  python exp3_downstream_prediction.py                      # default
  python exp3_downstream_prediction.py --quick              # quick test
  python exp3_downstream_prediction.py --full               # full sweep
  python exp3_downstream_prediction.py --bench electricity  # electricity only
  python exp3_downstream_prediction.py --bench lorenz96     # Lorenz-96 only
  python exp3_downstream_prediction.py --seeds 0 1 2 3 4
  python exp3_downstream_prediction.py --top-m 4 6 8
=============================================================================
"""

import os, sys, time, warnings, argparse, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# 2. Data Loading
# ====================================================================

# ---- 2a. Electricity Dataset ----
def load_electricity(excel_path: str, prior_csv_path: str,
                     diff_periods: int = 7, train_ratio: float = 0.8):
    """
    Load electricity consumption dataset.
    Returns: df_diff (DataFrame), P_prior (ndarray), split_idx (int), col_names
    """
    df_ts = pd.read_excel(excel_path, index_col=0)
    df_diff = df_ts.diff(periods=diff_periods).dropna()
    split = int(len(df_diff) * train_ratio)
    col_names = df_diff.columns.tolist()
    d = len(col_names)

    P_prior = None
    if os.path.exists(prior_csv_path):
        P_prior = pd.read_csv(prior_csv_path, index_col=0).values
        if P_prior.shape != (d, d):
            warnings.warn(f"Prior shape {P_prior.shape} != ({d},{d}), using uniform")
            P_prior = np.full((d, d), 0.5)
    else:
        P_prior = np.full((d, d), 0.5)

    return df_diff, P_prior, split, col_names


# ---- 2b. Lorenz-96 System ----
def _lorenz96_rhs(t, x, F):
    d = len(x)
    dxdt = np.zeros(d)
    for i in range(d):
        dxdt[i] = (x[(i + 1) % d] - x[(i - 2) % d]) * x[(i - 1) % d] - x[i] + F
    return dxdt


def lorenz96_ground_truth(d: int) -> np.ndarray:
    """B[j,i]=1 means j causally affects i."""
    B = np.zeros((d, d), dtype=int)
    for i in range(d):
        B[(i - 2) % d, i] = 1
        B[(i - 1) % d, i] = 1
        B[(i + 1) % d, i] = 1
    return B


def generate_lorenz96(d: int = 10, T: int = 2000, F: float = 10.0,
                      dt: float = 0.05, seed: int = 0):
    """
    Returns: X (T, d), B_true (d, d)
    """
    rng = np.random.default_rng(seed)
    x0 = F * np.ones(d) + rng.normal(0, 0.01, d)
    burn_in = 500
    total = burn_in + T
    t_span = (0, total * dt)
    t_eval = np.arange(0, total * dt, dt)

    sol = solve_ivp(_lorenz96_rhs, t_span, x0, args=(F,),
                    t_eval=t_eval, method="RK45", max_step=dt)
    if sol.status != 0:
        warnings.warn(f"Lorenz-96 integration failed: {sol.message}")
        return np.zeros((T, d)), lorenz96_ground_truth(d)

    X_full = sol.y.T[burn_in:][:T]
    if len(X_full) < T:
        pad = np.zeros((T - len(X_full), d))
        X_full = np.vstack([X_full, pad])

    X_std = standardize(X_full)
    return X_std, lorenz96_ground_truth(d)


def gen_prior_from_truth(B_true: np.ndarray, acc: float,
                         seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = B_true.shape[0]
    P = np.full((d, d), 0.5)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            true_edge = B_true[i, j] > 0
            agree = rng.random() < acc
            if agree:
                P[i, j] = rng.uniform(0.75, 0.95) if true_edge else rng.uniform(0.05, 0.25)
            else:
                P[i, j] = rng.uniform(0.05, 0.25) if true_edge else rng.uniform(0.75, 0.95)
    return P


# ====================================================================
# 3. Causal Discovery Methods (for feature selection)
# ====================================================================
def combine_W0_Wk(W0: np.ndarray, Wk_list: list) -> np.ndarray:
    """Max absolute value across W0 and all Wk, preserving sign from max-abs."""
    mats = [W0] + list(Wk_list)
    abs_stack = np.stack([np.abs(m) for m in mats], axis=0)
    combined = abs_stack.max(axis=0)
    return combined


def select_top_parents(W_combined: np.ndarray, target_idx: int,
                       top_m: int, col_names: list) -> list:
    """
    Select top-M parent features of target_idx from a weight matrix.
    W[i, j] = effect of i on j. Parents of target_idx = column target_idx.
    Returns list of column names (including target itself).
    """
    d = W_combined.shape[0]
    w_in = np.abs(W_combined[:, target_idx]).copy()
    w_in[target_idx] = 0.0  # exclude self

    n_parents = max(0, top_m - 1)  # reserve 1 slot for target itself
    n_parents = min(n_parents, d - 1)

    if n_parents > 0:
        parent_idx = np.argsort(-w_in)[:n_parents]
    else:
        parent_idx = []

    parent_names = [col_names[i] for i in parent_idx]
    target_name = col_names[target_idx]

    # Always include target as a feature (for autoregressive component)
    feats = parent_names + [target_name]
    # Remove duplicates while preserving order
    seen = set()
    unique_feats = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            unique_feats.append(f)
    return unique_feats


# ---- 3a. PRCD-MAP ----
def run_prcd_map_discovery(X_train: np.ndarray, P_prior: np.ndarray,
                           d: int, K: int,
                           lambda1: float = 0.003, lambda2: float = 0.01,
                           learn_tau: bool = True,
                           max_iter: int = 25, inner_iter: int = 500,
                           lr: float = 1e-2, seed: int = 0):
    """Run PRCD-MAP on training data, return combined weight matrix and tau."""
    set_seed(seed)
    # Standardize within training
    mu = X_train.mean(0)
    sd = X_train.std(0)
    sd[sd == 0] = 1.0
    Xn = (X_train - mu) / sd

    X_t, X_lags = make_lag_tensors(Xn, K)
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau, tau0=1.0,
        tau_min=0.1, tau_max=10.0,
    )
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=2.0, tol=1e-6,
        verbose=False, postprocess=False,
    )
    W_combined = combine_W0_Wk(W0, Wk)
    return W_combined, float(tau)


# ---- 3b. DYNOTEARS ----
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


def run_dynotears_discovery(X_train: np.ndarray, d: int, K: int,
                            lam: float = 0.01, max_outer: int = 25,
                            inner: int = 150, lr: float = 1e-2,
                            seed: int = 0):
    """Run DYNOTEARS, return combined weight matrix."""
    set_seed(seed)
    mu = X_train.mean(0)
    sd = X_train.std(0)
    sd[sd == 0] = 1.0
    Xn = (X_train - mu) / sd

    X_t, X_lags = make_lag_tensors(Xn, K)
    m = _DYNOTEARS(d, K, lam)
    opt_ = optim.Adam(m.parameters(), lr=lr)
    rho, alpha = 1.0, 0.0

    for _ in range(max_outer):
        for __ in range(inner):
            loss, h = m.loss(X_t, X_lags, rho, alpha)
            opt_.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt_.step()
        h_val = float(m._h().detach())
        if abs(h_val) < 1e-6:
            break
        alpha += rho * h_val
        rho *= 2.0

    with torch.no_grad():
        W0 = m._adj().cpu().numpy()
        Wk = [w.detach().cpu().numpy() for w in m.Wk]
    return combine_W0_Wk(W0, Wk)


# ---- 3c. PCMCI+ ----
def run_pcmci_discovery(X_train: np.ndarray, d: int, K: int,
                        alpha_level: float = 0.05, seed: int = 0):
    """Run PCMCI+, return combined weight matrix or None."""
    if not HAS_TIGRAMITE:
        return None
    try:
        mu = X_train.mean(0)
        sd = X_train.std(0)
        sd[sd == 0] = 1.0
        Xn = (X_train - mu) / sd

        df = pp.DataFrame(Xn, var_names=[f"V{i}" for i in range(d)])
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

        return combine_W0_Wk(W0, Wk)
    except Exception as e:
        warnings.warn(f"PCMCI+ failed: {e}")
        return None


# ---- 3d. VARLiNGAM ----
def run_varlingam_discovery(X_train: np.ndarray, d: int, K: int,
                            seed: int = 0):
    """Run VARLiNGAM, return combined weight matrix or None."""
    if not HAS_LINGAM:
        return None
    try:
        mu = X_train.mean(0)
        sd = X_train.std(0)
        sd[sd == 0] = 1.0
        Xn = (X_train - mu) / sd

        model = lingam.VARLiNGAM(lags=K, random_state=seed)
        model.fit(Xn)
        B0 = model.adjacency_matrices_[0]
        W0 = B0.T
        np.fill_diagonal(W0, 0.0)
        Wk = []
        for k in range(1, K + 1):
            Bk = model.adjacency_matrices_[k]
            Wk.append(Bk.T)
        return combine_W0_Wk(W0, Wk)
    except Exception as e:
        warnings.warn(f"VARLiNGAM failed: {e}")
        return None


# ---- 3e. Pearson Correlation ----
def run_pearson_selection(X_train: np.ndarray, target_idx: int,
                          top_m: int, col_names: list) -> list:
    """Select top-M features by Pearson correlation with target."""
    d = X_train.shape[1]
    target_name = col_names[target_idx]
    corrs = {}
    for j in range(d):
        if j == target_idx:
            continue
        r, _ = pearsonr(X_train[:, j], X_train[:, target_idx])
        corrs[j] = abs(r)

    n_parents = max(0, top_m - 1)
    sorted_idx = sorted(corrs, key=corrs.get, reverse=True)[:n_parents]
    feats = [col_names[i] for i in sorted_idx] + [target_name]
    return feats


# ====================================================================
# 4. Predictor Architectures
# ====================================================================

# ---- 4a. LSTM Forecaster ----
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


# ---- 4b. Linear VAR Forecaster ----
class VARForecaster(nn.Module):
    """
    Simple linear VAR: predicts y_{t+1} from flattened [x_t, x_{t-1}, ..., x_{t-L+1}].
    """
    def __init__(self, input_dim: int, seq_len: int):
        super().__init__()
        self.fc = nn.Linear(input_dim * seq_len, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch = x.shape[0]
        return self.fc(x.reshape(batch, -1))


# ---- 4c. Lightweight Transformer Forecaster (PatchTST-style) ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerForecaster(nn.Module):
    """
    Lightweight Transformer encoder for time series forecasting.
    Input: (batch, seq_len, input_dim) -> Output: (batch, 1)
    """
    def __init__(self, input_dim: int, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.input_proj(x)          # (batch, seq_len, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)             # (batch, seq_len, d_model)
        h = h[:, -1, :]                 # last time step
        return self.fc(h)               # (batch, 1)


# ====================================================================
# 5. Forecasting Pipeline
# ====================================================================
def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Create sliding window sequences."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def run_forecast(
    df_or_X,
    split: int,
    feature_cols_or_idx,
    target_col_or_idx,
    col_names: list,
    seq_len: int = 7,
    max_epochs: int = 50,
    lr: float = 3e-3,
    hidden: int = 32,
    seed: int = 0,
    patience: int = 8,
    predictor: str = "lstm",
    batch_size: int = 64,
) -> Tuple[float, float]:
    """
    Run a forecasting experiment with the specified predictor architecture.

    Args:
        df_or_X: DataFrame or ndarray (T, d)
        split: train/test split index
        feature_cols_or_idx: list of column names or column indices
        target_col_or_idx: target column name or index
        col_names: list of all column names
        predictor: "lstm" | "var" | "transformer"
    Returns:
        (rmse, mae)
    """
    set_seed(seed)

    # Handle DataFrame or ndarray input
    if isinstance(df_or_X, pd.DataFrame):
        X_raw = df_or_X[feature_cols_or_idx].values
        y_raw = df_or_X[[target_col_or_idx]].values
    else:
        # ndarray input with column indices
        if isinstance(feature_cols_or_idx[0], str):
            feat_idx = [col_names.index(c) for c in feature_cols_or_idx]
        else:
            feat_idx = feature_cols_or_idx
        if isinstance(target_col_or_idx, str):
            tgt_idx = col_names.index(target_col_or_idx)
        else:
            tgt_idx = target_col_or_idx
        X_raw = df_or_X[:, feat_idx]
        y_raw = df_or_X[:, [tgt_idx]]

    # Train/test split
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

    # Validation split from training (for early stopping)
    val_ratio = 0.2
    val_split = int(len(X_train_raw) * (1 - val_ratio))
    X_tr_raw, X_val_raw = X_train_raw[:val_split], X_train_raw[val_split:]
    y_tr_raw, y_val_raw = y_train_raw[:val_split], y_train_raw[val_split:]

    # Standardization (fit on training only)
    sx = StandardScaler()
    sy = StandardScaler()
    X_tr = sx.fit_transform(X_tr_raw)
    X_val = sx.transform(X_val_raw)
    X_test = sx.transform(X_test_raw)
    y_tr = sy.fit_transform(y_tr_raw)
    y_val = sy.transform(y_val_raw)
    y_test = sy.transform(y_test_raw)

    # Create sequences
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

    input_dim = Xtr.shape[2]

    # Build model
    if predictor == "lstm":
        model = LSTMForecaster(input_dim=input_dim, hidden_dim=hidden)
    elif predictor == "var":
        model = VARForecaster(input_dim=input_dim, seq_len=seq_len)
    elif predictor == "transformer":
        d_model = max(16, min(64, input_dim * 4))
        # nhead must divide d_model
        nhead = 4 if d_model % 4 == 0 else (2 if d_model % 2 == 0 else 1)
        model = TransformerForecaster(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=2, dim_feedforward=d_model * 2, dropout=0.1)
    else:
        raise ValueError(f"Unknown predictor: {predictor}")

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                        shuffle=True)
    opt_ = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        for bx, by in loader:
            opt_.zero_grad()
            pred = model(bx)
            loss = crit(pred, by)
            loss.backward()
            opt_.step()

        model.eval()
        with torch.no_grad():
            vpred = model(Xva)
            vloss = float(crit(vpred, yva).item())

        if vloss + 1e-9 < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(Xte).numpy()
        yte_scaled = yte.numpy()

    pred = sy.inverse_transform(pred_scaled).flatten()
    ytrue = sy.inverse_transform(yte_scaled).flatten()

    rmse = float(np.sqrt(mean_squared_error(ytrue, pred)))
    mae = float(mean_absolute_error(ytrue, pred))
    return rmse, mae


# ====================================================================
# 6. Main Experiment Runner
# ====================================================================
@dataclass
class Cfg:
    # Data
    electricity_xlsx:   str   = r"E:\electricity\0227test.xlsx"
    electricity_prior:  str   = r"E:\electricity\Auto_Generated_Prior.csv"
    # Lorenz-96
    lorenz_d:           int   = 10
    lorenz_T:           int   = 2000
    lorenz_F:           float = 10.0
    lorenz_prior_acc:   float = 0.7
    # Experiment params
    K:                  int   = 1
    top_ms:             List[int] = field(default_factory=lambda: [4, 6])
    seq_len:            int   = 7
    seeds:              List[int] = field(default_factory=lambda: list(range(5)))
    predictors:         List[str] = field(default_factory=lambda: [
        "lstm", "var", "transformer"])
    max_epochs:         int   = 50
    lr:                 float = 3e-3
    hidden:             int   = 32
    patience:           int   = 8
    # Causal discovery params
    cd_lambda1:         float = 0.003
    cd_lambda2:         float = 0.01
    cd_max_iter:        int   = 25
    cd_inner_iter:      int   = 500
    cd_lr:              float = 1e-2
    # Baselines
    do_dynotears:       bool  = True
    do_pcmci:           bool  = True
    do_varlingam:       bool  = True
    # Benchmarks
    benchmarks:         List[str] = field(default_factory=lambda: [
        "electricity", "lorenz96"])
    # Electricity-specific: target columns (Chinese names)
    elec_targets:       List[str] = field(default_factory=lambda: [
        "大工业电量", "居民生活", "商业用电"])
    # Lorenz-96: how many target variables to predict
    lorenz_n_targets:   int   = 3
    # Output
    output_dir:         str   = "exp3_results"


def cfg_quick():
    return Cfg(
        seeds=list(range(3)),
        top_ms=[4],
        predictors=["lstm"],
        max_epochs=20,
        cd_max_iter=15, cd_inner_iter=100,
        do_pcmci=False, do_varlingam=False,
        benchmarks=["electricity"],
        elec_targets=["大工业电量"],
        lorenz_n_targets=1,
    )


def cfg_full():
    return Cfg(
        seeds=list(range(10)),
        top_ms=[3, 4, 6, 8],
        predictors=["lstm", "var", "transformer"],
        max_epochs=80,
        benchmarks=["electricity", "lorenz96"],
        elec_targets=["大工业电量", "居民生活", "商业用电", "非普工业"],
        lorenz_n_targets=5,
    )


def discover_causal_graphs(
    X_train: np.ndarray,
    P_prior: np.ndarray,
    d: int, K: int,
    cfg: Cfg,
    seed: int,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Run all causal discovery methods on X_train.
    Returns dict: method_name -> combined weight matrix (d x d).
    """
    graphs = {}

    # PRCD-MAP (with domain prior, learn tau)
    try:
        W, tau = run_prcd_map_discovery(
            X_train, P_prior, d, K,
            lambda1=cfg.cd_lambda1, lambda2=cfg.cd_lambda2,
            learn_tau=True,
            max_iter=cfg.cd_max_iter, inner_iter=cfg.cd_inner_iter,
            lr=cfg.cd_lr, seed=seed)
        graphs["PRCD-MAP(prior)"] = W
    except Exception as e:
        warnings.warn(f"PRCD-MAP(prior) failed seed={seed}: {e}")
        graphs["PRCD-MAP(prior)"] = None

    # PRCD-MAP (uniform prior = pure data-driven)
    try:
        P_unif = np.full((d, d), 0.5)
        W, tau = run_prcd_map_discovery(
            X_train, P_unif, d, K,
            lambda1=cfg.cd_lambda1, lambda2=cfg.cd_lambda2,
            learn_tau=True,
            max_iter=cfg.cd_max_iter, inner_iter=cfg.cd_inner_iter,
            lr=cfg.cd_lr, seed=seed)
        graphs["PRCD-MAP(uniform)"] = W
    except Exception as e:
        warnings.warn(f"PRCD-MAP(uniform) failed seed={seed}: {e}")
        graphs["PRCD-MAP(uniform)"] = None

    # DYNOTEARS
    if cfg.do_dynotears:
        try:
            W = run_dynotears_discovery(
                X_train, d, K,
                lam=cfg.cd_lambda1,
                max_outer=cfg.cd_max_iter, inner=cfg.cd_inner_iter,
                lr=cfg.cd_lr, seed=seed)
            graphs["DYNOTEARS"] = W
        except Exception as e:
            warnings.warn(f"DYNOTEARS failed seed={seed}: {e}")
            graphs["DYNOTEARS"] = None

    # PCMCI+
    if cfg.do_pcmci and HAS_TIGRAMITE and d <= 80:
        try:
            W = run_pcmci_discovery(X_train, d, K, seed=seed)
            graphs["PCMCI+"] = W
        except Exception as e:
            warnings.warn(f"PCMCI+ failed seed={seed}: {e}")
            graphs["PCMCI+"] = None

    # VARLiNGAM
    if cfg.do_varlingam and HAS_LINGAM:
        try:
            W = run_varlingam_discovery(X_train, d, K, seed=seed)
            graphs["VARLiNGAM"] = W
        except Exception as e:
            warnings.warn(f"VARLiNGAM failed seed={seed}: {e}")
            graphs["VARLiNGAM"] = None

    return graphs


def run_electricity_experiment(cfg: Cfg) -> pd.DataFrame:
    """
    Run the full downstream prediction experiment on electricity data.
    """
    print("\n" + "=" * 60)
    print(">>> Electricity: Downstream Prediction via Causal Features")
    print("=" * 60)

    df_diff, P_prior, split, col_names = load_electricity(
        cfg.electricity_xlsx, cfg.electricity_prior)
    d = len(col_names)
    X_all = df_diff.values

    # Verify targets exist
    valid_targets = [t for t in cfg.elec_targets if t in col_names]
    if not valid_targets:
        warnings.warn("No valid electricity targets found!")
        return pd.DataFrame()

    print(f"  d={d}, T={len(df_diff)}, split={split}")
    print(f"  targets: {valid_targets}")
    print(f"  top_ms: {cfg.top_ms}, predictors: {cfg.predictors}")

    X_train = X_all[:split]
    all_rows = []

    for seed in cfg.seeds:
        print(f"\n  --- Seed {seed} ---")

        # 1. Discover causal graphs (once per seed, shared across targets)
        t0 = time.time()
        graphs = discover_causal_graphs(X_train, P_prior, d, cfg.K, cfg, seed)
        cd_time = time.time() - t0
        print(f"    Causal discovery: {cd_time:.1f}s")

        # 2. For each target, for each method, for each top_m, for each predictor
        for target in valid_targets:
            target_idx = col_names.index(target)
            target_en = ZH_TO_EN.get(target, target)

            for top_m in cfg.top_ms:
                # --- Feature selection by each method ---
                feature_sets = {}

                # All features
                feature_sets["AllFeatures"] = col_names

                # Pearson
                feats_p = run_pearson_selection(X_train, target_idx, top_m,
                                               col_names)
                feature_sets["Pearson"] = feats_p

                # Causal methods
                for method_name, W in graphs.items():
                    if W is not None:
                        feats = select_top_parents(W, target_idx, top_m,
                                                   col_names)
                        feature_sets[method_name] = feats

                # --- Forecast with each predictor architecture ---
                for pred_name in cfg.predictors:
                    for fs_name, fs_cols in feature_sets.items():
                        try:
                            rmse, mae = run_forecast(
                                df_diff, split, fs_cols, target,
                                col_names=col_names,
                                seq_len=cfg.seq_len,
                                max_epochs=cfg.max_epochs,
                                lr=cfg.lr, hidden=cfg.hidden,
                                seed=seed, patience=cfg.patience,
                                predictor=pred_name,
                            )
                            all_rows.append(dict(
                                bench="Electricity",
                                seed=seed,
                                target=target_en,
                                top_m=top_m,
                                predictor=pred_name,
                                feature_method=fs_name,
                                n_features=len(fs_cols),
                                rmse=rmse,
                                mae=mae,
                            ))
                        except Exception as e:
                            warnings.warn(
                                f"Forecast failed: {fs_name}/{pred_name}"
                                f"/{target_en} seed={seed}: {e}")

            print(f"    target={target_en} done")

    return pd.DataFrame(all_rows)


def run_lorenz96_experiment(cfg: Cfg) -> pd.DataFrame:
    """
    Run the downstream prediction experiment on Lorenz-96.
    Here we have ground truth, so we can also report whether causal
    parents actually improve prediction over non-causal features.
    """
    print("\n" + "=" * 60)
    print(">>> Lorenz-96: Downstream Prediction via Causal Features")
    print("=" * 60)

    d = cfg.lorenz_d
    col_names = [f"V{i}" for i in range(d)]

    # Select target variables
    n_targets = min(cfg.lorenz_n_targets, d)
    target_indices = list(range(n_targets))

    print(f"  d={d}, T={cfg.lorenz_T}, F={cfg.lorenz_F}")
    print(f"  targets: V0..V{n_targets-1}")
    print(f"  prior_acc={cfg.lorenz_prior_acc}")

    all_rows = []

    for seed in cfg.seeds:
        print(f"\n  --- Seed {seed} ---")

        # Generate data
        X, B_true = generate_lorenz96(d=d, T=cfg.lorenz_T, F=cfg.lorenz_F,
                                       seed=seed)
        if not np.all(np.isfinite(X)):
            warnings.warn(f"Lorenz-96 seed={seed}: non-finite, skipping")
            continue

        split = int(len(X) * 0.8)
        X_train = X[:split]

        # Generate prior from ground truth
        P_prior = gen_prior_from_truth(B_true, cfg.lorenz_prior_acc,
                                       seed=seed + 999)

        # Discover causal graphs
        t0 = time.time()
        graphs = discover_causal_graphs(X_train, P_prior, d, cfg.K, cfg, seed)
        cd_time = time.time() - t0
        print(f"    Causal discovery: {cd_time:.1f}s")

        # Also add "Oracle" — use true causal parents
        graphs["Oracle(true)"] = B_true.astype(float)

        for target_idx in target_indices:
            target_name = col_names[target_idx]

            for top_m in cfg.top_ms:
                feature_sets = {}

                # All features
                feature_sets["AllFeatures"] = col_names

                # Pearson
                feats_p = run_pearson_selection(X_train, target_idx, top_m,
                                               col_names)
                feature_sets["Pearson"] = feats_p

                # Causal methods (including Oracle)
                for method_name, W in graphs.items():
                    if W is not None:
                        feats = select_top_parents(W, target_idx, top_m,
                                                   col_names)
                        feature_sets[method_name] = feats

                for pred_name in cfg.predictors:
                    for fs_name, fs_cols in feature_sets.items():
                        try:
                            # Convert col names to indices for ndarray input
                            feat_idx = [col_names.index(c) for c in fs_cols]
                            rmse, mae = run_forecast(
                                X, split, fs_cols, target_name,
                                col_names=col_names,
                                seq_len=cfg.seq_len,
                                max_epochs=cfg.max_epochs,
                                lr=cfg.lr, hidden=cfg.hidden,
                                seed=seed, patience=cfg.patience,
                                predictor=pred_name,
                            )
                            all_rows.append(dict(
                                bench="Lorenz96",
                                seed=seed,
                                target=target_name,
                                top_m=top_m,
                                predictor=pred_name,
                                feature_method=fs_name,
                                n_features=len(fs_cols),
                                rmse=rmse,
                                mae=mae,
                            ))
                        except Exception as e:
                            warnings.warn(
                                f"Forecast failed: {fs_name}/{pred_name}"
                                f"/{target_name} seed={seed}: {e}")

            print(f"    target={target_name} done")

    return pd.DataFrame(all_rows)


# ====================================================================
# 7. Summary Tables
# ====================================================================
def generate_summaries(df: pd.DataFrame, out: str):
    if df.empty:
        return

    # --- Table A: Feature method comparison (aggregated across targets) ---
    g = df.groupby(["bench", "feature_method", "predictor"])
    agg = g.agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        n=("seed", "count"),
    ).reset_index()
    p = os.path.join(out, "summary_method_predictor.csv")
    agg.to_csv(p, index=False)
    print(f">>> {p}")

    # --- Table B: Per-target results ---
    g2 = df.groupby(["bench", "target", "feature_method", "predictor"])
    agg2 = g2.agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()
    p2 = os.path.join(out, "summary_per_target.csv")
    agg2.to_csv(p2, index=False)
    print(f">>> {p2}")

    # --- Table C: Aggregated across predictors (highlight feature quality) ---
    g3 = df.groupby(["bench", "feature_method"])
    agg3 = g3.agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()
    p3 = os.path.join(out, "summary_feature_method_overall.csv")
    agg3.to_csv(p3, index=False)
    print(f">>> {p3}")

    # --- Table D: Top-M sensitivity ---
    if df["top_m"].nunique() > 1:
        g4 = df.groupby(["bench", "feature_method", "top_m"])
        agg4 = g4.agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        ).reset_index()
        p4 = os.path.join(out, "summary_topm_sensitivity.csv")
        agg4.to_csv(p4, index=False)
        print(f">>> {p4}")

    # --- Console summary ---
    print("\n" + "=" * 72)
    print("Overall RMSE by feature selection method (mean across all)")
    print("=" * 72)
    for bench in df["bench"].unique():
        sub = df[df["bench"] == bench]
        print(f"\n--- {bench} ---")
        overall = sub.groupby("feature_method")["rmse"].agg(["mean", "std"])
        overall = overall.sort_values("mean")
        print(overall.round(4).to_string())

    print("\n" + "=" * 72)
    print("RMSE by predictor architecture x feature method")
    print("=" * 72)
    for bench in df["bench"].unique():
        sub = df[df["bench"] == bench]
        print(f"\n--- {bench} ---")
        piv = sub.groupby(["predictor", "feature_method"])["rmse"].mean()
        piv = piv.unstack("feature_method")
        print(piv.round(4).to_string())


# ====================================================================
# 8. Figures
# ====================================================================
COLORS = {
    "AllFeatures":       "#95A5A6",
    "Pearson":           "#F39C12",
    "PRCD-MAP(prior)":   "#E74C3C",
    "PRCD-MAP(uniform)": "#9B59B6",
    "DYNOTEARS":         "#2C3E50",
    "PCMCI+":            "#27AE60",
    "VARLiNGAM":         "#3498DB",
    "Oracle(true)":      "#1ABC9C",
}


def _save(prefix):
    plt.savefig(prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(prefix + ".pdf", bbox_inches="tight")
    plt.close()
    print(f">>> {prefix}.png / .pdf")


def generate_figures(df: pd.DataFrame, out: str):
    if df.empty:
        return
    _fig_main_comparison(df, out)
    _fig_per_predictor(df, out)
    _fig_per_target(df, out)
    if df["top_m"].nunique() > 1:
        _fig_topm_sensitivity(df, out)
    if df["bench"].nunique() > 1:
        _fig_cross_benchmark(df, out)


def _fig_main_comparison(df, out):
    """
    Figure 1: Main bar chart — RMSE by feature selection method.
    Aggregated across all targets and predictors, one panel per benchmark.
    """
    benchmarks = sorted(df["bench"].unique())
    n_b = len(benchmarks)

    fig, axes = plt.subplots(1, n_b, figsize=(7 * n_b, 6), squeeze=False)

    for idx, bench in enumerate(benchmarks):
        ax = axes[0][idx]
        sub = df[df["bench"] == bench]
        methods = sorted(sub["feature_method"].unique(),
                         key=lambda m: sub[sub["feature_method"] == m]["rmse"].mean())
        n_m = len(methods)

        means = [sub[sub["feature_method"] == m]["rmse"].mean() for m in methods]
        stds = [sub[sub["feature_method"] == m]["rmse"].std() for m in methods]
        colors = [COLORS.get(m, "#BDC3C7") for m in methods]

        bars = ax.bar(range(n_m), means, yerr=stds,
                      color=colors, alpha=0.85, capsize=4, edgecolor="white",
                      linewidth=0.8)

        ax.set_xticks(range(n_m))
        ax.set_xticklabels([m.replace("PRCD-MAP", "PRCD") for m in methods],
                           rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("RMSE (mean +/- std)", fontsize=11)
        ax.set_title(f"{bench}", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", ls=":", alpha=0.4)

        # Add value labels on bars
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Downstream Prediction: RMSE by Feature Selection Method",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(os.path.join(out, "fig1_main_comparison"))


def _fig_per_predictor(df, out):
    """
    Figure 2: Grouped bars — RMSE for each predictor architecture.
    Shows that causal features help across all architectures.
    """
    predictors = sorted(df["predictor"].unique())
    benchmarks = sorted(df["bench"].unique())

    for bench in benchmarks:
        sub_b = df[df["bench"] == bench]
        methods = sorted(sub_b["feature_method"].unique())
        n_p = len(predictors)
        n_m = len(methods)

        fig, ax = plt.subplots(figsize=(max(10, n_m * 1.5), 6))
        x = np.arange(n_m)
        w = 0.8 / max(n_p, 1)

        for i, pred in enumerate(predictors):
            vals, errs = [], []
            for m in methods:
                s = sub_b[(sub_b["feature_method"] == m)
                          & (sub_b["predictor"] == pred)]
                vals.append(s["rmse"].mean() if len(s) else 0)
                errs.append(s["rmse"].std() if len(s) else 0)
            ax.bar(x + i * w, vals, w, yerr=errs,
                   label=pred.upper(), alpha=0.85, capsize=2)

        ax.set_xticks(x + w * (n_p - 1) / 2)
        ax.set_xticklabels([m.replace("PRCD-MAP", "PRCD") for m in methods],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("RMSE")
        ax.set_title(f"{bench}: RMSE by Predictor Architecture",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        plt.tight_layout()
        _save(os.path.join(out, f"fig2_per_predictor_{bench.lower()}"))


def _fig_per_target(df, out):
    """
    Figure 3: Per-target RMSE comparison (heatmap style).
    """
    for bench in sorted(df["bench"].unique()):
        sub_b = df[df["bench"] == bench]
        targets = sorted(sub_b["target"].unique())
        methods = sorted(sub_b["feature_method"].unique())

        if len(targets) < 2 or len(methods) < 2:
            continue

        # Build pivot table
        piv = sub_b.groupby(["target", "feature_method"])["rmse"].mean()
        piv = piv.unstack("feature_method").fillna(0)

        fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2),
                                        max(4, len(targets) * 0.8)))
        im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([c.replace("PRCD-MAP", "PRCD") for c in piv.columns],
                           rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=10)

        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.values[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if val > piv.values.mean() else "black")

        ax.set_title(f"{bench}: RMSE by Target x Feature Method",
                     fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="RMSE", shrink=0.8)
        plt.tight_layout()
        _save(os.path.join(out, f"fig3_per_target_{bench.lower()}"))


def _fig_topm_sensitivity(df, out):
    """
    Figure 4: RMSE vs top-M (feature set size).
    """
    for bench in sorted(df["bench"].unique()):
        sub_b = df[df["bench"] == bench]
        top_ms = sorted(sub_b["top_m"].unique())
        methods = sorted(sub_b["feature_method"].unique())

        # Skip AllFeatures (doesn't depend on top_m)
        methods = [m for m in methods if m != "AllFeatures"]
        if len(top_ms) < 2 or len(methods) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for m in methods:
            sub = sub_b[sub_b["feature_method"] == m]
            agg = sub.groupby("top_m").agg(
                y=("rmse", "mean"), e=("rmse", "std")).reset_index()
            ax.errorbar(agg["top_m"], agg["y"], yerr=agg["e"],
                        label=m.replace("PRCD-MAP", "PRCD"),
                        color=COLORS.get(m, "grey"),
                        marker="o", lw=2, capsize=3, markersize=7)

        ax.set_xlabel("Top-M (number of selected features)", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(f"{bench}: Sensitivity to Feature Set Size",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, ls=":", alpha=0.5)
        plt.tight_layout()
        _save(os.path.join(out, f"fig4_topm_{bench.lower()}"))


def _fig_cross_benchmark(df, out):
    """
    Figure 5: Cross-benchmark comparison — normalized RMSE improvement
    over AllFeatures baseline.
    """
    methods = sorted(df["feature_method"].unique())
    methods = [m for m in methods if m != "AllFeatures"]
    benchmarks = sorted(df["bench"].unique())

    if len(benchmarks) < 2 or len(methods) < 2:
        return

    # Compute relative improvement: (RMSE_all - RMSE_method) / RMSE_all * 100
    improvements = {}
    for bench in benchmarks:
        sub_b = df[df["bench"] == bench]
        rmse_all = sub_b[sub_b["feature_method"] == "AllFeatures"]["rmse"].mean()
        if rmse_all == 0 or np.isnan(rmse_all):
            continue
        for m in methods:
            sub_m = sub_b[sub_b["feature_method"] == m]
            rmse_m = sub_m["rmse"].mean()
            imp = (rmse_all - rmse_m) / rmse_all * 100
            if m not in improvements:
                improvements[m] = {}
            improvements[m][bench] = imp

    if not improvements:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(benchmarks))
    w = 0.8 / len(methods)

    for i, m in enumerate(methods):
        vals = [improvements.get(m, {}).get(b, 0) for b in benchmarks]
        ax.bar(x + i * w, vals, w,
               label=m.replace("PRCD-MAP", "PRCD"),
               color=COLORS.get(m, "grey"), alpha=0.85)

    ax.axhline(y=0, color="black", lw=0.8, ls="-")
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylabel("RMSE Improvement over AllFeatures (%)", fontsize=11)
    ax.set_title("Relative RMSE Improvement by Feature Selection Method",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    _save(os.path.join(out, "fig5_cross_benchmark_improvement"))


# ====================================================================
# 9. Representative Forecast Curves
# ====================================================================
def plot_representative_curves(df_diff, split, col_names, P_prior,
                               cfg: Cfg, out: str):
    """
    Plot forecast curves for one representative seed/target/predictor,
    comparing all feature selection methods.
    """
    seed = cfg.seeds[0]
    target = cfg.elec_targets[0] if cfg.elec_targets else None
    if target is None or target not in col_names:
        return

    target_en = ZH_TO_EN.get(target, target)
    target_idx = col_names.index(target)
    d = len(col_names)
    top_m = cfg.top_ms[0]
    X_train = df_diff.values[:split]

    # Discover graphs
    graphs = discover_causal_graphs(X_train, P_prior, d, cfg.K, cfg, seed)

    # Feature sets
    feature_sets = {"AllFeatures": col_names}
    feature_sets["Pearson"] = run_pearson_selection(
        X_train, target_idx, top_m, col_names)
    for mname, W in graphs.items():
        if W is not None:
            feature_sets[mname] = select_top_parents(
                W, target_idx, top_m, col_names)

    # Forecast with LSTM and collect predictions
    predictions = {}
    y_true = None

    for fs_name, fs_cols in feature_sets.items():
        set_seed(seed)
        X_raw = df_diff[fs_cols].values
        y_raw = df_diff[[target]].values

        X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
        y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

        val_split = int(len(X_train_raw) * 0.8)
        X_tr_raw = X_train_raw[:val_split]
        y_tr_raw = y_train_raw[:val_split]

        sx = StandardScaler()
        sy = StandardScaler()
        X_tr = sx.fit_transform(X_tr_raw)
        X_test = sx.transform(X_test_raw)
        y_tr = sy.fit_transform(y_tr_raw)
        y_test = sy.transform(y_test_raw)

        seq_len = cfg.seq_len
        Xte, yte = make_sequences(X_test, y_test, seq_len)
        if len(Xte) < 5:
            continue

        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        yte_t = torch.tensor(yte, dtype=torch.float32)

        # Quick LSTM training
        X_val_raw = X_train_raw[val_split:]
        y_val_raw = y_train_raw[val_split:]
        X_val = sx.transform(X_val_raw)
        y_val = sy.transform(y_val_raw)

        Xtr_seq, ytr_seq = make_sequences(X_tr, y_tr, seq_len)
        Xva_seq, yva_seq = make_sequences(X_val, y_val, seq_len)

        if len(Xtr_seq) < 10:
            continue

        Xtr_t = torch.tensor(Xtr_seq, dtype=torch.float32)
        ytr_t = torch.tensor(ytr_seq, dtype=torch.float32)
        Xva_t = torch.tensor(Xva_seq, dtype=torch.float32)
        yva_t = torch.tensor(yva_seq, dtype=torch.float32)

        model = LSTMForecaster(input_dim=Xtr_t.shape[2], hidden_dim=cfg.hidden)
        opt_ = optim.Adam(model.parameters(), lr=cfg.lr)
        crit = nn.MSELoss()
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64,
                            shuffle=True)

        best_val = float("inf")
        best_state = None
        bad = 0
        for ep in range(cfg.max_epochs):
            model.train()
            for bx, by in loader:
                opt_.zero_grad()
                model(bx)
                loss = crit(model(bx), by)
                loss.backward()
                opt_.step()
            model.eval()
            with torch.no_grad():
                vl = float(crit(model(Xva_t), yva_t).item())
            if vl + 1e-9 < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= cfg.patience:
                break
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            pred_s = model(Xte_t).numpy()
        pred = sy.inverse_transform(pred_s).flatten()
        ytrue = sy.inverse_transform(yte_t.numpy()).flatten()

        if y_true is None:
            y_true = ytrue
        predictions[fs_name] = pred

    if y_true is None or not predictions:
        return

    # Plot
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.plot(y_true, color="black", lw=1.8, alpha=0.7, label="Ground Truth")

    for fs_name, pred in predictions.items():
        rmse = float(np.sqrt(mean_squared_error(y_true[:len(pred)], pred)))
        color = COLORS.get(fs_name, "grey")
        ls = "--" if fs_name == "AllFeatures" else ("-." if fs_name == "Pearson" else "-")
        lw = 2.5 if "PRCD" in fs_name else 1.5
        label = f"{fs_name.replace('PRCD-MAP', 'PRCD')} (RMSE={rmse:.2f})"
        ax.plot(pred, color=color, ls=ls, lw=lw, alpha=0.85, label=label)

    ax.set_xlabel("Test time step", fontsize=11)
    ax.set_ylabel("Differenced value", fontsize=11)
    ax.set_title(f"Forecast Comparison: {target_en} "
                 f"(LSTM, top-{top_m}, seed={seed})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    _save(os.path.join(out, "fig6_forecast_curves"))


# ====================================================================
# 10. Entry Point
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Downstream Prediction via Causal Feature Selection")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run")
    parser.add_argument("--full", action="store_true",
                        help="Full NeurIPS sweep")
    parser.add_argument("--bench", nargs="+", type=str, default=None,
                        choices=["electricity", "lorenz96"],
                        help="Specific benchmark(s)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--top-m", nargs="+", type=int, default=None)
    parser.add_argument("--predictors", nargs="+", type=str, default=None,
                        choices=["lstm", "var", "transformer"])
    parser.add_argument("--no-pcmci", action="store_true")
    parser.add_argument("--no-varlingam", action="store_true")
    parser.add_argument("--no-dynotears", action="store_true")
    parser.add_argument("--elec-targets", nargs="+", type=str, default=None,
                        help="Electricity target column names (Chinese)")
    args = parser.parse_args()

    # Select config
    if args.quick:
        cfg = cfg_quick()
    elif args.full:
        cfg = cfg_full()
    else:
        cfg = Cfg()

    # CLI overrides
    if args.output:       cfg.output_dir   = args.output
    if args.seeds:        cfg.seeds        = args.seeds
    if args.top_m:        cfg.top_ms       = args.top_m
    if args.predictors:   cfg.predictors   = args.predictors
    if args.bench:        cfg.benchmarks   = args.bench
    if args.no_pcmci:     cfg.do_pcmci     = False
    if args.no_varlingam: cfg.do_varlingam = False
    if args.no_dynotears: cfg.do_dynotears = False
    if args.elec_targets: cfg.elec_targets = args.elec_targets

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 68)
    print(" Experiment 3: Downstream Prediction via Causal Feature Selection")
    print("=" * 68)
    print(f"  benchmarks  = {cfg.benchmarks}")
    print(f"  top_ms      = {cfg.top_ms}")
    print(f"  predictors  = {cfg.predictors}")
    print(f"  seeds       = {cfg.seeds}")
    print(f"  baselines   = DYNOTEARS={cfg.do_dynotears}, "
          f"PCMCI+={cfg.do_pcmci and HAS_TIGRAMITE}, "
          f"VARLiNGAM={cfg.do_varlingam and HAS_LINGAM}")
    print(f"  output      = {cfg.output_dir}")
    print("=" * 68)

    t_global = time.time()
    all_dfs = []

    # ============================================
    # Electricity Dataset
    # ============================================
    if "electricity" in cfg.benchmarks:
        df_elec = run_electricity_experiment(cfg)
        all_dfs.append(df_elec)

    # ============================================
    # Lorenz-96 Dataset
    # ============================================
    if "lorenz96" in cfg.benchmarks:
        df_lorenz = run_lorenz96_experiment(cfg)
        all_dfs.append(df_lorenz)

    # ============================================
    # Aggregate Results
    # ============================================
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        csv_path = os.path.join(cfg.output_dir, "exp3_full_results.csv")
        df_all.to_csv(csv_path, index=False)
        print(f"\n>>> Saved {len(df_all)} rows -> {csv_path}")

        generate_summaries(df_all, cfg.output_dir)
        generate_figures(df_all, cfg.output_dir)

        # Plot representative forecast curves (electricity only)
        if "electricity" in cfg.benchmarks:
            try:
                df_diff, P_prior, split, col_names = load_electricity(
                    cfg.electricity_xlsx, cfg.electricity_prior)
                plot_representative_curves(df_diff, split, col_names,
                                           P_prior, cfg, cfg.output_dir)
            except Exception as e:
                warnings.warn(f"Failed to plot forecast curves: {e}")
    else:
        print("\n>>> No results to aggregate.")

    elapsed = time.time() - t_global
    print(f"\n>>> Experiment 3 complete in {elapsed:.1f}s")
    print(f">>> Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
