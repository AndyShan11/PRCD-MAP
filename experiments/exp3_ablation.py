"""
=============================================================================
Experiment 4 — Extended Ablation Study for PRCD-MAP
=============================================================================
Isolates each component's contribution via six ablation variants:

  (A) No prior            — lambda2=0, uniform prior, learn_tau=False
  (B) Fixed tau=1         — prior with learn_tau=False, tau0=1.0
  (C) Full model          — prior with learn_tau=True  (full PRCD-MAP)
  (D) No L1               — lambda1=0, prior regularisation only
  (E) Prior on lags only  — apply_prior_to_w0=False
  (F) Hard mask           — binarise P_prior -> 0/1 mask, force zeros

Evaluation domains:
  1. Synthetic SVAR       -> graph quality metrics (AUROC, AUPRC, F1, SHD)
  2. Lorenz-96            -> graph quality metrics
  3. Electricity (real)   -> downstream LSTM forecast RMSE

Key addition: "Prior Misspecification Robustness" figure —
  F1 vs prior_accuracy for all 6 variants (the core selling point).

Usage:
  python exp4_ablation.py                        # default
  python exp4_ablation.py --quick                # tiny test run
  python exp4_ablation.py --sub synthetic        # synthetic only
  python exp4_ablation.py --sub lorenz           # Lorenz-96 only
  python exp4_ablation.py --sub real             # real data only
  python exp4_ablation.py --sub hard_mask        # soft vs hard focus
  python exp4_ablation.py --seeds 0 1 2 3 4
=============================================================================
"""

import os, sys, time, warnings, argparse
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- shared utilities from exp_utils ----
from utils import (                           # noqa: F401
    set_seed, make_lag_tensors, standardize, ensure_dir,
    make_er_dag, make_lag_matrices, simulate_svar_linear,
    gen_prior, compute_all_metrics, combine_W0_Wk,
    run_dynotears, run_pcmci_plus, run_varlingam, run_prcd_map,
    generate_lorenz96, lorenz96_ground_truth, gen_prior_from_truth,
    binarize_prior_to_mask,
    PRCD_MAP_HardMask, train_hard_mask_alm,
    COLORS, MARKERS, save_fig, print_rhino_table, ZH_TO_EN,
    load_electricity,
    PRCD_MAP_Model, train_prcd_alm,
)

# ====================================================================
# Variant metadata
# ====================================================================
VARIANT_NAMES = {
    "A": "(A) No prior (lam2=0)",
    "B": "(B) Fixed tau=1",
    "C": "(C) Full model (learn tau)",
    "D": "(D) No L1 (lam1=0)",
    "E": "(E) Prior on lags only",
    "F": "(F) Hard mask",
    "G": "(G) NAM nonlinear",
    "H": "(H) No warm-start",
    "J": "(J) No lambda scheduling",
}
VARIANT_SHORT = {
    "A": "NoPrior", "B": "FixedTau", "C": "FullModel",
    "D": "NoL1",    "E": "LagsOnly", "F": "HardMask",
    "G": "NAM",
    "H": "NoWarmStart", "J": "NoLamSched",
}
VARIANT_ORDER = ["NoPrior", "FixedTau", "FullModel",
                 "NoL1", "LagsOnly", "HardMask", "NAM",
                 "NoWarmStart", "NoLamSched"]
VARIANT_COLORS = {
    "NoPrior":  "#95A5A6", "FixedTau":  "#E67E22",
    "FullModel": "#E74C3C", "NoL1":     "#9B59B6",
    "LagsOnly": "#3498DB",  "HardMask": "#2C3E50",
    "NAM":      "#C0392B",
    "NoWarmStart": "#1ABC9C", "NoLamSched": "#8E44AD",
    "AllFeatures": "#BDC3C7",
}
VARIANT_MARKERS = {
    "NoPrior": "^", "FixedTau": "s", "FullModel": "o",
    "NoL1": "D",    "LagsOnly": "v", "HardMask": "P",
    "NAM": "p",
    "NoWarmStart": "H", "NoLamSched": "X",
    "AllFeatures": "X",
}


# ====================================================================
# 1. Variant runner functions (A-F)
# ====================================================================
def _make_model_and_train(X, P_prior, d, K, seed, max_iter, inner_iter,
                          lr, lambda1, lambda2, learn_tau, tau0,
                          apply_prior_to_w0=True,
                          warm_start=True, lambda_schedule=True,
                          score_calibration=False):
    """Shared helper: build PRCD_MAP_Model, train, return (W0, Wk, tau)."""
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]
    model = PRCD_MAP_Model(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau, tau0=tau0,
        tau_min=0.05, tau_max=3.0,
        apply_prior_to_w0=apply_prior_to_w0,
    ).to(dev)
    W0, Wk, tau = train_prcd_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=False, postprocess=False,
        tau_warmup=0,
        warm_start=warm_start,
        lambda_schedule=lambda_schedule,
    )
    # Score calibration (row-wise normalization for AUROC)
    if score_calibration:
        from utils import _calibrate_scores
        W0 = _calibrate_scores(W0)
        Wk = [_calibrate_scores(wk) for wk in Wk]
    return W0, Wk, float(tau)


def run_variant_A(X, d, K, seed, max_iter, inner_iter, lr, lambda1):
    """(A) No prior: lambda2=0, uniform prior, learn_tau=False."""
    P_unif = np.full((d, d), 0.5)
    return _make_model_and_train(
        X, P_unif, d, K, seed, max_iter, inner_iter, lr,
        lambda1=lambda1, lambda2=0.0, learn_tau=False, tau0=1.0)


def run_variant_B(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """(B) Fixed tau=1: prior with learn_tau=False."""
    return _make_model_and_train(
        X, P_prior, d, K, seed, max_iter, inner_iter, lr,
        lambda1=lambda1, lambda2=lambda2, learn_tau=False, tau0=1.0)


def run_variant_C(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """(C) Full model: prior with learn_tau=True."""
    return _make_model_and_train(
        X, P_prior, d, K, seed, max_iter, inner_iter, lr,
        lambda1=lambda1, lambda2=lambda2, learn_tau=True, tau0=1.0)


def run_variant_D(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda2):
    """(D) No L1: lambda1=0, prior-only regularisation."""
    return _make_model_and_train(
        X, P_prior, d, K, seed, max_iter, inner_iter, lr,
        lambda1=0.0, lambda2=lambda2, learn_tau=True, tau0=1.0)


def run_variant_E(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """(E) Prior on lags only: apply_prior_to_w0=False."""
    return _make_model_and_train(
        X, P_prior, d, K, seed, max_iter, inner_iter, lr,
        lambda1=lambda1, lambda2=lambda2, learn_tau=True, tau0=1.0,
        apply_prior_to_w0=False)


def run_variant_F(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, mask_threshold=0.5):
    """(F) Hard mask: binarise prior, force masked edges to zero."""
    set_seed(seed)
    hard_mask = binarize_prior_to_mask(P_prior, threshold=mask_threshold)
    X_t, X_lags = make_lag_tensors(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]
    model = PRCD_MAP_HardMask(
        num_vars=d, lag_k=K, hard_mask=hard_mask, lambda1=lambda1,
    ).to(dev)
    W0, Wk = train_hard_mask_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=1e-6,
    )
    return W0, Wk, float("nan")


def run_variant_G(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """(G) NAM nonlinear: PRCD-MAP with Neural Additive Model parameterization."""
    from model_prcd_map_nam import PRCD_MAP_NAM, train_prcd_nam_alm
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]
    model = PRCD_MAP_NAM(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=True, tau_min=0.05, tau_max=3.0,
        edge_hidden=16, edge_layers=2,
    ).to(dev)
    W0, Wk, tau = train_prcd_nam_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=5e-4,
        verbose=False,
    )
    return W0, Wk, float(tau)


def run_variant_H(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """(H) No warm-start: disable Ridge OLS initialization."""
    return _make_model_and_train(
        X, P_prior, d, K, seed, max_iter, inner_iter, lr,
        lambda1=lambda1, lambda2=lambda2, learn_tau=True, tau0=1.0,
        warm_start=False)


def run_variant_I(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """(I) No score calibration: skip row-wise normalization."""
    return _make_model_and_train(
        X, P_prior, d, K, seed, max_iter, inner_iter, lr,
        lambda1=lambda1, lambda2=lambda2, learn_tau=True, tau0=1.0,
        score_calibration=False)


def run_variant_J(X, P_prior, d, K, seed, max_iter, inner_iter, lr,
                  lambda1, lambda2):
    """(J) No lambda scheduling: constant lambda throughout training."""
    return _make_model_and_train(
        X, P_prior, d, K, seed, max_iter, inner_iter, lr,
        lambda1=lambda1, lambda2=lambda2, learn_tau=True, tau0=1.0,
        lambda_schedule=False)


def _dispatch_variant(var, X_std, P_prior, d, K, seed, cfg):
    """Run one variant and return (W0, Wk, tau)."""
    mi, ii, lr = cfg.max_iter, cfg.inner_iter, cfg.lr
    l1, l2 = cfg.lambda1, cfg.lambda2
    if var == "A":
        return run_variant_A(X_std, d, K, seed, mi, ii, lr, l1)
    elif var == "B":
        return run_variant_B(X_std, P_prior, d, K, seed, mi, ii, lr, l1, l2)
    elif var == "C":
        return run_variant_C(X_std, P_prior, d, K, seed, mi, ii, lr, l1, l2)
    elif var == "D":
        return run_variant_D(X_std, P_prior, d, K, seed, mi, ii, lr, l2)
    elif var == "E":
        return run_variant_E(X_std, P_prior, d, K, seed, mi, ii, lr, l1, l2)
    elif var == "F":
        return run_variant_F(X_std, P_prior, d, K, seed, mi, ii, lr, l1,
                             cfg.mask_threshold)
    elif var == "G":
        if d > 30:
            warnings.warn(f"NAM variant G not suitable for d={d}>30, skipping")
            return None, None, float("nan")
        return run_variant_G(X_std, P_prior, d, K, seed, mi, ii, lr, l1, l2)
    elif var == "H":
        return run_variant_H(X_std, P_prior, d, K, seed, mi, ii, lr, l1, l2)
    elif var == "I":
        return run_variant_I(X_std, P_prior, d, K, seed, mi, ii, lr, l1, l2)
    elif var == "J":
        return run_variant_J(X_std, P_prior, d, K, seed, mi, ii, lr, l1, l2)
    raise ValueError(f"Unknown variant: {var}")


# ====================================================================
# 2. LSTMForecaster + helpers (local, for real-data ablation)
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


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def _select_top_parents(W_combined, target_idx, top_m, col_names):
    d = W_combined.shape[0]
    w_in = np.abs(W_combined[:, target_idx]).copy()
    w_in[target_idx] = 0.0
    n_parents = min(max(0, top_m - 1), d - 1)
    parent_idx = np.argsort(-w_in)[:n_parents] if n_parents > 0 else []
    feats = [col_names[i] for i in parent_idx] + [col_names[target_idx]]
    seen, unique = set(), []
    for f in feats:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def run_forecast(df_diff, split, feature_cols, target_col,
                 seq_len=7, max_epochs=50, lr=3e-3, hidden=32,
                 seed=0, patience=8, batch_size=64):
    """Train LSTM forecaster, return (rmse, mae)."""
    set_seed(seed)
    X_raw = df_diff[feature_cols].values
    y_raw = df_diff[[target_col]].values
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
    val_split = int(len(X_train_raw) * 0.8)
    X_tr_raw, X_val_raw = X_train_raw[:val_split], X_train_raw[val_split:]
    y_tr_raw, y_val_raw = y_train_raw[:val_split], y_train_raw[val_split:]
    sx, sy = StandardScaler(), StandardScaler()
    X_tr = sx.fit_transform(X_tr_raw)
    X_val = sx.transform(X_val_raw)
    X_test = sx.transform(X_test_raw)
    y_tr = sy.fit_transform(y_tr_raw)
    y_val = sy.transform(y_val_raw)
    y_test = sy.transform(y_test_raw)
    Xtr, ytr = _make_sequences(X_tr, y_tr, seq_len)
    Xva, yva = _make_sequences(X_val, y_val, seq_len)
    Xte, yte = _make_sequences(X_test, y_test, seq_len)
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
    best_val, best_state, bad = float("inf"), None, 0
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
# 3. Configuration
# ====================================================================
@dataclass
class Cfg:
    synth_dims:       List[int]   = field(default_factory=lambda: [10, 20])
    synth_Ts:         List[int]   = field(default_factory=lambda: [500])
    synth_noises:     List[str]   = field(default_factory=lambda: ["gaussian"])
    synth_K:          int         = 1
    synth_edge_prob:  float       = 0.15
    synth_prior_accs: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])

    lorenz_d:  int   = 10
    lorenz_T:  int   = 2000
    lorenz_F:  float = 10.0

    electricity_xlsx:  str = "./data/electricity.xlsx"
    electricity_prior: str = "./data/electricity_prior.csv"
    elec_targets: List[str] = field(default_factory=lambda: [
        "\u5927\u5de5\u4e1a\u7535\u91cf", "\u5c45\u6c11\u751f\u6d3b",
        "\u5546\u4e1a\u7528\u7535"])
    elec_top_m: int = 6

    seeds:    List[int] = field(default_factory=lambda: list(range(3)))
    variants: List[str] = field(default_factory=lambda: [
        "A", "B", "C", "D", "E", "F", "H", "J"])

    lambda1:    float = 0.001
    lambda2:    float = 0.01
    max_iter:   int   = 35
    inner_iter: int   = 400
    lr:         float = 1e-2

    seq_len:    int   = 7
    max_epochs: int   = 50
    fc_lr:      float = 3e-3
    hidden:     int   = 32
    patience:   int   = 8

    mask_threshold: float = 0.5

    do_synthetic: bool = True
    do_lorenz:    bool = True
    do_real:      bool = True

    output_dir: str = "exp4_results"


def cfg_quick():
    return Cfg(
        synth_dims=[10], synth_Ts=[500],
        synth_prior_accs=[0.3, 0.7],
        lorenz_T=500,
        seeds=list(range(2)),
        max_iter=10, inner_iter=100,
        max_epochs=20,
        elec_targets=["\u5927\u5de5\u4e1a\u7535\u91cf"],
        do_lorenz=False,
    )


def cfg_full():
    return Cfg(
        synth_dims=[10, 20],
        synth_Ts=[500],
        synth_noises=["gaussian", "laplace"],
        synth_prior_accs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        lorenz_T=2000,
        seeds=list(range(5)),
        max_epochs=60,
        elec_targets=["\u5927\u5de5\u4e1a\u7535\u91cf",
                      "\u5c45\u6c11\u751f\u6d3b",
                      "\u5546\u4e1a\u7528\u7535",
                      "\u975e\u666e\u5de5\u4e1a"],
    )


# ====================================================================
# 4. run_synthetic_ablation
# ====================================================================
def run_synthetic_ablation(cfg: Cfg) -> pd.DataFrame:
    """Graph quality metrics on synthetic SVAR data."""
    print("\n" + "=" * 60)
    print(">>> Synthetic Ablation: Graph Quality Metrics")
    print("=" * 60)

    all_rows = []
    settings = [
        dict(d=d, T=T, noise=nt, prior_acc=acc, seed=seed)
        for d in cfg.synth_dims
        for T in cfg.synth_Ts
        for nt in cfg.synth_noises
        for acc in cfg.synth_prior_accs
        for seed in cfg.seeds
    ]
    n_total = len(settings) * len(cfg.variants)
    print(f"  {len(settings)} settings x {len(cfg.variants)} variants "
          f"= {n_total} runs")
    t_global = time.time()

    for idx, st in enumerate(settings):
        d, T, nt = st["d"], st["T"], st["noise"]
        acc, seed = st["prior_acc"], st["seed"]
        K = cfg.synth_K

        W0_true = make_er_dag(d, cfg.synth_edge_prob, seed=seed)
        Wk_true = make_lag_matrices(d, K, seed=seed)
        X = simulate_svar_linear(T, W0_true, Wk_true, nt, seed=seed)
        if X is None or not np.all(np.isfinite(X)):
            continue
        X_std = standardize(X)
        P_prior = gen_prior(W0_true, Wk_true, acc, "random",
                            seed=seed + 999)

        B_comb_true = (np.abs(W0_true) > 1e-10).astype(int)
        for Wk_t in Wk_true:
            B_comb_true = np.maximum(
                B_comb_true, (np.abs(Wk_t) > 1e-10).astype(int))

        if (idx + 1) % max(1, len(settings) // 10) == 0 or idx == 0:
            elapsed = time.time() - t_global
            print(f"  [{idx+1}/{len(settings)}] d={d} T={T} {nt} "
                  f"acc={acc} s={seed}  ({elapsed:.0f}s)")

        for var in cfg.variants:
            try:
                t0 = time.time()
                W0_est, Wk_est, tau = _dispatch_variant(
                    var, X_std, P_prior, d, K, seed, cfg)
                elapsed_run = time.time() - t0

                met_w0 = compute_all_metrics(
                    (np.abs(W0_true) > 1e-10).astype(float), W0_est)
                W_comb_est = combine_W0_Wk(W0_est, Wk_est)
                met_comb = compute_all_metrics(
                    B_comb_true.astype(float), W_comb_est)

                all_rows.append(dict(
                    data="synthetic", d=d, T=T, noise=nt,
                    prior_acc=acc, seed=seed,
                    variant=var, variant_name=VARIANT_SHORT[var],
                    tau=tau, time=elapsed_run,
                    w0_auroc=met_w0["auroc"], w0_auprc=met_w0["auprc"],
                    w0_f1=met_w0["f1_opt"], w0_shd_norm=met_w0["shd_norm"],
                    comb_auroc=met_comb["auroc"], comb_auprc=met_comb["auprc"],
                    comb_f1=met_comb["f1_opt"],
                    comb_shd_norm=met_comb["shd_norm"],
                ))
            except Exception as e:
                warnings.warn(f"Variant {var} failed: d={d} T={T} "
                              f"acc={acc} seed={seed}: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        ensure_dir(cfg.output_dir)
        p = os.path.join(cfg.output_dir, "ablation_synthetic.csv")
        df.to_csv(p, index=False)
        print(f">>> {p} ({len(df)} rows)")
    return df


# ====================================================================
# 5. run_lorenz_ablation
# ====================================================================
def run_lorenz_ablation(cfg: Cfg) -> pd.DataFrame:
    """Graph quality metrics on Lorenz-96 nonlinear system."""
    print("\n" + "=" * 60)
    print(">>> Lorenz-96 Ablation")
    print("=" * 60)

    d, T, K = cfg.lorenz_d, cfg.lorenz_T, cfg.synth_K
    all_rows = []

    for seed in cfg.seeds:
        X, B_true = generate_lorenz96(d=d, T=T, F=cfg.lorenz_F, seed=seed)
        if not np.all(np.isfinite(X)):
            continue

        for acc in cfg.synth_prior_accs:
            P_prior = gen_prior_from_truth(B_true, acc, "random",
                                           seed=seed + 999)
            for var in cfg.variants:
                try:
                    t0 = time.time()
                    W0_est, Wk_est, tau = _dispatch_variant(
                        var, X, P_prior, d, K, seed, cfg)
                    W_comb = combine_W0_Wk(W0_est, Wk_est)
                    met = compute_all_metrics(B_true.astype(float), W_comb)
                    all_rows.append(dict(
                        data="lorenz96", d=d, T=T, noise="lorenz",
                        prior_acc=acc, seed=seed,
                        variant=var, variant_name=VARIANT_SHORT[var],
                        tau=tau, time=time.time() - t0,
                        auroc=met["auroc"], auprc=met["auprc"],
                        f1_opt=met["f1_opt"], shd_norm=met["shd_norm"],
                        f1_topk=met.get("f1_topk", 0.0),
                    ))
                except Exception as e:
                    warnings.warn(f"Lorenz {var} acc={acc} seed={seed}: {e}")
        print(f"  seed={seed} done")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        ensure_dir(cfg.output_dir)
        p = os.path.join(cfg.output_dir, "ablation_lorenz96.csv")
        df.to_csv(p, index=False)
        print(f">>> {p} ({len(df)} rows)")
    return df


# ====================================================================
# 6. run_real_ablation (downstream LSTM RMSE)
# ====================================================================
def run_real_ablation(cfg: Cfg) -> pd.DataFrame:
    """Downstream forecasting RMSE on electricity data."""
    print("\n" + "=" * 60)
    print(">>> Real Data Ablation: Downstream Forecasting RMSE")
    print("=" * 60)

    try:
        df_diff, P_prior, split, col_names = load_electricity(
            cfg.electricity_xlsx, cfg.electricity_prior)
    except Exception as e:
        warnings.warn(f"Failed to load electricity data: {e}")
        return pd.DataFrame()

    d = len(col_names)
    X_all = df_diff.values
    valid_targets = [t for t in cfg.elec_targets if t in col_names]
    if not valid_targets:
        warnings.warn("No valid electricity targets found!")
        return pd.DataFrame()

    print(f"  d={d}, T={len(df_diff)}, split={split}")
    print(f"  targets: {[ZH_TO_EN.get(t, t) for t in valid_targets]}")

    X_train = X_all[:split]
    mu, sd = X_train.mean(0), X_train.std(0)
    sd[sd == 0] = 1.0
    X_train_std = (X_train - mu) / sd
    K = cfg.synth_K
    all_rows = []

    for seed in cfg.seeds:
        print(f"\n  --- Seed {seed} ---")
        variant_graphs = {}
        for var in cfg.variants:
            try:
                t0 = time.time()
                W0, Wk, tau = _dispatch_variant(
                    var, X_train_std, P_prior, d, K, seed, cfg)
                W_comb = combine_W0_Wk(W0, Wk)
                variant_graphs[var] = (W_comb, tau)
                print(f"    {var} ({VARIANT_SHORT[var]}): "
                      f"tau={tau:.4f}, {time.time()-t0:.1f}s")
            except Exception as e:
                warnings.warn(f"Variant {var} seed={seed}: {e}")

        for target in valid_targets:
            target_idx = col_names.index(target)
            target_en = ZH_TO_EN.get(target, target)
            for var, (W_comb, tau) in variant_graphs.items():
                feats = _select_top_parents(
                    W_comb, target_idx, cfg.elec_top_m, col_names)
                try:
                    rmse, mae = run_forecast(
                        df_diff, split, feats, target,
                        seq_len=cfg.seq_len, max_epochs=cfg.max_epochs,
                        lr=cfg.fc_lr, hidden=cfg.hidden,
                        seed=seed, patience=cfg.patience)
                    all_rows.append(dict(
                        data="electricity", variant=var,
                        variant_name=VARIANT_SHORT[var], seed=seed,
                        target=target_en, top_m=cfg.elec_top_m,
                        n_features=len(feats), tau=tau,
                        rmse=rmse, mae=mae))
                except Exception as e:
                    warnings.warn(f"Forecast {var}/{target_en} "
                                  f"seed={seed}: {e}")

            # AllFeatures baseline
            try:
                rmse, mae = run_forecast(
                    df_diff, split, col_names, target,
                    seq_len=cfg.seq_len, max_epochs=cfg.max_epochs,
                    lr=cfg.fc_lr, hidden=cfg.hidden,
                    seed=seed, patience=cfg.patience)
                all_rows.append(dict(
                    data="electricity", variant="ALL",
                    variant_name="AllFeatures", seed=seed,
                    target=target_en, top_m=d,
                    n_features=d, tau=float("nan"),
                    rmse=rmse, mae=mae))
            except Exception:
                pass
        print(f"  seed={seed} done ({len(all_rows)} rows)")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        ensure_dir(cfg.output_dir)
        p = os.path.join(cfg.output_dir, "ablation_real.csv")
        df.to_csv(p, index=False)
        print(f">>> {p} ({len(df)} rows)")
    return df


# ====================================================================
# 7. plot_misspecification_robustness  (NEW — core selling point)
# ====================================================================
def plot_misspecification_robustness(df_synth: pd.DataFrame, out: str):
    """
    Plot F1 (y-axis) vs prior_accuracy (x-axis) for all 6 variants.
    Demonstrates how learned tau adapts to prior misspecification.
    """
    if df_synth.empty or df_synth["prior_acc"].nunique() < 2:
        print("  (skip misspecification plot: insufficient data)")
        return

    variants = [v for v in VARIANT_ORDER
                if v in df_synth["variant_name"].values]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for metric, ax, ylabel in [
        ("comb_f1",    axes[0], "F1 (combined graph)"),
        ("comb_auroc", axes[1], "AUROC (combined graph)"),
    ]:
        for v in variants:
            sub = df_synth[df_synth["variant_name"] == v]
            agg = sub.groupby("prior_acc").agg(
                y=(metric, "mean"), e=(metric, "std")).reset_index()
            lw = 3.0 if v == "FullModel" else 2.0
            ax.errorbar(
                agg["prior_acc"], agg["y"], yerr=agg["e"],
                label=v, color=VARIANT_COLORS.get(v, "grey"),
                marker=VARIANT_MARKERS.get(v, "x"),
                linewidth=lw, markersize=7, capsize=3)
        ax.set_xlabel("Prior Accuracy", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle(
        "Prior Misspecification Robustness\n"
        "(Learned tau adapts to prior quality — core PRCD-MAP insight)",
        fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(os.path.join(out, "fig_misspecification_robustness"))


# ====================================================================
# 8. Summary tables and additional figures
# ====================================================================
def _save_local(prefix):
    plt.savefig(prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(prefix + ".pdf", bbox_inches="tight")
    plt.close()
    print(f">>> {prefix}.png / .pdf")


def generate_summaries(df_synth, df_lorenz, df_real, out):
    print("\n" + "=" * 72)
    print("ABLATION STUDY SUMMARY")
    print("=" * 72)

    if not df_synth.empty:
        avail = [c for c in ["w0_auroc", "w0_f1", "comb_auroc", "comb_f1",
                              "comb_shd_norm"]
                 if c in df_synth.columns]
        g = df_synth.groupby(["variant", "variant_name"])
        agg = g.agg(
            **{f"{c}_mean": (c, "mean") for c in avail},
            **{f"{c}_std": (c, "std") for c in avail},
            tau_mean=("tau", "mean"), n=("seed", "count"),
        ).reset_index()
        p = os.path.join(out, "summary_synthetic_overall.csv")
        agg.to_csv(p, index=False)
        print(f">>> {p}")

        print("\n--- Synthetic: Overall by Variant ---")
        overall = df_synth.groupby("variant_name")[avail].mean()
        print(overall.round(4).to_string())

    if not df_lorenz.empty:
        g = df_lorenz.groupby(["variant", "variant_name"])
        agg = g.agg(
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
            f1_mean=("f1_opt", "mean"), f1_std=("f1_opt", "std"),
            shd_norm_mean=("shd_norm", "mean"),
            tau_mean=("tau", "mean"), n=("seed", "count"),
        ).reset_index()
        p = os.path.join(out, "summary_lorenz96.csv")
        agg.to_csv(p, index=False)
        print(f"\n>>> {p}")

        print("\n--- Lorenz-96: Overall by Variant ---")
        overall_l = df_lorenz.groupby("variant_name")[
            ["auroc", "f1_opt", "shd_norm"]].mean()
        print(overall_l.round(4).to_string())

    if not df_real.empty:
        g = df_real.groupby(["variant", "variant_name"])
        agg = g.agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            tau_mean=("tau", "mean"), n=("seed", "count"),
        ).reset_index()
        p = os.path.join(out, "summary_real_overall.csv")
        agg.to_csv(p, index=False)
        print(f"\n>>> {p}")

        print("\n--- Real Data: RMSE by Variant ---")
        overall_r = df_real.groupby("variant_name")[["rmse", "mae"]].mean()
        print(overall_r.sort_values("rmse").round(4).to_string())
    print()


def generate_figures(df_synth, df_lorenz, df_real, out):
    """Generate all ablation figures."""
    if not df_synth.empty:
        _fig_synth_bar(df_synth, out)
        plot_misspecification_robustness(df_synth, out)
        _fig_tau_analysis(df_synth, out)
    if not df_lorenz.empty:
        _fig_lorenz_bar(df_lorenz, out)
    if not df_real.empty:
        _fig_real_bar(df_real, out)
    _fig_combined_summary(df_synth, df_lorenz, df_real, out)


def _fig_synth_bar(df, out):
    """Bar chart: AUROC and F1 for each variant on synthetic data."""
    variants = [v for v in VARIANT_ORDER if v in df["variant_name"].values]
    n_v = len(variants)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for metric, ax, yl in [("comb_auroc", axes[0], "AUROC (combined)"),
                            ("comb_f1", axes[1], "F1 (combined)")]:
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
    fig.suptitle("Ablation: Graph Discovery Quality (Synthetic)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_local(os.path.join(out, "fig_synth_ablation_bar"))


def _fig_tau_analysis(df, out):
    """Learned tau vs prior accuracy for tau-learning variants."""
    tau_variants = ["FullModel", "NoL1", "LagsOnly"]
    sub_all = df[df["variant_name"].isin(tau_variants)].dropna(subset=["tau"])
    if sub_all.empty or sub_all["prior_acc"].nunique() < 2:
        return
    fig, ax1 = plt.subplots(figsize=(9, 6))
    c = {"FullModel": "#E74C3C", "NoL1": "#9B59B6", "LagsOnly": "#3498DB"}
    for v in tau_variants:
        sub = sub_all[sub_all["variant_name"] == v]
        if sub.empty:
            continue
        agg = sub.groupby("prior_acc").agg(
            tau_m=("tau", "mean"), tau_s=("tau", "std")).reset_index()
        ax1.errorbar(agg["prior_acc"], agg["tau_m"], yerr=agg["tau_s"],
                     color=c.get(v, "grey"),
                     marker=VARIANT_MARKERS.get(v, "o"),
                     lw=2.5, ms=8, capsize=4, label=f"tau ({v})")
    ax1.set_xlabel("Prior Accuracy", fontsize=13)
    ax1.set_ylabel("Learned Temperature tau", fontsize=13)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.set_title("Temperature Adaptation Across Variants",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_local(os.path.join(out, "fig_tau_analysis"))


def _fig_lorenz_bar(df, out):
    """Lorenz-96 ablation bar chart."""
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
    fig.suptitle("Ablation: Lorenz-96 (Nonlinear System)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_local(os.path.join(out, "fig_lorenz_ablation_bar"))


def _fig_real_bar(df, out):
    """Real data RMSE bar chart."""
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
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Ablation: Downstream Forecasting RMSE (Electricity)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _save_local(os.path.join(out, "fig_real_ablation_bar"))


def _fig_combined_summary(df_synth, df_lorenz, df_real, out):
    """Combined horizontal bar: synth AUROC, Lorenz F1, real RMSE."""
    has_s = not df_synth.empty
    has_l = not df_lorenz.empty
    has_r = not df_real.empty
    n_panels = sum([has_s, has_l, has_r])
    if n_panels < 2:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    pi = 0

    def _hbar(ax, df_sub, metric, title, xlabel, var_list):
        vs = [v for v in var_list if v in df_sub["variant_name"].values]
        means = [df_sub[df_sub["variant_name"] == v][metric].mean()
                 for v in vs]
        stds = [df_sub[df_sub["variant_name"] == v][metric].std()
                for v in vs]
        colors = [VARIANT_COLORS.get(v, "grey") for v in vs]
        ax.barh(range(len(vs)), means, xerr=stds, color=colors,
                alpha=0.85, capsize=3)
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels(vs, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", ls=":", alpha=0.4)
        ax.invert_yaxis()

    if has_s:
        _hbar(axes[pi], df_synth, "comb_auroc",
              "Synthetic\n(Graph Quality)", "AUROC", VARIANT_ORDER)
        pi += 1
    if has_l:
        _hbar(axes[pi], df_lorenz, "f1_opt",
              "Lorenz-96\n(Nonlinear)", "F1", VARIANT_ORDER)
        pi += 1
    if has_r:
        _hbar(axes[pi], df_real, "rmse",
              "Electricity\n(Downstream RMSE)", "RMSE",
              VARIANT_ORDER + ["AllFeatures"])
        pi += 1

    fig.suptitle("Ablation Study: Combined View",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_local(os.path.join(out, "fig_combined_summary"))


# ====================================================================
# 9. main + argparse
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Extended Ablation Study for PRCD-MAP")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny test run")
    parser.add_argument("--full", action="store_true",
                        help="Full sweep")
    parser.add_argument("--sub", type=str, default=None,
                        choices=["synthetic", "lorenz", "real", "hard_mask"],
                        help="Run a specific sub-experiment only")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--variants", nargs="+", type=str, default=None,
                        choices=["A", "B", "C", "D", "E", "F", "G"])
    args = parser.parse_args()

    if args.quick:
        cfg = cfg_quick()
    elif args.full:
        cfg = cfg_full()
    else:
        cfg = Cfg()

    if args.sub == "synthetic":
        cfg.do_lorenz = False; cfg.do_real = False
    elif args.sub == "lorenz":
        cfg.do_synthetic = False; cfg.do_real = False
    elif args.sub == "real":
        cfg.do_synthetic = False; cfg.do_lorenz = False
    elif args.sub == "hard_mask":
        cfg.variants = ["C", "F", "A"]

    if args.output:   cfg.output_dir = args.output
    if args.seeds:    cfg.seeds = args.seeds
    if args.variants: cfg.variants = args.variants

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 64)
    print(" Experiment 4: Extended Ablation Study")
    print("=" * 64)
    print(f"  variants     = {cfg.variants}")
    print(f"  seeds        = {cfg.seeds}")
    if cfg.do_synthetic:
        print(f"  synth dims   = {cfg.synth_dims}")
        print(f"  synth Ts     = {cfg.synth_Ts}")
        print(f"  prior_accs   = {cfg.synth_prior_accs}")
    if cfg.do_lorenz:
        print(f"  lorenz       = d={cfg.lorenz_d}, T={cfg.lorenz_T}")
    if cfg.do_real:
        print(f"  elec targets = {cfg.elec_targets}")
    print(f"  output       = {cfg.output_dir}")
    for v in cfg.variants:
        print(f"    {v}: {VARIANT_NAMES[v]}")
    print("=" * 64)

    t_global = time.time()
    df_synth = pd.DataFrame()
    df_lorenz = pd.DataFrame()
    df_real = pd.DataFrame()

    if cfg.do_synthetic:
        df_synth = run_synthetic_ablation(cfg)
    if cfg.do_lorenz:
        df_lorenz = run_lorenz_ablation(cfg)
    if cfg.do_real:
        df_real = run_real_ablation(cfg)

    generate_summaries(df_synth, df_lorenz, df_real, cfg.output_dir)
    generate_figures(df_synth, df_lorenz, df_real, cfg.output_dir)

    # Save combined CSV
    dfs = []
    if not df_synth.empty:  dfs.append(("synthetic", df_synth))
    if not df_lorenz.empty: dfs.append(("lorenz96", df_lorenz))
    if not df_real.empty:   dfs.append(("real", df_real))
    if dfs:
        parts = []
        for name, df in dfs:
            d2 = df.copy(); d2["experiment"] = name; parts.append(d2)
        combined = pd.concat(parts, ignore_index=True)
        p = os.path.join(cfg.output_dir, "exp4_full_results.csv")
        combined.to_csv(p, index=False)
        print(f"\n>>> Combined: {p} ({len(combined)} rows)")

    elapsed = time.time() - t_global
    print(f"\n>>> Experiment 4 complete in {elapsed:.1f}s")
    print(f">>> Results in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
