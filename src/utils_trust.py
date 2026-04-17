"""
exp_utils.py — Shared utilities for PRCD-MAP trust propagation experiments (0415).

Imports from original code directory for baselines, data gen, metrics.
Adds trust-propagation-specific wrappers.
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

# ---- Import from src/utils.py (core utilities, baselines, metrics) ----
import importlib.util as _ilu

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_orig_exp_utils_path = os.path.join(_SRC_DIR, "utils.py")

_spec = _ilu.spec_from_file_location("_core_utils", _orig_exp_utils_path)
_orig = _ilu.module_from_spec(_spec)
sys.modules["_core_utils"] = _orig
_spec.loader.exec_module(_orig)

# Re-export everything from original exp_utils
set_seed = _orig.set_seed
make_lag_tensors = _orig.make_lag_tensors
standardize = _orig.standardize
ensure_dir = _orig.ensure_dir
fmt_time = _orig.fmt_time
save_fig = _orig.save_fig
make_er_dag = _orig.make_er_dag
make_ba_dag = _orig.make_ba_dag
make_lag_matrices = _orig.make_lag_matrices
simulate_svar_linear = _orig.simulate_svar_linear
simulate_svar_nonlinear = _orig.simulate_svar_nonlinear
gen_prior = _orig.gen_prior
gen_prior_from_truth = _orig.gen_prior_from_truth
binarize_prior_to_mask = _orig.binarize_prior_to_mask
compute_all_metrics = _orig.compute_all_metrics
combine_W0_Wk = _orig.combine_W0_Wk
compute_dual_metrics = _orig.compute_dual_metrics
run_dynotears = _orig.run_dynotears
run_pcmci_plus = _orig.run_pcmci_plus
run_varlingam = _orig.run_varlingam
run_prcd_map = _orig.run_prcd_map
generate_lorenz96 = _orig.generate_lorenz96
lorenz96_ground_truth = _orig.lorenz96_ground_truth
load_electricity = _orig.load_electricity
print_rhino_table = _orig.print_rhino_table
inject_missing = _orig.inject_missing
make_lag_tensors_with_mask = _orig.make_lag_tensors_with_mask
COLORS = _orig.COLORS
MARKERS = _orig.MARKERS
ZH_TO_EN = _orig.ZH_TO_EN

# Optional: compute_significance may not exist in older code versions
compute_significance = getattr(_orig, "compute_significance", None)

# Conditional re-exports
HAS_NGC = getattr(_orig, "HAS_NGC", False)
run_ngc = getattr(_orig, "run_ngc", lambda *a, **k: (None, None, None))

HAS_RHINO = getattr(_orig, "HAS_RHINO", False)
run_rhino = getattr(_orig, "run_rhino", lambda *a, **k: (None, None, None))

load_causaltime = getattr(_orig, "load_causaltime", lambda *a, **k: (None, None))
load_netsim = getattr(_orig, "load_netsim", lambda *a, **k: (None, None))

run_prcd_map_nam = getattr(_orig, "run_prcd_map_nam", lambda *a, **k: (None, None, None))

# ---- Local model imports ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_linear_trust import PRCD_MAP_Trust, train_prcd_trust_alm
from model_nam_trust import PRCD_MAP_NAM_Trust, train_prcd_nam_trust_alm

# ---- Updated colors/markers ----
COLORS.update({
    "PRCD-MAP(trust)": "#C0392B",
    "PRCD-MAP(trust+NAM)": "#8E44AD",
    "PRCD-MAP(per-group)": "#E74C3C",
})
MARKERS.update({
    "PRCD-MAP(trust)": "o",
    "PRCD-MAP(trust+NAM)": "p",
    "PRCD-MAP(per-group)": "s",
})


# ====================================================================
# PRCD-MAP Trust wrapper
# ====================================================================

def run_prcd_trust(X, P_prior, d, K, lambda1=0.001, lambda2=0.01,
                   learn_tau=True, tau0=1.0,
                   max_iter=35, inner_iter=400, lr=1e-2,
                   seed=0, loss_type="huber", n_tau_groups=4,
                   trust_feat_dim=16, trust_n_layers=2, trust_lite=False,
                   verbose=False, score_calibration=False):
    """Run PRCD-MAP with structure-aware trust propagation."""
    set_seed(seed)
    X_t, X_lags, obs_mask = make_lag_tensors_with_mask(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]
    if obs_mask is not None:
        obs_mask = obs_mask.to(dev)

    model = PRCD_MAP_Trust(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau, tau0=tau0,
        tau_min=0.05, tau_max=3.0,
        loss_type=loss_type, prior_l1_weight=True,
        n_tau_groups=n_tau_groups,
        trust_feat_dim=trust_feat_dim,
        trust_n_layers=trust_n_layers,
        trust_lite=trust_lite,
    ).to(dev)

    W0, Wk, tau = train_prcd_trust_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=verbose, postprocess=False,
        obs_mask=obs_mask,
    )

    if score_calibration:
        W0 = _calibrate_scores(W0)
        Wk = [_calibrate_scores(wk) for wk in Wk]

    return W0, Wk, tau


def run_prcd_nam_trust(X, P_prior, d, K, lambda1=0.001, lambda2=0.01,
                       learn_tau=True, edge_hidden=16, edge_layers=2,
                       max_iter=35, inner_iter=400, lr=5e-4, seed=0,
                       n_tau_groups=4, verbose=False):
    """Run PRCD-MAP NAM with structure-aware trust propagation."""
    set_seed(seed)
    X_t, X_lags = make_lag_tensors(X, K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = X_t.to(dev)
    X_lags = [x.to(dev) for x in X_lags]

    model = PRCD_MAP_NAM_Trust(
        num_vars=d, lag_k=K, P_prior=P_prior,
        lambda1=lambda1, lambda2=lambda2,
        learn_tau=learn_tau,
        tau_min=0.05, tau_max=3.0,
        edge_hidden=edge_hidden, edge_layers=edge_layers,
        n_tau_groups=n_tau_groups,
    ).to(dev)

    W0, Wk, tau = train_prcd_nam_trust_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=verbose,
    )
    return W0, Wk, tau


def _calibrate_scores(W):
    """Row-wise normalization for better AUROC."""
    W_abs = np.abs(W)
    row_max = W_abs.max(axis=1, keepdims=True)
    row_max[row_max < 1e-8] = 1.0
    return W_abs / row_max


# ====================================================================
# Experiment helpers
# ====================================================================

def run_single_setting(
    X, d, K, W0_true, Wk_true, P_prior, seed,
    lambda1=0.001, lambda2=0.01, max_iter=35, inner_iter=400, lr=1e-2,
    do_baselines=True, do_trust=True, do_per_group=True,
    do_nam_trust=False, do_dynotears=True, do_pcmci=True,
    do_varlingam=True, verbose=False,
):
    """Run all methods on a single (X, P_prior) setting, return list of dicts."""
    results = []
    B_w0 = (np.abs(W0_true) > 1e-10).astype(float)
    B_comb = B_w0.copy().astype(int)
    for Wk_t in Wk_true:
        B_comb = np.maximum(B_comb, (np.abs(Wk_t) > 1e-10).astype(int))

    def _eval(name, W0_est, Wk_est):
        if W0_est is None:
            return
        met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
        met["method"] = name
        met["seed"] = seed
        results.append(met)

    # PRCD-MAP (trust propagation)
    if do_trust:
        try:
            W0, Wk, tau = run_prcd_trust(
                X, P_prior, d, K, lambda1=lambda1, lambda2=lambda2,
                max_iter=max_iter, inner_iter=inner_iter, lr=lr, seed=seed,
                verbose=verbose)
            _eval("PRCD-MAP(trust)", W0, Wk)
        except Exception as e:
            warnings.warn(f"PRCD-MAP(trust) failed: {e}")

    # PRCD-MAP (per-group tau, original)
    if do_per_group:
        try:
            W0, Wk, tau = run_prcd_map(
                X, P_prior, d, K, lambda1=lambda1, lambda2=lambda2,
                max_iter=max_iter, inner_iter=inner_iter, lr=lr, seed=seed,
                verbose=verbose)
            _eval("PRCD-MAP(per-group)", W0, Wk)
        except Exception as e:
            warnings.warn(f"PRCD-MAP(per-group) failed: {e}")

    # NAM + Trust
    if do_nam_trust and d <= 10:  # NAM仅d<=10, d>=20的380+MLPs太慢
        try:
            W0, Wk, tau = run_prcd_nam_trust(
                X, P_prior, d, K, lambda1=lambda1, lambda2=lambda2,
                max_iter=max_iter, inner_iter=inner_iter, lr=5e-4, seed=seed,
                verbose=verbose)
            _eval("PRCD-MAP(trust+NAM)", W0, Wk)
        except Exception as e:
            warnings.warn(f"PRCD-MAP(trust+NAM) failed: {e}")

    # Baselines
    if do_baselines and do_dynotears:
        try:
            W0, Wk = run_dynotears(X, d, K, seed=seed)
            _eval("DYNOTEARS", W0, Wk)
        except Exception as e:
            warnings.warn(f"DYNOTEARS failed: {e}")

    if do_baselines and do_pcmci:
        try:
            W0, Wk = run_pcmci_plus(X, d, K, seed=seed)
            _eval("PCMCI+", W0, Wk)
        except Exception as e:
            warnings.warn(f"PCMCI+ failed: {e}")

    if do_baselines and do_varlingam:
        try:
            W0, Wk = run_varlingam(X, d, K, seed=seed)
            _eval("VARLiNGAM", W0, Wk)
        except Exception as e:
            warnings.warn(f"VARLiNGAM failed: {e}")

    return results


def aggregate_results(all_results, group_cols=None):
    """Aggregate results across seeds."""
    if not all_results:
        return pd.DataFrame()
    df = pd.DataFrame(all_results)
    if group_cols is None:
        group_cols = [c for c in df.columns if c not in
                      ["seed", "auroc", "auprc", "f1_opt", "shd", "shd_norm",
                       "prec_opt", "rec_opt", "best_thr", "f1_topk", "shd_topk", "k_true",
                       "w0_auroc", "w0_auprc", "w0_f1_opt", "w0_shd", "w0_shd_norm",
                       "w0_prec_opt", "w0_rec_opt", "w0_best_thr", "w0_f1_topk", "w0_shd_topk", "w0_k_true",
                       "comb_auroc", "comb_auprc", "comb_f1_opt", "comb_shd", "comb_shd_norm",
                       "comb_prec_opt", "comb_rec_opt", "comb_best_thr", "comb_f1_topk", "comb_shd_topk", "comb_k_true"]]
    metric_cols = [c for c in df.columns if c not in group_cols and c != "seed"]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    return agg
