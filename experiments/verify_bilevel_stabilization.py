"""
Bilevel verification (Asm 2): track active-set stability across ALM iterations.

Paper claims (App bilevel_sketch):
  "the support stabilizes by iteration ~20 across all settings"

This script verifies the claim by tracking the active-set Hamming distance
across consecutive ALM outer iterations:
  H_t = sum_{(i,j)} I[ supp(W_t)[i,j] != supp(W_{t-1})[i,j] ]

We replicate the train_prcd_trust_alm outer loop but record support after
each outer iter, then report the iteration at which Hamming-to-final == 0.

Usage:
    python verify_bilevel_stabilization.py --seeds 0 1 2
"""
import os, sys, time, argparse, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
sys.path.insert(0, SRC_DIR)

from utils import (set_seed, make_er_dag, make_lag_matrices, simulate_svar_linear,
                   standardize, gen_prior)
from utils_trust import make_lag_tensors_with_mask
from model_linear_trust import PRCD_MAP_Trust


def trace_alm(model, X_t, X_lags, max_iter=35, inner_iter=400,
              rho_0=1.0, gamma=3.0, rho_max=1e6, lr=1e-2,
              tau_eb_steps=8, tau_eb_lr=1e-3,
              support_thr_ratio=0.10, obs_mask=None):
    """Replicates train_prcd_trust_alm but records support after each outer iter."""
    device = next(model.parameters()).device
    X_t = X_t.to(device)
    X_lags = [x.to(device) for x in X_lags]
    if obs_mask is not None:
        obs_mask = obs_mask.to(device)

    svar_params = [model.W0] + list(model.Wk.parameters())
    trust_params = list(model.trust_module.parameters())

    rho = float(rho_0); alpha = 0.0
    support_history = []
    h_history = []

    for it in range(max_iter):
        optimizer = optim.Adam(svar_params, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=inner_iter, eta_min=lr * 0.01)

        for step in range(inner_iter):
            loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau = \
                model.compute_losses(X_t, X_lags, rho, alpha, obs_mask)
            optimizer.zero_grad(set_to_none=True)
            loss_alm.backward()
            ec = 5.0 * max(1.0, math.log1p(rho))
            torch.nn.utils.clip_grad_norm_(svar_params, max_norm=ec)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            W0_abs = torch.abs(model.W0 * model.off_diag_mask)
            w_max = W0_abs.max().clamp(min=1e-6)
            thr = support_thr_ratio * w_max
            supp = (W0_abs > thr).cpu().numpy().astype(int)
            support_history.append(supp)
            h_now = float(model._compute_h_w0().detach().cpu().item())
            h_history.append(h_now)

        # EB update
        if model.learn_tau and tau_eb_steps > 0:
            trust_optimizer = optim.Adam(trust_params, lr=tau_eb_lr)
            for _ in range(tau_eb_steps):
                eb_loss = model.compute_eb_objective(X_t, X_lags)
                trust_optimizer.zero_grad(set_to_none=True)
                eb_loss.backward()
                torch.nn.utils.clip_grad_norm_(trust_params, max_norm=1.0)
                trust_optimizer.step()

        alpha = alpha + rho * h_now
        rho = min(gamma * rho, rho_max)

    return support_history, h_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--accs", type=float, nargs="+", default=[0.4, 0.6, 0.9])
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--out", type=str, default="results/bilevel_stabilization.csv")
    args = parser.parse_args()

    rows = []
    for acc in args.accs:
        for seed in args.seeds:
            print(f"\n=== acc={acc} seed={seed} ===", flush=True)
            set_seed(seed)
            W0_true = make_er_dag(args.d, edge_prob=0.15, seed=seed)
            Wk_true = make_lag_matrices(args.d, args.K, edge_prob=0.10, seed=seed)
            X = simulate_svar_linear(args.T, W0_true, Wk_true, seed=seed)
            if X is None:
                continue
            X = standardize(X)
            P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)

            X_t, X_lags, obs_mask = make_lag_tensors_with_mask(X, args.K)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_t = X_t.to(dev)
            X_lags = [x.to(dev) for x in X_lags]
            if obs_mask is not None:
                obs_mask = obs_mask.to(dev)

            model = PRCD_MAP_Trust(
                num_vars=args.d, lag_k=args.K, P_prior=P_prior,
                lambda1=0.001, lambda2=0.01,
                learn_tau=True, tau0=1.0,
                tau_min=0.05, tau_max=3.0,
                loss_type="huber", prior_l1_weight=True,
                n_tau_groups=4, trust_feat_dim=16, trust_n_layers=2,
                trust_lite=False,
            ).to(dev)

            try:
                support_hist, h_hist = trace_alm(
                    model, X_t, X_lags,
                    max_iter=args.max_iter, inner_iter=args.inner_iter,
                    obs_mask=obs_mask)
            except Exception as e:
                warnings.warn(f"trace failed for acc={acc} s{seed}: {e}")
                continue

            final_support = support_hist[-1]
            for it_idx, supp in enumerate(support_hist):
                hamming = int(np.sum(supp != final_support))
                rows.append({
                    "acc": acc, "seed": seed, "outer_iter": it_idx + 1,
                    "support_size": int(supp.sum()),
                    "hamming_to_final": hamming,
                    "h_dagma": h_hist[it_idx],
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {args.out}")

    if len(df):
        print("\n=== First iteration where support stabilizes (Hamming=0 to final, sustained) ===")
        for acc in df["acc"].unique():
            for seed in df["seed"].unique():
                sub = df[(df["acc"] == acc) & (df["seed"] == seed)].sort_values("outer_iter")
                stable_iter = None
                for _, row in sub.iterrows():
                    if row["hamming_to_final"] == 0:
                        stable_iter = int(row["outer_iter"])
                        break
                print(f"  acc={acc:.1f} seed={seed}: stable from iter {stable_iter}")


if __name__ == "__main__":
    main()
