"""
Realised constants verification: directly measure quantities the paper claims.

Paper claims (post-PAT-fix):
  - min(c_ij) realised = 0.51 (sigmoid floor; was 0.31 before)
  - max(c_ij) realised = 1.50
  - mean(c_ij) ~ 0.94
  - K_realised = 3 * c_max/c_min ~ 8.8
  - lambda_min(Sigma_hat) >= 0.4
  - C_1 ~ O(lambda_min^-2) ~ 6.25 worst case, ~0.09 realised
  - Delta_proxy realised <= 0.045 at acc=0.6

This script measures all these from the converged PRCD-MAP fixed point at
d=20, T=500, acc=0.6, 10 seeds, and prints a self-contained verification
table for the appendix.

Usage:
    python verify_realised_constants.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
sys.path.insert(0, SRC_DIR)

from utils import (set_seed, make_er_dag, make_lag_matrices, simulate_svar_linear,
                   standardize, gen_prior, compute_dual_metrics)
from utils_trust import make_lag_tensors_with_mask
from model_linear_trust import PRCD_MAP_Trust, train_prcd_trust_alm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--acc", type=float, default=0.6)
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--out", type=str, default="results/realised_constants.csv")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        print(f"\n=== seed={seed} ===", flush=True)
        set_seed(seed)
        W0_true = make_er_dag(args.d, edge_prob=0.15, seed=seed)
        Wk_true = make_lag_matrices(args.d, args.K, edge_prob=0.10, seed=seed)
        X = simulate_svar_linear(args.T, W0_true, Wk_true, seed=seed)
        if X is None:
            continue
        X = standardize(X)
        P_prior = gen_prior(W0_true, Wk_true, acc=args.acc, seed=seed)

        # Compute lambda_min(Sigma_hat) from the lag-augmented Gram matrix
        # Z_t = (x_t, x_{t-1}, ..., x_{t-K})  in R^{(K+1)d}
        T_eff = args.T - args.K
        Z = np.zeros((T_eff, (args.K + 1) * args.d))
        for k in range(args.K + 1):
            Z[:, k*args.d:(k+1)*args.d] = X[args.K - k : args.K - k + T_eff]
        Sigma_hat = (Z.T @ Z) / T_eff
        eigs = np.linalg.eigvalsh(Sigma_hat)
        lambda_min_sigma = float(eigs.min())
        lambda_max_sigma = float(eigs.max())

        # Run PRCD-MAP with trust to convergence
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

        train_prcd_trust_alm(
            model, X_t, X_lags,
            max_iter=args.max_iter, inner_iter=args.inner_iter, lr=1e-2,
            rho_0=1.0, gamma=3.0, tol=1e-6,
            verbose=False, postprocess=False,
            obs_mask=obs_mask,
        )

        # Extract realised c_ij distribution at the converged tau*
        with torch.no_grad():
            if model.use_per_edge_trust:
                tau_matrix = model.trust_module(model._prior_logits.detach(),
                                                model.W0.detach(),
                                                model.off_diag_mask)
            else:
                tau_groups = model.trust_module()
                tau_matrix = tau_groups[model.tau_group_idx].view(args.d, args.d) * model.off_diag_mask
            P_hat = model.calibrated_prior(tau_matrix)
            coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * model.off_diag_mask
            mask = (model.off_diag_mask > 0)
            cvals = coeff[mask].detach().cpu().numpy()
            P_vals = P_hat[mask].detach().cpu().numpy()
            tau_vals = tau_matrix[mask].detach().cpu().numpy()

        c_min = float(cvals.min())
        c_max = float(cvals.max())
        c_med = float(np.median(cvals))
        c_mean = float(cvals.mean())
        c_q05 = float(np.quantile(cvals, 0.05))
        c_q95 = float(np.quantile(cvals, 0.95))
        K_realised = 3.0 * c_max / max(c_min, 1e-6)

        # C_1 ~ O(lambda_min^-2)
        C1_realised = 1.0 / max(lambda_min_sigma ** 2, 1e-6)

        # Delta_proxy realised: ||tau*||_inf^2 * ||P_prior - P_true||_F^2 / T * C_1
        # P_true = adjacency of W0_true
        A_star = (np.abs(W0_true) > 0).astype(float)
        np.fill_diagonal(A_star, 0)
        # P_prior is already off-diagonal-zeroed by gen_prior
        diff = P_prior - A_star
        np.fill_diagonal(diff, 0)
        prior_dist_sq = float((diff ** 2).sum())
        tau_inf_sq = float(np.max(np.abs(tau_vals)) ** 2) if len(tau_vals) else 0.0
        Delta_proxy = C1_realised * tau_inf_sq * prior_dist_sq / args.T

        rows.append({
            "seed": seed, "d": args.d, "T": args.T, "acc": args.acc,
            "lambda_min_sigma": lambda_min_sigma,
            "lambda_max_sigma": lambda_max_sigma,
            "c_min": c_min, "c_max": c_max, "c_mean": c_mean, "c_med": c_med,
            "c_q05": c_q05, "c_q95": c_q95,
            "K_realised": K_realised,
            "tau_inf_sq": tau_inf_sq,
            "prior_dist_sq": prior_dist_sq,
            "C1_realised": C1_realised,
            "Delta_proxy": Delta_proxy,
        })
        print(f"  c_ij: min={c_min:.4f} med={c_med:.4f} mean={c_mean:.4f} max={c_max:.4f}", flush=True)
        print(f"  K_realised={K_realised:.2f}, lambda_min={lambda_min_sigma:.4f}, C1={C1_realised:.3f}", flush=True)
        print(f"  Delta_proxy={Delta_proxy:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {args.out}")

    if len(df):
        print("\n=== Realised constants summary (mean ± std across seeds) ===")
        for col in ["c_min", "c_max", "c_mean", "c_med", "K_realised",
                    "lambda_min_sigma", "C1_realised", "Delta_proxy"]:
            print(f"  {col:20s}: {df[col].mean():.4f} ± {df[col].std():.4f}")

        print(f"\n  Paper claims:")
        print(f"    c_min realised = 0.51   (you measured: {df['c_min'].mean():.4f})")
        print(f"    K_realised     ~ 8.8    (you measured: {df['K_realised'].mean():.2f})")
        print(f"    lambda_min(Sigma) >= 0.4 (you measured: {df['lambda_min_sigma'].mean():.4f})")
        print(f"    Delta_proxy    <= 0.045 (you measured: {df['Delta_proxy'].mean():.4f})")


if __name__ == "__main__":
    main()
