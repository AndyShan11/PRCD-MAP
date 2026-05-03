"""
Cor 4 / T-3 verification: Delta_proxy realised across the acc grid.

Paper claims (Eq~\ref{eq:delta_proxy_uniform}):
  Delta_proxy <= C_1 * tau_max^2 * d^2 / T * acc * (1 - acc)

with realised Delta_proxy ~ 0.045 at acc=0.6.

PAT also flagged that as acc->1, the absorption d^2/||P_prior-P_true||_F^2
in C_1 diverges, making the bound vacuous near the perfect-prior boundary.

This script measures realised Delta_proxy at acc in {0.1, 0.3, 0.5, 0.6,
0.7, 0.9, 0.95}, with 5 seeds each, at d=20, T=500. We track:
  - actual ||P_prior - P_true||_F^2 (controls the absorption denominator)
  - actual ||tau*||_inf^2 (the EB-driven trust)
  - the product (the "realised proxy gap")

Usage:
    python verify_cor4_proxy_grid.py --seeds 0 1 2 3 4
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
sys.path.insert(0, SRC_DIR)

from utils import (set_seed, make_er_dag, make_lag_matrices, simulate_svar_linear,
                   standardize, gen_prior)
from utils_trust import make_lag_tensors_with_mask
from model_linear_trust import PRCD_MAP_Trust, train_prcd_trust_alm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--accs", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 0.95])
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--out", type=str, default="results/cor4_proxy_grid.csv")
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

            A_star = (np.abs(W0_true) > 0).astype(float)
            np.fill_diagonal(A_star, 0)
            diff = P_prior - A_star
            np.fill_diagonal(diff, 0)
            prior_dist_sq = float((diff ** 2).sum())

            # Compute lambda_min(Sigma_hat)
            T_eff = args.T - args.K
            Z = np.zeros((T_eff, (args.K + 1) * args.d))
            for k in range(args.K + 1):
                Z[:, k*args.d:(k+1)*args.d] = X[args.K - k : args.K - k + T_eff]
            Sigma_hat = (Z.T @ Z) / T_eff
            lambda_min = float(np.linalg.eigvalsh(Sigma_hat).min())
            C1_realised = 1.0 / max(lambda_min ** 2, 1e-6)

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
                train_prcd_trust_alm(
                    model, X_t, X_lags,
                    max_iter=args.max_iter, inner_iter=args.inner_iter, lr=1e-2,
                    rho_0=1.0, gamma=3.0, tol=1e-6,
                    verbose=False, postprocess=False,
                    obs_mask=obs_mask,
                )
            except Exception as e:
                warnings.warn(f"train failed: {e}")
                continue

            with torch.no_grad():
                if model.use_per_edge_trust:
                    tau_matrix = model.trust_module(model._prior_logits.detach(),
                                                    model.W0.detach(),
                                                    model.off_diag_mask)
                else:
                    tau_groups = model.trust_module()
                    tau_matrix = tau_groups[model.tau_group_idx].view(args.d, args.d) * model.off_diag_mask
                tau_inf = float(torch.max(torch.abs(tau_matrix)).item())

            tau_inf_sq = tau_inf ** 2
            Delta_proxy_realised = C1_realised * tau_inf_sq * prior_dist_sq / args.T
            # Uniform bound: C1 * tau_max^2 * d^2 / T * acc(1-acc)
            tau_max_sq = 3.0 ** 2
            Delta_proxy_uniform_bound = C1_realised * tau_max_sq * (args.d ** 2) / args.T * acc * (1 - acc)

            rows.append({
                "acc": acc, "seed": seed, "d": args.d, "T": args.T,
                "lambda_min_sigma": lambda_min,
                "C1_realised": C1_realised,
                "tau_inf": tau_inf, "tau_inf_sq": tau_inf_sq,
                "prior_dist_sq": prior_dist_sq,
                "Delta_proxy_realised": Delta_proxy_realised,
                "Delta_proxy_uniform_bound": Delta_proxy_uniform_bound,
                "ratio_realised_over_bound": Delta_proxy_realised / max(Delta_proxy_uniform_bound, 1e-9),
            })
            print(f"  Delta_proxy_realised = {Delta_proxy_realised:.4f}, "
                  f"uniform_bound = {Delta_proxy_uniform_bound:.4f}, "
                  f"ratio = {rows[-1]['ratio_realised_over_bound']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {args.out}")

    if len(df):
        print("\n=== Delta_proxy realised vs uniform bound across acc grid ===")
        piv1 = df.pivot_table(values="Delta_proxy_realised", index="acc", aggfunc=["mean", "std"])
        print("\nRealised:"); print(piv1.to_string(float_format="%.4f"))
        piv2 = df.pivot_table(values="Delta_proxy_uniform_bound", index="acc", aggfunc=["mean", "std"])
        print("\nUniform bound:"); print(piv2.to_string(float_format="%.4f"))
        piv3 = df.pivot_table(values="tau_inf", index="acc", aggfunc=["mean"])
        print("\ntau_inf (max trust at convergence):"); print(piv3.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
