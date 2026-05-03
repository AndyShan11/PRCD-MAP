"""
Method-#3 verification: threshold-based indicator soft labels for the EB
agreement loss, as an alternative to the current linear normalization
|W*|/max|W*|.

PAT noted that the linear normalization can under-represent small-effect
true edges (weight 0.2 when max=1.0 -> soft label 0.2), so a high-confidence
correct prior (P=0.9) incurs cross-entropy penalty rather than reward.

We monkey-patch model.compute_eb_objective by mirroring the original
implementation but replacing the W0_prob construction:
  Original (linear):    W0_prob = (|W0_adj| / max|W0_adj|).clamp(1e-6, 1-1e-6)
  Alternative (threshold): W0_prob = sigmoid((|W0_adj| - thr*max) * sharpness/max)

The rest of the EB loss (Laplace log-det, prior L1/L2, trust reg) stays identical.

Usage:
    python verify_method3_threshold_labels.py --seeds 0 1 2 3 4
"""
import os, sys, time, argparse, warnings, types
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


def make_threshold_eb_objective(thresh_ratio: float = 0.30, sharpness: float = 20.0):
    """Replacement for compute_eb_objective using threshold-indicator W0_prob."""
    def patched(self, X_t, X_lags):
        dev = X_t.device
        T, d = X_t.shape
        tau_matrix = self._compute_tau_matrix()
        Omega = self.omega_mask(tau_matrix)
        W0_adj = self.get_W0_adj().detach()

        # Prior L2 (mirrors original)
        loss_prior = torch.tensor(0.0, device=dev)
        for k in range(self.K):
            Wk_det = self.Wk[k].detach()
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (Wk_det ** 2))
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega_w0 * (W0_adj ** 2))

        # Prior L1 (mirrors original)
        if self.prior_l1_weight:
            P_hat_l1 = self.calibrated_prior(tau_matrix)
            coeff = torch.clamp(1.5 - P_hat_l1, 0.1, 1.5) * self.off_diag_mask
            loss_l1 = self.lambda1 * torch.sum(coeff * torch.abs(W0_adj))
        else:
            loss_l1 = self.lambda1 * torch.sum(torch.abs(W0_adj))

        # Laplace log-det (mirrors original)
        X_t_det = X_t.detach()
        data_hess_row = (X_t_det ** 2).sum(0) / (T * d)
        log_det_term = torch.tensor(0.0, device=dev)
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            H_w0 = data_hess_row.unsqueeze(1).expand(d, d) + self.lambda2 * Omega_w0
            log_det_term = log_det_term + torch.sum(
                torch.log(H_w0.clamp(min=1e-10)) * self.off_diag_mask
            )
        for k_idx in range(self.K):
            X_lag_det = X_lags[k_idx].detach()
            data_hess_k = (X_lag_det ** 2).sum(0) / (T * d)
            H_wk = data_hess_k.unsqueeze(1).expand(d, d) + self.lambda2 * Omega
            log_det_term = log_det_term + torch.sum(torch.log(H_wk.clamp(min=1e-10)))

        # *** Agreement loss with THRESHOLD-INDICATOR soft labels ***
        P_hat = self.calibrated_prior(tau_matrix)
        W0_abs = torch.abs(W0_adj)
        w_max = W0_abs.max().clamp(min=1e-6)
        threshold = thresh_ratio * w_max
        W0_prob = torch.sigmoid((W0_abs - threshold) * sharpness / w_max)
        W0_prob = W0_prob.clamp(1e-6, 1.0 - 1e-6) * self.off_diag_mask
        P_hat_safe = P_hat.clamp(1e-6, 1.0 - 1e-6)
        agreement_loss = -torch.sum(
            (W0_prob * torch.log(P_hat_safe) + (1.0 - W0_prob) * torch.log(1.0 - P_hat_safe))
            * self.off_diag_mask
        )

        # Trust module parameter regularization (mirrors original)
        trust_reg = torch.tensor(0.0, device=dev)
        for p in self.trust_module.parameters():
            trust_reg = trust_reg + 0.01 * torch.sum(p ** 2)

        return agreement_loss + 0.5 * log_det_term + loss_prior + loss_l1 + trust_reg
    return patched


def run_one(X, P_prior, d, K, seed, soft_label_mode="linear",
            thresh_ratio=0.30, max_iter=35, inner_iter=400, lr=1e-2,
            lambda1=0.001, lambda2=0.01):
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
        learn_tau=True, tau0=1.0,
        tau_min=0.05, tau_max=3.0,
        loss_type="huber", prior_l1_weight=True,
        n_tau_groups=4, trust_feat_dim=16, trust_n_layers=2,
        trust_lite=(d > 50),
    ).to(dev)

    if soft_label_mode == "threshold":
        model.compute_eb_objective = types.MethodType(
            make_threshold_eb_objective(thresh_ratio=thresh_ratio), model)

    W0, Wk, tau = train_prcd_trust_alm(
        model, X_t, X_lags,
        max_iter=max_iter, inner_iter=inner_iter, lr=lr,
        rho_0=1.0, gamma=3.0, tol=1e-6,
        verbose=False, postprocess=False,
        obs_mask=obs_mask,
    )
    return W0, Wk, tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--Ts", type=int, nargs="+", default=[100, 200, 500])
    parser.add_argument("--accs", type=float, nargs="+", default=[0.4, 0.6, 0.9])
    parser.add_argument("--thresh_ratios", type=float, nargs="+", default=[0.20, 0.30])
    parser.add_argument("--max_iter", type=int, default=35)
    parser.add_argument("--inner_iter", type=int, default=400)
    parser.add_argument("--out", type=str, default="results/method3_threshold_labels.csv")
    args = parser.parse_args()

    rows = []
    for T in args.Ts:
        for acc in args.accs:
            for seed in args.seeds:
                set_seed(seed)
                W0_true = make_er_dag(args.d, edge_prob=0.15, seed=seed)
                Wk_true = make_lag_matrices(args.d, args.K, edge_prob=0.10, seed=seed)
                X = simulate_svar_linear(T, W0_true, Wk_true, seed=seed)
                if X is None:
                    continue
                X = standardize(X)
                P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)

                # Variant A: linear soft labels (current paper main)
                try:
                    t0 = time.time()
                    W0, Wk, _ = run_one(X, P_prior, args.d, args.K, seed,
                                        soft_label_mode="linear",
                                        max_iter=args.max_iter, inner_iter=args.inner_iter)
                    dt = time.time() - t0
                    met = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
                    rows.append({"variant": "linear", "thresh": None,
                                 "T": T, "acc": acc, "seed": seed,
                                 "runtime": dt, **met})
                    print(f"[T={T} acc={acc} s{seed}] linear:    AUROC={met['auroc']:.4f}", flush=True)
                except Exception as e:
                    warnings.warn(f"linear failed: {e}")

                # Variant B: threshold-indicator soft labels (multiple thresholds)
                for thr in args.thresh_ratios:
                    try:
                        t0 = time.time()
                        W0, Wk, _ = run_one(X, P_prior, args.d, args.K, seed,
                                            soft_label_mode="threshold",
                                            thresh_ratio=thr,
                                            max_iter=args.max_iter, inner_iter=args.inner_iter)
                        dt = time.time() - t0
                        met = compute_dual_metrics(W0_true, Wk_true, W0, Wk)
                        rows.append({"variant": f"threshold_{thr}", "thresh": thr,
                                     "T": T, "acc": acc, "seed": seed,
                                     "runtime": dt, **met})
                        print(f"[T={T} acc={acc} s{seed}] thresh={thr}:  AUROC={met['auroc']:.4f}", flush=True)
                    except Exception as e:
                        warnings.warn(f"threshold {thr} failed: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {args.out}")

    if len(df):
        print("\n=== Summary: AUROC by (variant, T, acc) ===")
        for variant in df["variant"].unique():
            sub = df[df["variant"] == variant]
            piv = sub.pivot_table(values="auroc", index="T", columns="acc", aggfunc="mean")
            print(f"\n  {variant}:")
            print(piv.to_string(float_format="%.4f"))
