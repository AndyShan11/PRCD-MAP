"""
model_prcd_map_trust.py — PRCD-MAP with Structure-Aware Trust Propagation.

Key change: tau is upgraded from a per-group scalar vector to a structure-aware
per-edge matrix. A lightweight GAT propagates trust on the prior graph, using
neighborhood-consistency signals.

Differences vs the original PRCD_MAP_Model:
  - tau_groups (buffer) -> TrustPropagationModule (nn.Module)
  - _expand_tau() -> trust_module.forward(P_prior, W_strength, ...)
  - EB update: gradient descent on trust_module parameters (replaces direct SGD on tau_groups)
  - The rest of the framework (SVAR, prior modulation, ALM, DAG constraint) is unchanged
"""

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ConstantInputWarning

from trust_propagation import TrustPropagationModule, TrustPropagationLite


class PRCD_MAP_Trust(nn.Module):
    """
    PRCD-MAP with Structure-Aware Trust Propagation.

    Parametrization: P_hat = sigmoid(logit(P) * tau(i,j))
    where tau(i,j) = TrustPropagation(P_prior, |W|, graph_structure)

    Definition (Structure-Aware Trust Propagation):
      Given prior graph P ∈ [0,1]^{d×d} and current weight estimate W ∈ R^{d×d},
      the structure-aware trust temperature τ: E → [τ_min, τ_max] is defined as:
        τ_{ij} = f_θ( h_{ij}^{(L)} )
      where h_{ij}^{(l)} = GAT_l({h_{kl}^{(l-1)} : (k,l) ∈ N(i,j)})
      and h_{ij}^{(0)} = [P_{ij}, ||W_{ij}||, g(group(i,j))]
      f_θ maps to [τ_min, τ_max] via sigmoid scaling.
    """

    def __init__(
        self,
        num_vars: int,
        lag_k: int,
        P_prior,
        lambda1: float = 0.01,
        lambda2: float = 0.01,
        eps_prior: float = 1e-3,
        delta: float = 1e-3,
        learn_tau: bool = True,
        tau0: float = 1.0,
        tau_min: float = 0.05,
        tau_max: float = 3.0,
        apply_prior_to_w0: bool = True,
        acyclicity: str = "dagma",
        dagma_s: float = None,
        loss_type: str = "huber",
        huber_delta: float = 1.0,
        prior_l1_weight: bool = True,
        n_tau_groups: int = 4,
        tau_prior_sigma: float = 2.0,
        # Trust propagation params
        trust_feat_dim: int = 16,
        trust_n_layers: int = 2,
        trust_n_heads: int = 2,
        trust_dropout: float = 0.0,
        trust_lite: bool = False,
    ):
        super().__init__()
        self.d = int(num_vars)
        self.K = int(lag_k)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.eps_prior = float(eps_prior)
        self.delta = float(delta)
        self.apply_prior_to_w0 = bool(apply_prior_to_w0)
        self.loss_type = str(loss_type)
        self.huber_delta = float(huber_delta)
        self.prior_l1_weight = bool(prior_l1_weight)
        self.tau_prior_sigma = float(tau_prior_sigma)
        self.learn_tau = bool(learn_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        if acyclicity not in ("dagma", "notears"):
            raise ValueError(f"acyclicity must be 'dagma' or 'notears', got '{acyclicity}'")
        self.acyclicity = acyclicity
        if dagma_s is None:
            self.dagma_s = max(1.0, math.log(self.d)) * 1.0
        else:
            self.dagma_s = float(dagma_s)

        # SVAR parameters
        init_scale = 1e-2
        self.W0 = nn.Parameter(init_scale * torch.randn(self.d, self.d))
        self.Wk = nn.ParameterList(
            [nn.Parameter(init_scale * torch.randn(self.d, self.d)) for _ in range(self.K)]
        )

        # Prior
        P_prior_tensor = torch.tensor(P_prior, dtype=torch.float32)
        self.register_buffer("P_prior", P_prior_tensor)
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.d))

        P_clamped = torch.clamp(P_prior_tensor, self.eps_prior, 1.0 - self.eps_prior)
        self.register_buffer("_prior_logits", torch.log(P_clamped) - torch.log1p(-P_clamped))

        # Tau group indices (for feature extraction in trust module)
        self.n_tau_groups = max(1, int(n_tau_groups))
        group_indices, actual_n_groups = self._build_tau_groups(P_prior_tensor)
        self.n_tau_groups = actual_n_groups
        self.register_buffer("group_indices", group_indices)

        # Trust Propagation Module
        # GAT only used when d<=10 and lite is not requested; d>10 always uses Lite (MLP + neighborhood stats)
        # Reason: edge-level GAT is O(d^3); at d=20 each step is ~3h, which is unacceptable
        use_lite = trust_lite or self.d > 10
        if use_lite:
            self.trust_module = TrustPropagationLite(
                tau_min=tau_min, tau_max=tau_max, hidden=trust_feat_dim
            )
        else:
            self.trust_module = TrustPropagationModule(
                feat_dim=trust_feat_dim, n_layers=trust_n_layers,
                n_heads=trust_n_heads, dropout=trust_dropout,
                tau_min=tau_min, tau_max=tau_max
            )

        # Fallback: per-group tau for when trust module is disabled
        if self.learn_tau:
            init_tau = float(np.clip(tau_min, tau_min, tau_max))
        else:
            init_tau = float(np.clip(tau0, tau_min, tau_max))
        self.register_buffer(
            "tau_groups",
            torch.full((self.n_tau_groups,), init_tau, dtype=torch.float32),
        )

    def _build_tau_groups(self, P_prior_tensor):
        mask = self.off_diag_mask.bool()
        p_offdiag = P_prior_tensor[mask]
        if self.n_tau_groups <= 1:
            return torch.zeros(self.d, self.d, dtype=torch.long), 1
        quantile_pts = torch.linspace(0.0, 1.0, self.n_tau_groups + 1)[1:-1]
        boundaries = torch.quantile(p_offdiag.float(), quantile_pts)
        boundaries = torch.unique(boundaries)
        actual_n_groups = len(boundaries) + 1
        group_indices = torch.bucketize(P_prior_tensor, boundaries)
        return group_indices.long(), actual_n_groups

    # ---- τ computation ----

    def _get_W_strength(self):
        """Current |W0| (detached, for use as trust-module input)."""
        return (self.W0.detach() * self.off_diag_mask).abs()

    def _compute_tau_matrix(self):
        """Compute per-edge tau via trust propagation."""
        if self.learn_tau:
            W_str = self._get_W_strength()
            return self.trust_module(self.P_prior, W_str, self.group_indices, self.off_diag_mask)
        else:
            return self.tau_groups[self.group_indices]

    def _expand_tau(self):
        """Compatibility shim: returns the (d, d) tau matrix."""
        return self._compute_tau_matrix()

    def get_tau(self):
        """Return mean tau."""
        if self.learn_tau:
            return self.trust_module.get_tau_mean(
                self.P_prior, self._get_W_strength(),
                self.group_indices, self.off_diag_mask
            )
        return self.tau_groups.mean()

    def set_tau(self, value: float):
        value = float(np.clip(value, self.tau_min, self.tau_max))
        self.tau_groups.fill_(value)

    def get_W0_adj(self):
        return self.W0 * self.off_diag_mask

    def calibrated_prior(self, tau_matrix):
        return torch.sigmoid(self._prior_logits * tau_matrix)

    def omega_mask(self, tau_matrix):
        P_hat = self.calibrated_prior(tau_matrix)
        return (1.0 - P_hat) + self.delta

    # ---- DAG constraint ----

    def _compute_h_dagma(self):
        W0_adj = self.get_W0_adj()
        s = self.dagma_s
        W2 = W0_adj * W0_adj
        M = s * torch.eye(self.d, device=W0_adj.device) - W2
        sign, logabsdet = torch.linalg.slogdet(M)
        if sign.item() <= 0:
            excess = torch.clamp(W2.sum(dim=1) - s, min=0.0)
            return self.d * 1.0 + excess.sum()
        return -logabsdet + self.d * math.log(s)

    def _compute_h_notears(self):
        W0_adj = self.get_W0_adj()
        W0_adj = torch.clamp(W0_adj, -3.0, 3.0)
        M = torch.matrix_exp(W0_adj * W0_adj)
        return torch.trace(M) - self.d

    def _compute_h_w0(self):
        if self.acyclicity == "dagma":
            return self._compute_h_dagma()
        return self._compute_h_notears()

    # ---- Loss computation ----

    def _compute_robust_loss(self, residuals, obs_mask=None):
        T, d = residuals.shape
        n_obs = obs_mask.sum().clamp(min=1.0) if obs_mask is not None else T * d
        if self.loss_type == "mse":
            loss = 0.5 * (residuals ** 2)
        elif self.loss_type == "huber":
            delta = self.huber_delta
            abs_r = torch.abs(residuals)
            quadratic = torch.clamp(abs_r, max=delta)
            linear = abs_r - quadratic
            loss = 0.5 * quadratic ** 2 + delta * linear
        elif self.loss_type == "laplace":
            loss = torch.abs(residuals)
        else:
            raise RuntimeError(f"Unknown loss_type: {self.loss_type}")
        if obs_mask is not None:
            loss = loss * obs_mask
        return torch.sum(loss) / n_obs

    def _compute_prior_adjusted_l1(self, tau_matrix):
        W0_adj = self.get_W0_adj()
        if not self.prior_l1_weight:
            return self.lambda1 * torch.norm(W0_adj, p=1)
        P_hat = self.calibrated_prior(tau_matrix)
        coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
        return self.lambda1 * torch.sum(coeff * torch.abs(W0_adj))

    def forward(self, X_t, X_lags):
        W0_adj = self.get_W0_adj()
        pred = X_t @ W0_adj
        for k in range(self.K):
            pred = pred + X_lags[k] @ self.Wk[k]
        return pred

    def compute_losses(self, X_t, X_lags, rho, alpha, obs_mask=None):
        # During inner optimization, detach tau: trust-module gradients are only computed in the EB phase
        with torch.no_grad():
            tau_matrix = self._compute_tau_matrix()
        tau_matrix = tau_matrix.detach()
        Omega = self.omega_mask(tau_matrix)

        pred = self.forward(X_t, X_lags)
        residuals = X_t - pred
        loss_mse = self._compute_robust_loss(residuals, obs_mask=obs_mask)
        loss_l1 = self._compute_prior_adjusted_l1(tau_matrix)

        W0_adj = self.get_W0_adj()
        loss_prior = 0.0
        for k in range(self.K):
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (self.Wk[k] ** 2))
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega_w0 * (W0_adj ** 2))

        h_val = self._compute_h_w0()
        loss_alm = loss_mse + loss_l1 + loss_prior + (alpha * h_val) + 0.5 * rho * (h_val ** 2)

        if not torch.isfinite(loss_alm):
            raise RuntimeError("loss_alm is NaN/Inf.")

        tau_mean = tau_matrix[self.off_diag_mask.bool()].mean().detach()
        return loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau_mean

    # ---- EB objective for trust module ----

    def compute_eb_objective(self, X_t, X_lags):
        """
        EB objective: gradients for the trust module flow through this function.
        W0, Wk are fixed (detached); only the trust-module parameters carry gradients.
        """
        dev = X_t.device
        T, d = X_t.shape

        tau_matrix = self._compute_tau_matrix()
        Omega = self.omega_mask(tau_matrix)

        W0_adj = self.get_W0_adj().detach()

        # Prior L2
        loss_prior = torch.tensor(0.0, device=dev)
        for k in range(self.K):
            Wk_det = self.Wk[k].detach()
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (Wk_det ** 2))
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega_w0 * (W0_adj ** 2))

        # Prior L1
        if self.prior_l1_weight:
            P_hat = self.calibrated_prior(tau_matrix)
            coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
            loss_l1 = self.lambda1 * torch.sum(coeff * torch.abs(W0_adj))
        else:
            loss_l1 = self.lambda1 * torch.sum(torch.abs(W0_adj))

        # Laplace log-det
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

        # Agreement loss
        P_hat = self.calibrated_prior(tau_matrix)
        W0_abs = torch.abs(W0_adj)
        w_max = W0_abs.max().clamp(min=1e-6)
        W0_prob = (W0_abs / w_max).clamp(1e-6, 1.0 - 1e-6) * self.off_diag_mask
        P_hat_safe = P_hat.clamp(1e-6, 1.0 - 1e-6)
        agreement_loss = -torch.sum(
            (W0_prob * torch.log(P_hat_safe) + (1.0 - W0_prob) * torch.log(1.0 - P_hat_safe))
            * self.off_diag_mask
        )

        # Trust module parameter regularization
        trust_reg = torch.tensor(0.0, device=dev)
        for p in self.trust_module.parameters():
            trust_reg = trust_reg + 0.01 * torch.sum(p ** 2)

        return agreement_loss + 0.5 * log_det_term + loss_prior + loss_l1 + trust_reg

    # ---- Pre-training calibration ----

    def calibrate_tau_from_data(self, X_t, X_lags):
        from scipy.stats import spearmanr
        with torch.no_grad():
            X_t_np = X_t.detach().cpu().numpy()
            d = self.d
            C = np.abs(np.corrcoef(X_t_np.T))
            np.fill_diagonal(C, 0.0)
            P = self.P_prior.cpu().numpy()
            mask = self.off_diag_mask.bool().cpu().numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConstantInputWarning)
                corr, _ = spearmanr(C[mask], P[mask])
            if np.isnan(corr):
                corr = 0.0
            # Adjust the trust-module bias based on the correlation
            lo, hi = -0.05, 0.20
            if corr <= lo:
                target_sigmoid = 0.01  # close to tau_min
            elif corr >= hi:
                target_sigmoid = 0.99  # close to tau_max
            else:
                frac = (corr - lo) / (hi - lo)
                target_sigmoid = frac
            # inverse sigmoid: logit
            target_sigmoid = max(0.01, min(0.99, target_sigmoid))
            bias_val = math.log(target_sigmoid / (1.0 - target_sigmoid))
            self.trust_module.tau_bias.data.fill_(bias_val)


# ============================================================
# OLS Warm-Start
# ============================================================
def _ols_warm_start(model: PRCD_MAP_Trust, X_t, X_lags, verbose=False):
    d = model.d
    K = model.K
    with torch.no_grad():
        parts = [X_t]
        for lag in X_lags:
            parts.append(lag)
        R = torch.cat(parts, dim=1)
        ridge_lam = 0.1
        RtR = R.T @ R + ridge_lam * torch.eye(R.shape[1], device=R.device)
        RtY = R.T @ X_t
        try:
            W_all = torch.linalg.solve(RtR, RtY)
        except torch.linalg.LinAlgError:
            if verbose:
                print(">>> OLS warm-start: solve failed, using random init")
            return
        W0_init = W_all[:d, :]
        W0_init.fill_diagonal_(0.0)
        w0_max = W0_init.abs().max().clamp(min=1e-6)
        if w0_max > 0.5:
            W0_init = W0_init * (0.5 / w0_max)
        model.W0.data.copy_(W0_init)
        for k in range(K):
            Wk_init = W_all[d * (k + 1):d * (k + 2), :]
            wk_max = Wk_init.abs().max().clamp(min=1e-6)
            if wk_max > 0.5:
                Wk_init = Wk_init * (0.5 / wk_max)
            model.Wk[k].data.copy_(Wk_init)
        if verbose:
            print(f">>> OLS warm-start: W0 max={model.W0.data.abs().max():.4f}")


# ============================================================
# Training Function
# ============================================================
def train_prcd_trust_alm(
    model: PRCD_MAP_Trust,
    X_t: torch.Tensor,
    X_lags,
    max_iter: int = 35,
    inner_iter: int = 500,
    rho_0: float = 1.0,
    gamma: float = 3.0,
    rho_max: float = 1e6,
    tol: float = 1e-6,
    lr: float = 1e-2,
    verbose: bool = True,
    postprocess: bool = False,
    thr_ratio: float = 0.10,
    grad_clip: float = 5.0,
    use_lr_schedule: bool = True,
    tau_warmup: int = 0,
    tau_eb_steps: int = 8,
    tau_eb_lr: float = 1e-3,
    warm_start: bool = True,
    lambda_schedule: bool = True,
    inner_early_stop: bool = True,
    inner_es_patience: int = 50,
    inner_es_tol: float = 1e-6,
    obs_mask: torch.Tensor = None,
):
    """
    ALM training with Structure-Aware Trust Propagation.

    Differences vs the original train_prcd_alm:
    - Middle EB update: Adam over trust_module parameters (replaces SGD on tau_groups)
    - Inner Adam: W0, Wk and trust_module parameters are optimized together (trust module contributes tau -> loss)
    """
    device = next(model.parameters()).device
    X_t = X_t.to(device)
    X_lags = [x.to(device) for x in X_lags]
    if obs_mask is not None:
        obs_mask = obs_mask.to(device)

    if warm_start:
        _ols_warm_start(model, X_t, X_lags, verbose=verbose)

    if model.learn_tau:
        model.calibrate_tau_from_data(X_t, X_lags)
        if verbose:
            tau_val = float(model.get_tau())
            print(f">>> τ pre-calibrated: mean={tau_val:.3f}")

    lambda1_orig = model.lambda1
    lambda1_warmup = lambda1_orig * 5.0
    phase1_end = max_iter // 3

    rho = float(rho_0)
    alpha = 0.0

    # Separate parameter groups: SVAR params vs trust module params
    svar_params = [model.W0] + list(model.Wk.parameters())
    trust_params = list(model.trust_module.parameters())

    for it in range(max_iter):
        if lambda_schedule:
            model.lambda1 = lambda1_warmup if it < phase1_end else lambda1_orig

        # Inner loop: optimize W0, Wk (and trust module contributes τ)
        # Only SVAR params get gradient from inner loss
        optimizer = optim.Adam(svar_params, lr=lr)
        if use_lr_schedule:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=inner_iter, eta_min=lr * 0.01)
        else:
            scheduler = None

        prev_loss = float("inf")
        stale_count = 0
        for step in range(inner_iter):
            loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau = \
                model.compute_losses(X_t, X_lags, rho, alpha, obs_mask)
            optimizer.zero_grad(set_to_none=True)
            loss_alm.backward()
            if grad_clip > 0:
                ec = float(grad_clip) * max(1.0, math.log1p(rho))
                torch.nn.utils.clip_grad_norm_(svar_params, max_norm=ec)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if inner_early_stop:
                cur_loss = float(loss_alm.detach())
                if abs(prev_loss - cur_loss) < inner_es_tol:
                    stale_count += 1
                    if stale_count >= inner_es_patience:
                        break
                else:
                    stale_count = 0
                prev_loss = cur_loss

        with torch.no_grad():
            h_now = float(model._compute_h_w0().detach().cpu().item())

        # Middle level: EB update of trust module
        if model.learn_tau and it >= tau_warmup and tau_eb_steps > 0:
            trust_optimizer = optim.Adam(trust_params, lr=tau_eb_lr)
            for _ in range(tau_eb_steps):
                eb_loss = model.compute_eb_objective(X_t, X_lags)
                trust_optimizer.zero_grad(set_to_none=True)
                eb_loss.backward()
                torch.nn.utils.clip_grad_norm_(trust_params, max_norm=1.0)
                trust_optimizer.step()

        if verbose:
            tau_val = float(model.get_tau())
            print(
                f"[outer {it+1:02d}/{max_iter}] "
                f"rho={rho:.3g}, alpha={alpha:.3g}, h={h_now:.3e}, "
                f"tau_mean={tau_val:.3f}, "
                f"loss_alm={float(loss_alm.detach().cpu().item()):.6f}"
            )

        if abs(h_now) <= tol:
            break

        alpha = alpha + rho * h_now
        rho = min(rho * float(gamma), float(rho_max))

    model.lambda1 = lambda1_orig

    with torch.no_grad():
        W0_raw = model.get_W0_adj().detach().cpu().numpy()
        Wk_raw = [wk.detach().cpu().numpy() for wk in model.Wk]
        tau_est = float(model.get_tau())

        if not postprocess:
            return W0_raw, Wk_raw, tau_est

        max_weight = float(np.max(np.abs(W0_raw))) if W0_raw.size else 0.0
        thr = float(thr_ratio) * max_weight
        W0_est = W0_raw.copy()
        W0_est[np.abs(W0_est) < thr] = 0.0
        Wk_est = [wk.copy() for wk in Wk_raw]
        for wk in Wk_est:
            wk[np.abs(wk) < thr] = 0.0

    return W0_est, Wk_est, tau_est
