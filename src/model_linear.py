"""
model_prcd_map.py — PRCD-MAP v3 (Empirical Bayes)
Changes vs v2:
  - Grouped temperature tau_g (edges grouped by prior-probability quantiles, 4 groups by default)
  - Empirical-Bayes update of tau: bilevel optimization with Laplace-approx marginal likelihood
  - Three-level optimization: outer ALM(alpha, rho) -> middle EB update of tau -> inner Adam over W
  - Removed the legacy agreement-based tau update and adaptive_lambda
  - Diagonal Hessian approximation (analytic for linear SVAR; cheap)
"""

import math
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ConstantInputWarning


class PRCD_MAP_Model(nn.Module):
    """
    MAP-consistent PRCD with Empirical Bayes temperature calibration.

    Key change: tau is upgraded from a global scalar to a grouped vector tau_g, updated by maximizing
    a Laplace-approx marginal likelihood between ALM outer iterations, decoupled from the inner W loop.

    Parametrization: sigmoid(logit(P) * tau), tau in [0.01, 1.0].
    tau=1 -> fully use the prior; tau->0 -> P_hat->0.5 (ignore prior).
    Gradient d sigmoid / d tau = logit(P) * sigmoid'(...), does not vanish when P~0.5.

    Grouping: edges are split into n_tau_groups groups by quantiles of off-diagonal P_prior;
    edges within a group share a temperature, so low- and high-prior regions calibrate independently.
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
    ):
        super().__init__()
        self.d = int(num_vars)
        self.K = int(lag_k)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.eps_prior = float(eps_prior)
        self.delta = float(delta)
        self.apply_prior_to_w0 = bool(apply_prior_to_w0)

        # Loss function type
        self.loss_type = str(loss_type)
        if self.loss_type not in ("mse", "huber", "laplace"):
            raise ValueError(f"loss_type must be 'mse'/'huber'/'laplace', got '{loss_type}'")
        self.huber_delta = float(huber_delta)
        self.prior_l1_weight = bool(prior_l1_weight)
        self.tau_prior_sigma = float(tau_prior_sigma)

        # Acyclicity constraint type
        if acyclicity not in ("dagma", "notears"):
            raise ValueError(f"acyclicity must be 'dagma' or 'notears', got '{acyclicity}'")
        self.acyclicity = acyclicity
        if dagma_s is None:
            self.dagma_s = max(1.0, math.log(self.d)) * 1.0
        else:
            self.dagma_s = float(dagma_s)

        # Model parameters
        init_scale = 1e-2
        self.W0 = nn.Parameter(init_scale * torch.randn(self.d, self.d))
        self.Wk = nn.ParameterList(
            [nn.Parameter(init_scale * torch.randn(self.d, self.d)) for _ in range(self.K)]
        )

        # Prior matrix
        P_prior_tensor = torch.tensor(P_prior, dtype=torch.float32)
        if P_prior_tensor.shape != (self.d, self.d):
            raise ValueError(f"P_prior must have shape ({self.d},{self.d}), got {tuple(P_prior_tensor.shape)}")
        self.register_buffer("P_prior", P_prior_tensor)
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.d))

        # Precompute logit(P_prior) to avoid recomputation
        P_clamped = torch.clamp(P_prior_tensor, self.eps_prior, 1.0 - self.eps_prior)
        self.register_buffer("_prior_logits", torch.log(P_clamped) - torch.log1p(-P_clamped))

        # tau configuration
        self.learn_tau = bool(learn_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        if not (self.tau_min < self.tau_max):
            raise ValueError("tau_min must be < tau_max")

        # ---- grouped tau ----
        self.n_tau_groups = max(1, int(n_tau_groups))
        group_indices, actual_n_groups = self._build_tau_groups(P_prior_tensor)
        self.n_tau_groups = actual_n_groups
        self.register_buffer("group_indices", group_indices)

        # Init: when learn_tau is on, start from tau_min (conservative: ignore prior until EB raises it)
        # Multiplicative form: tau->0 ignores prior, tau=1 fully uses prior
        if self.learn_tau:
            init_tau = float(np.clip(self.tau_min, self.tau_min, self.tau_max))
        else:
            init_tau = float(np.clip(tau0, self.tau_min, self.tau_max))
        self.register_buffer(
            "tau_groups",
            torch.full((self.n_tau_groups,), init_tau, dtype=torch.float32),
        )

    def _build_tau_groups(self, P_prior_tensor: torch.Tensor):
        """
        Group edges by quantiles of off-diagonal P_prior.
        Returns (group_indices: LongTensor[d,d], actual_n_groups: int).
        """
        mask = self.off_diag_mask.bool()
        p_offdiag = P_prior_tensor[mask]

        if self.n_tau_groups <= 1:
            return torch.zeros(self.d, self.d, dtype=torch.long), 1

        # Quantile boundaries (deduped to handle many tied values)
        quantile_pts = torch.linspace(0.0, 1.0, self.n_tau_groups + 1)[1:-1]
        boundaries = torch.quantile(p_offdiag.float(), quantile_pts)
        boundaries = torch.unique(boundaries)
        actual_n_groups = len(boundaries) + 1

        group_indices = torch.bucketize(P_prior_tensor, boundaries)
        return group_indices.long(), actual_n_groups

    # --------------------------------------------------------
    # tau-related
    # --------------------------------------------------------
    def _expand_tau(self) -> torch.Tensor:
        """Expand grouped tau to a (d, d) matrix."""
        return self.tau_groups[self.group_indices]

    def _expand_tau_from(self, tau_groups_var: torch.Tensor) -> torch.Tensor:
        """Expand from a given tau vector (autograd-compatible)."""
        return tau_groups_var[self.group_indices]

    def get_tau(self) -> torch.Tensor:
        """Return mean tau (for logging / compatibility)."""
        return self.tau_groups.mean()

    def set_tau(self, value: float):
        """Set all tau groups to the same value."""
        value = float(np.clip(value, self.tau_min, self.tau_max))
        self.tau_groups.fill_(value)

    def get_W0_adj(self) -> torch.Tensor:
        return self.W0 * self.off_diag_mask

    def calibrated_prior(self, tau_matrix: torch.Tensor) -> torch.Tensor:
        """
        tau_matrix: (d, d).
        Multiplicative form: sigmoid(logit(P) * tau).
        tau=1 -> fully use the prior; tau->0 -> P_hat->0.5 (ignore prior).
        Gradient d/d tau = logit(P), does not vanish when P~0.5.
        """
        return torch.sigmoid(self._prior_logits * tau_matrix)

    def omega_mask(self, tau_matrix: torch.Tensor) -> torch.Tensor:
        P_hat = self.calibrated_prior(tau_matrix)
        return (1.0 - P_hat) + self.delta

    # --------------------------------------------------------
    # Acyclicity constraint
    # --------------------------------------------------------
    def _compute_h_notears(self) -> torch.Tensor:
        """NOTEARS: h(W) = tr(exp(W⊙W)) - d"""
        W0_adj = self.get_W0_adj()
        W0_adj = torch.clamp(W0_adj, -3.0, 3.0)
        M = torch.matrix_exp(W0_adj * W0_adj)
        return torch.trace(M) - self.d

    def _compute_h_dagma(self) -> torch.Tensor:
        """DAGMA: h(W) = -log det(sI - W⊙W) + d·log(s)"""
        W0_adj = self.get_W0_adj()
        s = self.dagma_s
        W2 = W0_adj * W0_adj
        M = s * torch.eye(self.d, device=W0_adj.device) - W2
        sign, logabsdet = torch.linalg.slogdet(M)
        if sign.item() <= 0:
            excess = torch.clamp(W2.sum(dim=1) - s, min=0.0)
            return self.d * 1.0 + excess.sum()
        return -logabsdet + self.d * math.log(s)

    def _compute_h_w0(self) -> torch.Tensor:
        if self.acyclicity == "dagma":
            return self._compute_h_dagma()
        else:
            return self._compute_h_notears()

    # --------------------------------------------------------
    # Loss computation
    # --------------------------------------------------------
    def _compute_robust_loss(self, residuals: torch.Tensor,
                             obs_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Robust regression loss, returns a scalar.
        obs_mask: (T, d) binary mask, 1=observed, 0=missing. None=all observed.
        """
        T, d = residuals.shape
        if obs_mask is not None:
            n_obs = obs_mask.sum().clamp(min=1.0)
        else:
            n_obs = T * d

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

    def _compute_prior_adjusted_l1(self, tau_matrix: torch.Tensor) -> torch.Tensor:
        """Prior-modulated L1: high-prior edges get less penalty, low-prior edges more."""
        W0_adj = self.get_W0_adj()
        if not self.prior_l1_weight:
            return self.lambda1 * torch.norm(W0_adj, p=1)
        P_hat = self.calibrated_prior(tau_matrix)
        coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
        return self.lambda1 * torch.sum(coeff * torch.abs(W0_adj))

    def forward(self, X_t: torch.Tensor, X_lags) -> torch.Tensor:
        W0_adj = self.get_W0_adj()
        pred = X_t @ W0_adj
        for k in range(self.K):
            pred = pred + X_lags[k] @ self.Wk[k]
        return pred

    def compute_losses(self, X_t: torch.Tensor, X_lags, rho: float, alpha: float,
                       obs_mask: torch.Tensor = None):
        """Inner-loop loss: tau is fixed (read from buffer, no grad).
        obs_mask: (T, d) binary mask for missing data support. None=all observed.
        """
        tau_matrix = self._expand_tau()
        Omega = self.omega_mask(tau_matrix)

        pred = self.forward(X_t, X_lags)
        residuals = X_t - pred
        loss_mse = self._compute_robust_loss(residuals, obs_mask=obs_mask)
        loss_l1 = self._compute_prior_adjusted_l1(tau_matrix)

        W0_adj = self.get_W0_adj()

        # Prior-weighted L2 (prior-weighted ridge)
        loss_prior = 0.0
        for k in range(self.K):
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (self.Wk[k] ** 2))
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega_w0 * (W0_adj ** 2))

        # Acyclicity constraint + ALM
        h_val = self._compute_h_w0()
        loss_alm = loss_mse + loss_l1 + loss_prior + (alpha * h_val) + 0.5 * rho * (h_val ** 2)

        if not torch.isfinite(loss_alm):
            raise RuntimeError("loss_alm is NaN/Inf. Try smaller lr / stronger clamp / smaller init_scale.")

        tau_mean = self.tau_groups.mean()
        return loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau_mean

    # --------------------------------------------------------
    # Prior quality assessment (quick pre-training check)
    # --------------------------------------------------------
    def calibrate_tau_from_data(self, X_t: torch.Tensor, X_lags):
        """
        Before training, run a fast Lasso VAR edge estimate, compare with P_prior, and set tau.

        Rationale: OLS/Lasso fits without a DAG constraint and is millisecond-cheap.
        Absolute coefficients reflect pairwise association strength. Compare them with P_prior via Spearman:
        positive correlation -> trust prior (tau=1.0), otherwise -> unreliable prior (tau_max).
        """
        from scipy.stats import spearmanr

        with torch.no_grad():
            X_t_np = X_t.detach().cpu().numpy()
            d = self.d

            # Simple pairwise correlation as an edge-strength proxy (fast, ok for d<=100)
            C = np.abs(np.corrcoef(X_t_np.T))
            np.fill_diagonal(C, 0.0)

            # Spearman: edge_strength vs P_prior
            P = self.P_prior.cpu().numpy()
            mask = self.off_diag_mask.bool().cpu().numpy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConstantInputWarning)
                corr, _ = spearmanr(C[mask], P[mask])
            if np.isnan(corr):
                corr = 0.0

            # Smooth mapping: corr -> tau (multiplicative form)
            # High corr → high τ (trust prior), low/negative corr → low τ (ignore prior)
            # Piecewise linear:
            #   corr ≤ -0.05  → τ = tau_min  (bad prior, ignore)
            #   corr ≥  0.20  → τ = tau_max  (good prior, full trust)
            #   in between    → linear interpolation
            # Note: threshold lowered from 0.40 to 0.20 so a medium-quality prior can still get a high tau
            lo, hi = -0.05, 0.20
            if corr <= lo:
                target = self.tau_min
            elif corr >= hi:
                target = self.tau_max
            else:
                frac = (corr - lo) / (hi - lo)
                target = self.tau_min * (1.0 - frac) + self.tau_max * frac

            target = float(np.clip(target, self.tau_min, self.tau_max))
            self.tau_groups.fill_(target)

    # --------------------------------------------------------
    # Empirical Bayes: Laplace-approx marginal likelihood (kept as fallback)
    # --------------------------------------------------------
    def compute_eb_objective(
        self, X_t: torch.Tensor, X_lags, tau_groups_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the empirical-Bayes objective (Laplace approx of negative log marginal likelihood).

        L_EB(τ) = L_prior(W*, τ) + L_l1(W*, τ) + (1/2) Σ log H_ii(τ)

        Here W* is fixed (detached); tau_groups_var carries the gradient.
        The data-fit term L_data(W*) does not depend on tau and is omitted.

        Diagonal Hessian approximation (analytic for linear SVAR):
          H_{W0[i,j]} = ‖X_t[:,i]‖²/(T·d) + λ₂·Ω_{ij}(τ)
          H_{Wk[i,j]} = ‖X_lag_k[:,i]‖²/(T·d) + λ₂·Ω_{ij}(τ)

        Parameters
        ----------
        tau_groups_var : (n_tau_groups,) tensor, requires_grad=True
        """
        dev = X_t.device
        T, d = X_t.shape

        # tau matrix (grad flows through tau_groups_var -> group_indices indexing)
        tau_matrix = self._expand_tau_from(tau_groups_var)
        Omega = self.omega_mask(tau_matrix)

        W0_adj = self.get_W0_adj().detach()

        # --- prior L2 term ---
        loss_prior = torch.tensor(0.0, device=dev)
        for k in range(self.K):
            Wk_det = self.Wk[k].detach()
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (Wk_det ** 2))
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega_w0 * (W0_adj ** 2))

        # --- prior-modulated L1 term ---
        if self.prior_l1_weight:
            P_hat = self.calibrated_prior(tau_matrix)
            coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
            loss_l1 = self.lambda1 * torch.sum(coeff * torch.abs(W0_adj))
        else:
            loss_l1 = self.lambda1 * torch.sum(torch.abs(W0_adj))

        # --- Laplace approx: (1/2) sum log H_ii ---
        X_t_det = X_t.detach()
        # Data-term diagonal Hessian (MSE approx; reasonable for Huber/Laplace too)
        data_hess_row = (X_t_det ** 2).sum(0) / (T * d)  # (d,) -- contribution of row i

        log_det_term = torch.tensor(0.0, device=dev)

        # W0 contribution (off-diagonal only)
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            H_w0 = data_hess_row.unsqueeze(1).expand(d, d) + self.lambda2 * Omega_w0
            log_det_term = log_det_term + torch.sum(
                torch.log(H_w0.clamp(min=1e-10)) * self.off_diag_mask
            )

        # Wk contribution
        for k_idx in range(self.K):
            X_lag_det = X_lags[k_idx].detach()
            data_hess_k = (X_lag_det ** 2).sum(0) / (T * d)
            H_wk = data_hess_k.unsqueeze(1).expand(d, d) + self.lambda2 * Omega
            log_det_term = log_det_term + torch.sum(torch.log(H_wk.clamp(min=1e-10)))

        # --- Agreement term: cross-entropy between prior and |W*| ---
        # If the prior is good, P_hat matches |W*|, cross-entropy is low -> push tau up
        # If the prior is bad, cross-entropy is high -> push tau down (P_hat->0.5, CE -> log2)
        # More direct than loss_prior+loss_l1: it explicitly measures the prior's predictive ability
        P_hat = self.calibrated_prior(tau_matrix)
        W0_abs = torch.abs(W0_adj)
        # soft targets: map |W*| to a [0,1] probability
        w_max = W0_abs.max().clamp(min=1e-6)
        W0_prob = (W0_abs / w_max).clamp(1e-6, 1.0 - 1e-6) * self.off_diag_mask
        P_hat_safe = P_hat.clamp(1e-6, 1.0 - 1e-6)
        # binary cross-entropy (soft)
        agreement_loss = -torch.sum(
            (W0_prob * torch.log(P_hat_safe) + (1.0 - W0_prob) * torch.log(1.0 - P_hat_safe))
            * self.off_diag_mask
        )

        # --- tau regularizer: weak quadratic penalty against extreme values ---
        tau_reg = torch.tensor(0.0, device=dev)
        if self.tau_prior_sigma > 0:
            tau_reg = 0.5 * torch.sum((tau_groups_var - 0.5) ** 2) / (self.tau_prior_sigma ** 2)

        return agreement_loss + 0.5 * log_det_term + tau_reg


# ============================================================
# OLS Warm-Start
# ============================================================
def _ols_warm_start(model: PRCD_MAP_Model, X_t: torch.Tensor, X_lags, verbose=False):
    """
    Use ridge regression to initialize W0, Wk; avoids random-init local minima.
    Standard practice: NOTEARS (Zheng et al. 2018) uses a similar warm start.
    """
    d = model.d
    K = model.K
    T = X_t.shape[0]

    with torch.no_grad():
        # Build regressor matrix: Y = X_t, regressors = [X_t, X_lag1, ..., X_lagK]
        # For each target j: x_t[:,j] = X_t @ w0[:,j] + sum_k X_lag_k @ wk[:,j]
        # For simplicity, use a global ridge: W = (R^T R + lambda I)^-1 R^T Y
        parts = [X_t]
        for lag in X_lags:
            parts.append(lag)
        R = torch.cat(parts, dim=1)  # (T, d*(K+1))
        ridge_lam = 0.1  # strong regularizer to avoid overfitting
        RtR = R.T @ R + ridge_lam * torch.eye(R.shape[1], device=R.device)
        RtY = R.T @ X_t
        try:
            W_all = torch.linalg.solve(RtR, RtY)  # (d*(K+1), d)
        except torch.linalg.LinAlgError:
            if verbose:
                print(">>> OLS warm-start: solve failed, using random init")
            return

        # Extract W0 (first d rows) and Wk (subsequent d rows)
        W0_init = W_all[:d, :]  # (d, d)
        # Zero the diagonal (autoregressive terms are not in W0)
        W0_init.fill_diagonal_(0.0)
        # Scale down to prevent the DAG constraint from blowing up at init
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
            print(f">>> OLS warm-start: W0 max={model.W0.data.abs().max():.4f}, "
                  f"Wk max={max(wk.data.abs().max() for wk in model.Wk):.4f}")


# ============================================================
# Training functions
# ============================================================
def train_prcd_alm(
    model: PRCD_MAP_Model,
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
    tau_eb_lr: float = 0.2,
    warm_start: bool = True,
    lambda_schedule: bool = True,
    inner_early_stop: bool = True,
    inner_es_patience: int = 50,
    inner_es_tol: float = 1e-6,
    obs_mask: torch.Tensor = None,
):
    """
    ALM training:

    Pre-train: OLS warm-start + fast ridge VAR to assess prior quality and set tau
    Outer (Augmented Lagrangian): update alpha, rho to enforce acyclicity
      Inner (Adam): with tau fixed, optimize W0, W1:K (with early stopping)
    Supports lambda scheduling: first 1/3 uses a larger lambda for sparsity, last 2/3 fine-tunes
    """
    device = next(model.parameters()).device
    X_t = X_t.to(device)
    X_lags = [x.to(device) for x in X_lags]
    if obs_mask is not None:
        obs_mask = obs_mask.to(device)

    # === OLS/Ridge Warm-Start ===
    if warm_start:
        _ols_warm_start(model, X_t, X_lags, verbose=verbose)

    def _run_alm_phase(n_iters, rho_start, alpha_start, phase_label=""):
        """Run the ALM outer loop and return (rho, alpha, h_now)."""
        rho_local = float(rho_start)
        alpha_local = float(alpha_start)
        h_now = float("inf")

        for it in range(n_iters):
            optimizer = optim.Adam(model.parameters(), lr=lr)
            if use_lr_schedule:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=inner_iter, eta_min=lr * 0.01)
            else:
                scheduler = None

            for _ in range(inner_iter):
                loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau = \
                    model.compute_losses(X_t=X_t, X_lags=X_lags,
                                         rho=rho_local, alpha=alpha_local,
                                         obs_mask=obs_mask)
                optimizer.zero_grad(set_to_none=True)
                loss_alm.backward()
                if grad_clip is not None and grad_clip > 0:
                    ec = float(grad_clip) * max(1.0, math.log1p(rho_local))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ec)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                h_now = float(model._compute_h_w0().detach().cpu().item())

            if verbose:
                tau_vals = model.tau_groups.tolist()
                tau_str = ", ".join(f"{t:.3f}" for t in tau_vals)
                print(
                    f"[{phase_label} {it+1:02d}/{n_iters}] "
                    f"rho={rho_local:.3g}, alpha={alpha_local:.3g}, "
                    f"h={h_now:.3e}, tau=[{tau_str}], "
                    f"loss_alm={float(loss_alm.detach().cpu().item()):.6f}"
                )

            if abs(h_now) <= tol:
                break

            alpha_local = alpha_local + rho_local * h_now
            rho_local = min(rho_local * float(gamma), float(rho_max))

        return rho_local, alpha_local, h_now

    # === pre-training tau calibration: fast ridge VAR + correlation with P_prior ===
    if model.learn_tau:
        model.calibrate_tau_from_data(X_t, X_lags)
        if verbose:
            tau_str = ", ".join(f"{t:.3f}" for t in model.tau_groups.tolist())
            print(f">>> τ pre-calibrated: [{tau_str}]")

    # === Lambda scheduling: large lambda for the first 1/3, then reduced ===
    lambda1_orig = model.lambda1
    # Warmup factor defaults to 5.0; can be overridden via env var for
    # lambda-sensitivity experiments (exp16) without touching this file.
    _lam1_factor = float(os.environ.get("PRCD_LAM1_WARMUP_FACTOR", "5.0"))
    lambda1_warmup = lambda1_orig * _lam1_factor
    phase1_end = max_iter // 3

    # === Main loop: three-level optimization (outer ALM -> inner Adam over W -> middle EB update of tau) ===
    rho = float(rho_0)
    alpha = 0.0

    for it in range(max_iter):
        # Lambda scheduling
        if lambda_schedule:
            if it < phase1_end:
                model.lambda1 = lambda1_warmup
            else:
                model.lambda1 = lambda1_orig

        optimizer = optim.Adam(model.parameters(), lr=lr)
        if use_lr_schedule:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=inner_iter, eta_min=lr * 0.01)
        else:
            scheduler = None

        prev_loss = float("inf")
        stale_count = 0
        for step in range(inner_iter):
            loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau = \
                model.compute_losses(X_t=X_t, X_lags=X_lags, rho=rho, alpha=alpha,
                                     obs_mask=obs_mask)
            optimizer.zero_grad(set_to_none=True)
            loss_alm.backward()
            if grad_clip is not None and grad_clip > 0:
                ec = float(grad_clip) * max(1.0, math.log1p(rho))
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ec)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Inner early stopping
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

        # --- Middle level: Empirical Bayes update of τ ---
        if model.learn_tau and it >= tau_warmup and tau_eb_steps > 0:
            for _ in range(tau_eb_steps):
                tau_var = model.tau_groups.clone().detach().requires_grad_(True)
                eb_loss = model.compute_eb_objective(X_t, X_lags, tau_var)
                eb_loss.backward()
                with torch.no_grad():
                    tau_var = tau_var - tau_eb_lr * tau_var.grad
                    tau_var = tau_var.clamp(model.tau_min, model.tau_max)
                model.tau_groups.copy_(tau_var)

        if verbose:
            tau_vals = model.tau_groups.tolist()
            tau_str = ", ".join(f"{t:.3f}" for t in tau_vals)
            print(
                f"[outer {it+1:02d}/{max_iter}] "
                f"rho={rho:.3g}, alpha={alpha:.3g}, h={h_now:.3e}, "
                f"tau=[{tau_str}], "
                f"loss_alm={float(loss_alm.detach().cpu().item()):.6f}"
            )

        if abs(h_now) <= tol:
            break

        alpha = alpha + rho * h_now
        rho = min(rho * float(gamma), float(rho_max))

    # Restore original lambda1
    model.lambda1 = lambda1_orig

    # Extract results
    with torch.no_grad():
        W0_raw = model.get_W0_adj().detach().cpu().numpy()
        Wk_raw = [wk.detach().cpu().numpy() for wk in model.Wk]
        tau_est = float(model.get_tau().detach().cpu().item())

        if not postprocess:
            return W0_raw, Wk_raw, tau_est

        max_weight = float(np.max(np.abs(W0_raw))) if W0_raw.size else 0.0
        thr = float(thr_ratio) * max_weight
        if verbose:
            tau_str = ", ".join(f"{t:.3f}" for t in model.tau_groups.tolist())
            print(f">>> Post-processing: max|W0|={max_weight:.6f}, threshold={thr:.6f}, tau=[{tau_str}]")
        W0_est = W0_raw.copy()
        W0_est[np.abs(W0_est) < thr] = 0.0
        Wk_est = []
        for wk in Wk_raw:
            wk2 = wk.copy()
            wk2[np.abs(wk2) < thr] = 0.0
            Wk_est.append(wk2)

    return W0_est, Wk_est, tau_est
