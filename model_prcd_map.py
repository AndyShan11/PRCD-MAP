import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PRCD_MAP_Model(nn.Module):
    """
    MAP-consistent PRCD (SVAR-like + acyclicity on instantaneous W0):
      - Parameters: instantaneous W0 (acyclic constraint), lagged W1..WK
      - Prior matrix P_prior in [0,1]^{dxd}
      - Temperature-calibrated prior P_hat(tau)
      - Omega(tau) = (1 - P_hat(tau)) + delta
      - Gaussian prior => weighted L2:
          (lambda2/2) * sum_{k} sum_{ij} Omega_ij * Wk_{ij}^2
        (optionally also applied to W0 off-diagonal)
      - When learn_tau=True, tau is updated externally in the outer ALM loop
        based on agreement between estimated graph and prior (not by gradient).

    Acyclicity constraint options:
      - "dagma": h(W) = -log det(sI - W⊙W) + d·log(s)  [DAGMA, Kevin Bello et al. 2022]
      - "notears": h(W) = tr(exp(W⊙W)) - d              [NOTEARS, Zheng et al. 2018]
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
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        apply_prior_to_w0: bool = True,
        acyclicity: str = "dagma",
        dagma_s: float = None,
    ):
        super().__init__()
        self.d = int(num_vars)
        self.K = int(lag_k)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.eps_prior = float(eps_prior)
        self.delta = float(delta)
        self.apply_prior_to_w0 = bool(apply_prior_to_w0)

        # Acyclicity constraint type
        if acyclicity not in ("dagma", "notears"):
            raise ValueError(f"acyclicity must be 'dagma' or 'notears', got '{acyclicity}'")
        self.acyclicity = acyclicity
        # DAGMA s must exceed max eigenvalue of W⊙W. Scale with d for safety.
        if dagma_s is None:
            self.dagma_s = max(1.0, math.log(self.d)) * 1.0
        else:
            self.dagma_s = float(dagma_s)

        init_scale = 1e-2
        self.W0 = nn.Parameter(init_scale * torch.randn(self.d, self.d))
        self.Wk = nn.ParameterList(
            [nn.Parameter(init_scale * torch.randn(self.d, self.d)) for _ in range(self.K)]
        )

        P_prior_tensor = torch.tensor(P_prior, dtype=torch.float32)
        if P_prior_tensor.shape != (self.d, self.d):
            raise ValueError(f"P_prior must have shape ({self.d},{self.d}), got {tuple(P_prior_tensor.shape)}")
        self.register_buffer("P_prior", P_prior_tensor)
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.d))

        self.learn_tau = bool(learn_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        if not (self.tau_min < self.tau_max):
            raise ValueError("tau_min must be < tau_max")

        # tau is always a buffer (never a Parameter) — no gradient flows through it.
        # When learn_tau=True, tau starts from tau_max (conservative: don't trust prior initially)
        # and is updated externally via set_tau() in the outer ALM loop.
        if self.learn_tau:
            tau0 = float(self.tau_max)
        else:
            tau0 = float(np.clip(tau0, self.tau_min, self.tau_max))
        self.register_buffer("tau_val", torch.tensor(tau0, dtype=torch.float32))

    def get_tau(self) -> torch.Tensor:
        return self.tau_val

    def set_tau(self, value: float):
        """Externally set tau, clamped to [tau_min, tau_max]."""
        value = float(np.clip(value, self.tau_min, self.tau_max))
        self.tau_val.fill_(value)

    def get_W0_adj(self) -> torch.Tensor:
        return self.W0 * self.off_diag_mask

    def calibrated_prior(self, tau: torch.Tensor) -> torch.Tensor:
        P = torch.clamp(self.P_prior, self.eps_prior, 1.0 - self.eps_prior)
        logits = torch.log(P) - torch.log1p(-P)
        return torch.sigmoid(logits / tau)

    def omega_mask(self, tau: torch.Tensor) -> torch.Tensor:
        P_hat = self.calibrated_prior(tau)
        return (1.0 - P_hat) + self.delta

    def _compute_h_notears(self) -> torch.Tensor:
        """NOTEARS: h(W) = tr(exp(W⊙W)) - d"""
        W0_adj = self.get_W0_adj()
        W0_adj = torch.clamp(W0_adj, -3.0, 3.0)
        M = torch.matrix_exp(W0_adj * W0_adj)
        return torch.trace(M) - self.d

    def _compute_h_dagma(self) -> torch.Tensor:
        """DAGMA: h(W) = -log det(sI - W⊙W) + d·log(s)
        When sign <= 0, M is not positive definite (W has cycles).
        Fallback: return a large but bounded penalty proportional to
        how much W⊙W exceeds s, preserving meaningful gradient signal.
        """
        W0_adj = self.get_W0_adj()
        s = self.dagma_s
        W2 = W0_adj * W0_adj
        M = s * torch.eye(self.d, device=W0_adj.device) - W2
        sign, logabsdet = torch.linalg.slogdet(M)
        if sign.item() <= 0:
            # Bounded fallback: penalize excess eigenvalues over s
            # This gives gradient to reduce large weights without collapsing all to zero
            excess = torch.clamp(W2.sum(dim=1) - s, min=0.0)
            return self.d * 1.0 + excess.sum()
        h = -logabsdet + self.d * math.log(s)
        return h

    def _compute_h_w0(self) -> torch.Tensor:
        if self.acyclicity == "dagma":
            return self._compute_h_dagma()
        else:
            return self._compute_h_notears()

    def forward(self, X_t: torch.Tensor, X_lags) -> torch.Tensor:
        W0_adj = self.get_W0_adj()
        pred = X_t @ W0_adj
        for k in range(self.K):
            pred = pred + X_lags[k] @ self.Wk[k]
        return pred

    def compute_losses(self, X_t: torch.Tensor, X_lags, rho: float, alpha: float):
        T = X_t.shape[0]
        tau = self.get_tau()
        Omega = self.omega_mask(tau)
        pred = self.forward(X_t, X_lags)
        loss_mse = 0.5 * torch.sum((X_t - pred) ** 2) / (T * self.d)
        W0_adj = self.get_W0_adj()
        # L1 only on W0; adaptive reduction when prior is strong (avoid double-sparsification)
        prior_strength = (1.0 - self.calibrated_prior(tau)).mean()
        effective_lambda1 = self.lambda1 * (1.0 - 0.5 * prior_strength)
        loss_l1 = effective_lambda1 * torch.norm(W0_adj, p=1)
        loss_prior = 0.0
        for k in range(self.K):
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (self.Wk[k] ** 2))
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega_w0 * (W0_adj ** 2))
        h_val = self._compute_h_w0()
        loss_alm = loss_mse + loss_l1 + loss_prior + (alpha * h_val) + 0.5 * rho * (h_val ** 2)
        if not torch.isfinite(loss_alm):
            raise RuntimeError("loss_alm is NaN/Inf. Try smaller lr / stronger clamp / smaller init_scale.")
        return loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau

    def compute_agreement(self) -> float:
        """
        Compute agreement between current estimated graph and prior using
        soft F1 on off-diagonal entries.

        Uses sigmoid-smoothed edge indicators from |W_combined| compared
        against P_prior. High soft-F1 => prior is reliable => small tau.

        Returns agreement in [0, 1].  0.0 if W is near-zero (early training).
        """
        with torch.no_grad():
            W0_adj = self.get_W0_adj()
            W_combined = torch.abs(W0_adj)
            for wk in self.Wk:
                W_combined = torch.max(W_combined, torch.abs(wk))

            mask = self.off_diag_mask.bool()
            w_vec = W_combined[mask]
            p_vec = self.P_prior[mask]

            # If W is near-zero (early training), return 0.0 (don't trust prior yet)
            if w_vec.max().item() < 1e-8:
                return 0.0

            # Adaptive sigmoid steepness based on weight distribution
            w_std = w_vec.std()
            k = 10.0 / (w_std + 1e-8)
            w_mean = w_vec.mean()
            W_soft = torch.sigmoid(k * (w_vec - w_mean))

            # Soft F1 computation
            tp = (W_soft * p_vec).sum()
            fp = (W_soft * (1.0 - p_vec)).sum()
            fn = ((1.0 - W_soft) * p_vec).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            return float(f1.clamp(0.0, 1.0).item())


def train_prcd_alm(
    model: PRCD_MAP_Model,
    X_t: torch.Tensor,
    X_lags,
    max_iter: int = 20,
    inner_iter: int = 500,
    rho_0: float = 1.0,
    gamma: float = 2.0,
    rho_max: float = 1e6,
    tol: float = 1e-6,
    lr: float = 1e-2,
    verbose: bool = True,
    postprocess: bool = False,
    thr_ratio: float = 0.10,
    grad_clip: float = 5.0,
    tau_ema: float = 0.1,
    use_lr_schedule: bool = True,
    tau_warmup: int = 1,
):
    device = next(model.parameters()).device
    X_t = X_t.to(device)
    X_lags = [x.to(device) for x in X_lags]

    rho = float(rho_0)
    alpha = 0.0

    for it in range(max_iter):
        # Fresh optimizer each outer iteration to avoid stale momentum from
        # drastically different loss scales as rho grows
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # LR scheduling: CosineAnnealingLR per outer iteration
        if use_lr_schedule:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=inner_iter, eta_min=lr * 0.01
            )
        else:
            scheduler = None

        # --- Inner loop: fix tau, optimize W0 and Wk ---
        for _ in range(inner_iter):
            loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau = model.compute_losses(
                X_t=X_t, X_lags=X_lags, rho=rho, alpha=alpha
            )
            optimizer.zero_grad(set_to_none=True)
            loss_alm.backward()
            if grad_clip is not None and grad_clip > 0:
                # Scale-aware clipping: allow larger gradients when rho is large
                effective_clip = float(grad_clip) * max(1.0, math.log1p(rho))
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=effective_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            h_now = float(model._compute_h_w0().detach().cpu().item())
            tau_now = float(model.get_tau().detach().cpu().item())

        # --- Agreement-based tau update (outer loop, with warmup) ---
        agreement = None
        if model.learn_tau and it >= tau_warmup:
            agreement = model.compute_agreement()
            # Exponential schedule: high agreement => tau near tau_min; low => near tau_max
            tau_target = model.tau_min * (model.tau_max / model.tau_min) ** (1.0 - agreement)
            # Adaptive EMA: high agreement => fast descent (low smoothing)
            effective_ema = tau_ema * (1.0 - agreement)
            tau_new = effective_ema * tau_now + (1.0 - effective_ema) * tau_target
            model.set_tau(tau_new)
            tau_now = tau_new

        if verbose:
            agree_str = ""
            if agreement is not None:
                agree_str = f", agree={agreement:.4f}"
            print(
                f"[outer {it+1:02d}/{max_iter}] "
                f"rho={rho:.3g}, alpha={alpha:.3g}, h={h_now:.3e}, tau={tau_now:.4f}{agree_str}, "
                f"loss_alm={float(loss_alm.detach().cpu().item()):.6f}"
            )

        if abs(h_now) <= tol:
            break

        alpha = alpha + rho * h_now
        rho = min(rho * float(gamma), float(rho_max))

    with torch.no_grad():
        W0_raw = model.get_W0_adj().detach().cpu().numpy()
        Wk_raw = [wk.detach().cpu().numpy() for wk in model.Wk]
        tau_est = float(model.get_tau().detach().cpu().item())

        if not postprocess:
            return W0_raw, Wk_raw, tau_est

        max_weight = float(np.max(np.abs(W0_raw))) if W0_raw.size else 0.0
        thr = float(thr_ratio) * max_weight
        if verbose:
            print(f">>> Post-processing: max|W0|={max_weight:.6f}, threshold={thr:.6f}, tau={tau_est:.4f}")
        W0_est = W0_raw.copy()
        W0_est[np.abs(W0_est) < thr] = 0.0
        Wk_est = []
        for wk in Wk_raw:
            wk2 = wk.copy()
            wk2[np.abs(wk2) < thr] = 0.0
            Wk_est.append(wk2)

    return W0_est, Wk_est, tau_est
