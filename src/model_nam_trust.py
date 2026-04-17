"""
model_prcd_map_nam_trust.py — PRCD-MAP NAM with Structure-Aware Trust Propagation.

线性 SVAR → NAM (Neural Additive Model): 每条边 (i→j) 用 MLP f_{ij}(x_i).
同时集成 structure-aware trust propagation 替代 per-group τ.

适用范围: d ≤ 30.
"""

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ConstantInputWarning

from trust_propagation import TrustPropagationModule, TrustPropagationLite


class EdgeMLP(nn.Module):
    """单边非线性函数 f_{ij}: R → R."""

    def __init__(self, hidden=16, n_layers=2):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def edge_strength(self):
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            total = total + torch.sum(p ** 2)
        return torch.sqrt(total + 1e-12)


class PRCD_MAP_NAM_Trust(nn.Module):
    """PRCD-MAP NAM with Structure-Aware Trust Propagation."""

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
        dagma_s: float = None,
        edge_hidden: int = 16,
        edge_layers: int = 2,
        n_tau_groups: int = 4,
        tau_prior_sigma: float = 2.0,
        trust_feat_dim: int = 16,
        trust_n_layers: int = 2,
        trust_n_heads: int = 2,
    ):
        super().__init__()
        self.d = int(num_vars)
        self.K = int(lag_k)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.eps_prior = float(eps_prior)
        self.delta = float(delta)
        self.tau_prior_sigma = float(tau_prior_sigma)
        self.learn_tau = bool(learn_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        if dagma_s is None:
            self.dagma_s = max(1.0, math.log(self.d)) * 1.0
        else:
            self.dagma_s = float(dagma_s)

        # Edge MLPs
        self.edge_mlps = nn.ModuleList()
        self._edge_idx = {}
        idx = 0
        for i in range(self.d):
            for j in range(self.d):
                if i != j:
                    self.edge_mlps.append(EdgeMLP(edge_hidden, edge_layers))
                    self._edge_idx[(i, j)] = idx
                    idx += 1

        # Lag matrices
        init_scale = 1e-2
        self.Wk = nn.ParameterList(
            [nn.Parameter(init_scale * torch.randn(self.d, self.d)) for _ in range(self.K)]
        )

        # Prior
        P_prior_tensor = torch.tensor(P_prior, dtype=torch.float32)
        self.register_buffer("P_prior", P_prior_tensor)
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.d))

        P_clamped = torch.clamp(P_prior_tensor, self.eps_prior, 1.0 - self.eps_prior)
        self.register_buffer("_prior_logits", torch.log(P_clamped) - torch.log1p(-P_clamped))

        # Tau groups (for feature extraction)
        self.n_tau_groups = max(1, int(n_tau_groups))
        group_indices, actual_n_groups = self._build_tau_groups(P_prior_tensor)
        self.n_tau_groups = actual_n_groups
        self.register_buffer("group_indices", group_indices)

        # Trust Propagation — 一律用 Lite (NAM 本身已经很重, d*(d-1) 个 MLP)
        self.trust_module = TrustPropagationLite(
            tau_min=tau_min, tau_max=tau_max, hidden=trust_feat_dim
        )

        # Fallback
        init_tau = float(np.clip(tau_min, tau_min, tau_max))
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

    def get_W0_strength(self):
        dev = next(self.parameters()).device
        S = torch.zeros(self.d, self.d, device=dev)
        for (i, j), idx in self._edge_idx.items():
            S[i, j] = self.edge_mlps[idx].edge_strength()
        return S

    def get_W0_adj(self):
        return self.get_W0_strength()

    def _compute_tau_matrix(self):
        if self.learn_tau:
            W_str = self.get_W0_strength().detach()
            return self.trust_module(self.P_prior, W_str, self.group_indices, self.off_diag_mask)
        return self.tau_groups[self.group_indices]

    def get_tau(self):
        if self.learn_tau:
            return self.trust_module.get_tau_mean(
                self.P_prior, self.get_W0_strength().detach(),
                self.group_indices, self.off_diag_mask
            )
        return self.tau_groups.mean()

    def calibrated_prior(self, tau_matrix):
        return torch.sigmoid(self._prior_logits * tau_matrix)

    def omega_mask(self, tau_matrix):
        P_hat = self.calibrated_prior(tau_matrix)
        return (1.0 - P_hat) + self.delta

    def _compute_h_w0(self):
        S = self.get_W0_strength()
        s = self.dagma_s
        S2 = S * S
        M = s * torch.eye(self.d, device=S.device) - S2
        sign, logabsdet = torch.linalg.slogdet(M)
        if sign.item() <= 0:
            excess = torch.clamp(S2.sum(dim=1) - s, min=0.0)
            return self.d * 1.0 + excess.sum()
        return -logabsdet + self.d * math.log(s)

    def forward(self, X_t, X_lags):
        T = X_t.shape[0]
        dev = X_t.device
        pred = torch.zeros(T, self.d, device=dev)
        for (i, j), idx in self._edge_idx.items():
            x_i = X_t[:, i:i+1]
            pred[:, j] = pred[:, j] + self.edge_mlps[idx](x_i).squeeze(1)
        for k in range(self.K):
            pred = pred + X_lags[k] @ self.Wk[k]
        return pred

    def compute_losses(self, X_t, X_lags, rho, alpha):
        with torch.no_grad():
            tau_matrix = self._compute_tau_matrix()
        tau_matrix = tau_matrix.detach()
        Omega = self.omega_mask(tau_matrix)
        S = self.get_W0_strength()

        pred = self.forward(X_t, X_lags)
        residuals = X_t - pred
        T, d = residuals.shape
        loss_mse = 0.5 * torch.sum(residuals ** 2) / (T * d)

        P_hat = self.calibrated_prior(tau_matrix)
        coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
        loss_l1 = self.lambda1 * torch.sum(coeff * S)

        Omega_w0 = Omega * self.off_diag_mask
        loss_prior = 0.5 * self.lambda2 * torch.sum(Omega_w0 * (S ** 2))
        for k in range(self.K):
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (self.Wk[k] ** 2))

        h_val = self._compute_h_w0()
        loss_alm = loss_mse + loss_l1 + loss_prior + (alpha * h_val) + 0.5 * rho * (h_val ** 2)

        if not torch.isfinite(loss_alm):
            raise RuntimeError("loss_alm is NaN/Inf in NAM-Trust model.")

        tau_mean = tau_matrix[self.off_diag_mask.bool()].mean().detach()
        return loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau_mean

    def compute_eb_objective(self, X_t, X_lags):
        dev = X_t.device
        T, d = X_t.shape
        tau_matrix = self._compute_tau_matrix()
        Omega = self.omega_mask(tau_matrix)
        S = self.get_W0_strength().detach()

        Omega_w0 = Omega * self.off_diag_mask
        loss_prior = 0.5 * self.lambda2 * torch.sum(Omega_w0 * (S ** 2))
        for k in range(self.K):
            Wk_det = self.Wk[k].detach()
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (Wk_det ** 2))

        P_hat = self.calibrated_prior(tau_matrix)
        coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
        loss_l1 = self.lambda1 * torch.sum(coeff * S)

        s_max = S.max().clamp(min=1e-6)
        S_prob = (S / s_max).clamp(1e-6, 1.0 - 1e-6) * self.off_diag_mask
        P_hat_safe = P_hat.clamp(1e-6, 1.0 - 1e-6)
        agreement_loss = -torch.sum(
            (S_prob * torch.log(P_hat_safe) + (1.0 - S_prob) * torch.log(1.0 - P_hat_safe))
            * self.off_diag_mask
        )

        trust_reg = torch.tensor(0.0, device=dev)
        for p in self.trust_module.parameters():
            trust_reg = trust_reg + 0.01 * torch.sum(p ** 2)

        return agreement_loss + loss_prior + loss_l1 + trust_reg

    def calibrate_tau_from_data(self, X_t, X_lags):
        from scipy.stats import spearmanr
        with torch.no_grad():
            X_t_np = X_t.detach().cpu().numpy()
            C = np.abs(np.corrcoef(X_t_np.T))
            np.fill_diagonal(C, 0.0)
            P = self.P_prior.cpu().numpy()
            mask = self.off_diag_mask.bool().cpu().numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConstantInputWarning)
                corr, _ = spearmanr(C[mask], P[mask])
            if np.isnan(corr):
                corr = 0.0
            lo, hi = -0.05, 0.20
            if corr <= lo:
                target_sigmoid = 0.01
            elif corr >= hi:
                target_sigmoid = 0.99
            else:
                target_sigmoid = (corr - lo) / (hi - lo)
            target_sigmoid = max(0.01, min(0.99, target_sigmoid))
            bias_val = math.log(target_sigmoid / (1.0 - target_sigmoid))
            self.trust_module.tau_bias.data.fill_(bias_val)


def train_prcd_nam_trust_alm(
    model: PRCD_MAP_NAM_Trust,
    X_t: torch.Tensor,
    X_lags,
    max_iter: int = 35,
    inner_iter: int = 400,
    rho_0: float = 1.0,
    gamma: float = 3.0,
    rho_max: float = 1e6,
    tol: float = 1e-6,
    lr: float = 5e-4,
    verbose: bool = False,
    grad_clip: float = 5.0,
    tau_warmup: int = 0,
    tau_eb_steps: int = 8,
    tau_eb_lr: float = 1e-3,
):
    device = next(model.parameters()).device
    X_t = X_t.to(device)
    X_lags = [x.to(device) for x in X_lags]

    if model.learn_tau:
        model.calibrate_tau_from_data(X_t, X_lags)
        if verbose:
            print(f">>> [NAM-Trust] τ pre-calibrated: mean={float(model.get_tau()):.3f}")

    # Separate params
    svar_params = list(model.edge_mlps.parameters()) + list(model.Wk.parameters())
    trust_params = list(model.trust_module.parameters())

    rho = float(rho_0)
    alpha = 0.0

    for it in range(max_iter):
        optimizer = optim.Adam(svar_params, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=inner_iter, eta_min=lr * 0.01)

        for _ in range(inner_iter):
            loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau = \
                model.compute_losses(X_t, X_lags, rho, alpha)
            optimizer.zero_grad(set_to_none=True)
            loss_alm.backward()
            if grad_clip > 0:
                ec = float(grad_clip) * max(1.0, math.log1p(rho))
                torch.nn.utils.clip_grad_norm_(svar_params, max_norm=ec)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            h_now = float(model._compute_h_w0().detach().cpu().item())

        # EB update of trust module
        if model.learn_tau and it >= tau_warmup and tau_eb_steps > 0:
            trust_opt = optim.Adam(trust_params, lr=tau_eb_lr)
            for _ in range(tau_eb_steps):
                eb_loss = model.compute_eb_objective(X_t, X_lags)
                trust_opt.zero_grad(set_to_none=True)
                eb_loss.backward()
                torch.nn.utils.clip_grad_norm_(trust_params, max_norm=1.0)
                trust_opt.step()

        if verbose:
            tau_val = float(model.get_tau())
            print(
                f"[NAM-Trust outer {it+1:02d}/{max_iter}] "
                f"rho={rho:.3g}, alpha={alpha:.3g}, h={h_now:.3e}, "
                f"tau_mean={tau_val:.3f}, "
                f"loss_alm={float(loss_alm.detach().cpu().item()):.6f}"
            )

        if abs(h_now) <= tol:
            break

        alpha = alpha + rho * h_now
        rho = min(rho * float(gamma), float(rho_max))

    with torch.no_grad():
        W0_strength = model.get_W0_strength().detach().cpu().numpy()
        Wk_raw = [wk.detach().cpu().numpy() for wk in model.Wk]
        tau_est = float(model.get_tau())

    return W0_strength, Wk_raw, tau_est
