"""
model_prcd_map_nam.py — PRCD-MAP with Neural Additive Model (NAM) parameterization.

线性 SVAR 的非线性扩展: 每条边 (i→j) 用一个小 MLP f_{ij}(x_i) 替代线性权重 w_{ij} * x_i.
滞后项保持线性 (Wk 不变), 仅瞬时效应非线性化.

适用范围: d ≤ 30 (d=30 → 870 MLP, ~50K 参数).
d>30 时训练时间显著增长, 建议使用线性模型.

与 PRCD_MAP_Model 的关系:
  - 共享: τ 分组机制, sigmoid(logit(P)*τ) 校准, Omega mask, EB 更新
  - 替换: W0 线性矩阵 → d*(d-1) 个 EdgeMLP
  - DAGMA 约束作用于 edge_strength 矩阵 (替代 W0⊙W0)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from scipy.stats import ConstantInputWarning


class EdgeMLP(nn.Module):
    """单边非线性函数 f_{ij}: R → R."""

    def __init__(self, hidden=16, n_layers=2):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        # 小初始化, 防止初始输出过大
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (T, 1) → (T, 1)"""
        return self.net(x)

    def edge_strength(self):
        """可微分的边强度 (参数 Frobenius 范数)."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            total = total + torch.sum(p ** 2)
        return torch.sqrt(total + 1e-12)


class PRCD_MAP_NAM(nn.Module):
    """
    PRCD-MAP with Neural Additive Model (NAM) for instantaneous effects.

    pred[t, j] = sum_{i≠j} f_{ij}(X_t[t, i]) + sum_k X_lags[k] @ Wk[k][:, j]

    Edge strength matrix S[i,j] = edge_strength(f_{ij}) replaces |W0[i,j]|
    in DAGMA constraint, prior-weighted L1/L2, and evaluation.
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
        dagma_s: float = None,
        edge_hidden: int = 16,
        edge_layers: int = 2,
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
        self.tau_prior_sigma = float(tau_prior_sigma)

        if dagma_s is None:
            self.dagma_s = max(1.0, math.log(self.d)) * 1.0
        else:
            self.dagma_s = float(dagma_s)

        # Edge MLPs: d*(d-1) 个, 按 (i,j) i≠j 排列
        self.edge_mlps = nn.ModuleList()
        self._edge_idx = {}  # (i, j) → index in ModuleList
        idx = 0
        for i in range(self.d):
            for j in range(self.d):
                if i != j:
                    self.edge_mlps.append(EdgeMLP(edge_hidden, edge_layers))
                    self._edge_idx[(i, j)] = idx
                    idx += 1

        # 滞后矩阵 (保持线性)
        init_scale = 1e-2
        self.Wk = nn.ParameterList(
            [nn.Parameter(init_scale * torch.randn(self.d, self.d)) for _ in range(self.K)]
        )

        # 先验矩阵
        P_prior_tensor = torch.tensor(P_prior, dtype=torch.float32)
        self.register_buffer("P_prior", P_prior_tensor)
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.d))

        P_clamped = torch.clamp(P_prior_tensor, self.eps_prior, 1.0 - self.eps_prior)
        self.register_buffer("_prior_logits", torch.log(P_clamped) - torch.log1p(-P_clamped))

        # τ 配置
        self.learn_tau = bool(learn_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.n_tau_groups = max(1, int(n_tau_groups))

        group_indices, actual_n_groups = self._build_tau_groups(P_prior_tensor)
        self.n_tau_groups = actual_n_groups
        self.register_buffer("group_indices", group_indices)

        if self.learn_tau:
            init_tau = float(np.clip(self.tau_min, self.tau_min, self.tau_max))
        else:
            init_tau = float(np.clip(tau0, self.tau_min, self.tau_max))
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

    def _expand_tau(self):
        return self.tau_groups[self.group_indices]

    def _expand_tau_from(self, tau_groups_var):
        return tau_groups_var[self.group_indices]

    def get_tau(self):
        return self.tau_groups.mean()

    def set_tau(self, value: float):
        value = float(np.clip(value, self.tau_min, self.tau_max))
        self.tau_groups.fill_(value)

    def calibrated_prior(self, tau_matrix):
        return torch.sigmoid(self._prior_logits * tau_matrix)

    def omega_mask(self, tau_matrix):
        P_hat = self.calibrated_prior(tau_matrix)
        return (1.0 - P_hat) + self.delta

    # --------------------------------------------------------
    # Edge strength matrix
    # --------------------------------------------------------
    def get_W0_strength(self):
        """返回 (d, d) 边强度矩阵, 对角线为 0. 可微分."""
        dev = next(self.parameters()).device
        S = torch.zeros(self.d, self.d, device=dev)
        for (i, j), idx in self._edge_idx.items():
            S[i, j] = self.edge_mlps[idx].edge_strength()
        return S

    def get_W0_adj(self):
        """兼容接口: 返回 detached numpy-compatible strength matrix."""
        return self.get_W0_strength()

    # --------------------------------------------------------
    # DAGMA 约束 (作用于 edge strength)
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, X_t, X_lags):
        """
        X_t: (T, d), X_lags: list of (T, d).
        pred[t, j] = sum_{i≠j} f_{ij}(X_t[t, i:i+1]) + lag terms
        """
        T = X_t.shape[0]
        dev = X_t.device
        pred = torch.zeros(T, self.d, device=dev)

        # 瞬时 NAM 效应
        for (i, j), idx in self._edge_idx.items():
            x_i = X_t[:, i:i+1]  # (T, 1)
            pred[:, j] = pred[:, j] + self.edge_mlps[idx](x_i).squeeze(1)

        # 滞后线性效应
        for k in range(self.K):
            pred = pred + X_lags[k] @ self.Wk[k]

        return pred

    # --------------------------------------------------------
    # 损失计算
    # --------------------------------------------------------
    def compute_losses(self, X_t, X_lags, rho, alpha):
        tau_matrix = self._expand_tau()
        Omega = self.omega_mask(tau_matrix)
        S = self.get_W0_strength()

        pred = self.forward(X_t, X_lags)
        residuals = X_t - pred
        T, d = residuals.shape
        loss_mse = 0.5 * torch.sum(residuals ** 2) / (T * d)

        # Prior-weighted L1 on edge strength
        P_hat = self.calibrated_prior(tau_matrix)
        coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
        loss_l1 = self.lambda1 * torch.sum(coeff * S)

        # Prior-weighted L2 on edge strength + lag matrices
        Omega_w0 = Omega * self.off_diag_mask
        loss_prior = 0.5 * self.lambda2 * torch.sum(Omega_w0 * (S ** 2))
        for k in range(self.K):
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (self.Wk[k] ** 2))

        h_val = self._compute_h_w0()
        loss_alm = loss_mse + loss_l1 + loss_prior + (alpha * h_val) + 0.5 * rho * (h_val ** 2)

        if not torch.isfinite(loss_alm):
            raise RuntimeError("loss_alm is NaN/Inf in NAM model.")

        tau_mean = self.tau_groups.mean()
        return loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau_mean

    # --------------------------------------------------------
    # EB objective (Laplace approximation)
    # --------------------------------------------------------
    def compute_eb_objective(self, X_t, X_lags, tau_groups_var):
        dev = X_t.device
        T, d = X_t.shape
        tau_matrix = self._expand_tau_from(tau_groups_var)
        Omega = self.omega_mask(tau_matrix)

        S = self.get_W0_strength().detach()

        # Prior L2
        Omega_w0 = Omega * self.off_diag_mask
        loss_prior = 0.5 * self.lambda2 * torch.sum(Omega_w0 * (S ** 2))
        for k in range(self.K):
            Wk_det = self.Wk[k].detach()
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (Wk_det ** 2))

        # Prior L1
        P_hat = self.calibrated_prior(tau_matrix)
        coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
        loss_l1 = self.lambda1 * torch.sum(coeff * S)

        # Agreement loss
        s_max = S.max().clamp(min=1e-6)
        S_prob = (S / s_max).clamp(1e-6, 1.0 - 1e-6) * self.off_diag_mask
        P_hat_safe = P_hat.clamp(1e-6, 1.0 - 1e-6)
        agreement_loss = -torch.sum(
            (S_prob * torch.log(P_hat_safe) + (1.0 - S_prob) * torch.log(1.0 - P_hat_safe))
            * self.off_diag_mask
        )

        # τ regularization
        tau_reg = torch.tensor(0.0, device=dev)
        if self.tau_prior_sigma > 0:
            tau_reg = 0.5 * torch.sum((tau_groups_var - 0.5) ** 2) / (self.tau_prior_sigma ** 2)

        return agreement_loss + loss_prior + loss_l1 + tau_reg

    # --------------------------------------------------------
    # Pre-training τ calibration
    # --------------------------------------------------------
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

            lo, hi = -0.05, 0.40
            if corr <= lo:
                target = self.tau_min
            elif corr >= hi:
                target = self.tau_max
            else:
                frac = (corr - lo) / (hi - lo)
                target = self.tau_min * (1.0 - frac) + self.tau_max * frac

            target = float(np.clip(target, self.tau_min, self.tau_max))
            self.tau_groups.fill_(target)


# ============================================================
# 训练函数
# ============================================================
def train_prcd_nam_alm(
    model: PRCD_MAP_NAM,
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
    tau_eb_lr: float = 0.1,
):
    """ALM training for PRCD-MAP NAM. Same 3-level structure as linear version."""
    device = next(model.parameters()).device
    X_t = X_t.to(device)
    X_lags = [x.to(device) for x in X_lags]

    # Pre-training τ calibration
    if model.learn_tau:
        model.calibrate_tau_from_data(X_t, X_lags)
        if verbose:
            tau_str = ", ".join(f"{t:.3f}" for t in model.tau_groups.tolist())
            print(f">>> [NAM] τ pre-calibrated: [{tau_str}]")

    rho = float(rho_0)
    alpha = 0.0

    for it in range(max_iter):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=inner_iter, eta_min=lr * 0.01)

        for _ in range(inner_iter):
            loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau = \
                model.compute_losses(X_t=X_t, X_lags=X_lags, rho=rho, alpha=alpha)
            optimizer.zero_grad(set_to_none=True)
            loss_alm.backward()
            if grad_clip > 0:
                ec = float(grad_clip) * max(1.0, math.log1p(rho))
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ec)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            h_now = float(model._compute_h_w0().detach().cpu().item())

        # EB tau update
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
                f"[NAM outer {it+1:02d}/{max_iter}] "
                f"rho={rho:.3g}, alpha={alpha:.3g}, h={h_now:.3e}, "
                f"tau=[{tau_str}], "
                f"loss_alm={float(loss_alm.detach().cpu().item()):.6f}"
            )

        if abs(h_now) <= tol:
            break

        alpha = alpha + rho * h_now
        rho = min(rho * float(gamma), float(rho_max))

    # 提取结果
    with torch.no_grad():
        W0_strength = model.get_W0_strength().detach().cpu().numpy()
        Wk_raw = [wk.detach().cpu().numpy() for wk in model.Wk]
        tau_est = float(model.get_tau().detach().cpu().item())

    return W0_strength, Wk_raw, tau_est
