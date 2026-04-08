"""
model_prcd_map.py — PRCD-MAP v3 (Empirical Bayes)
改动 (相对 v2):
  - 分组温度 τ_g (按先验概率分位数分组, 默认4组)
  - 经验贝叶斯更新 τ: 双层优化, Laplace近似边际似然
  - 三层优化结构: 外层ALM(α,ρ) → 中间层EB更新τ → 内层Adam优化W
  - 去掉旧的 agreement-based τ 更新 和 adaptive_lambda
  - Hessian对角近似 (线性SVAR解析形式, 计算代价可控)
"""

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ConstantInputWarning


class PRCD_MAP_Model(nn.Module):
    """
    MAP-consistent PRCD with Empirical Bayes temperature calibration.

    核心改进: τ 从全局标量升级为分组向量 τ_g, 通过最大化 Laplace 近似
    边际似然在 ALM 外循环之间更新, 与内层 W 优化解耦.

    参数化: sigmoid(logit(P) * τ), τ∈[0.01, 1.0].
    τ=1 → 完全使用先验, τ→0 → P_hat→0.5 (忽略先验).
    梯度 ∂sigmoid/∂τ = logit(P) · sigmoid'(...), 不因 P≈0.5 而消失.

    分组策略: 按 P_prior 非对角元素的分位数将边分为 n_tau_groups 组,
    同组边共享温度参数. 低先验/高先验区域可独立校准, 抗 systematic bias.
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

        # 损失函数类型
        self.loss_type = str(loss_type)
        if self.loss_type not in ("mse", "huber", "laplace"):
            raise ValueError(f"loss_type must be 'mse'/'huber'/'laplace', got '{loss_type}'")
        self.huber_delta = float(huber_delta)
        self.prior_l1_weight = bool(prior_l1_weight)
        self.tau_prior_sigma = float(tau_prior_sigma)

        # 无环约束类型
        if acyclicity not in ("dagma", "notears"):
            raise ValueError(f"acyclicity must be 'dagma' or 'notears', got '{acyclicity}'")
        self.acyclicity = acyclicity
        if dagma_s is None:
            self.dagma_s = max(1.0, math.log(self.d)) * 1.0
        else:
            self.dagma_s = float(dagma_s)

        # 模型参数
        init_scale = 1e-2
        self.W0 = nn.Parameter(init_scale * torch.randn(self.d, self.d))
        self.Wk = nn.ParameterList(
            [nn.Parameter(init_scale * torch.randn(self.d, self.d)) for _ in range(self.K)]
        )

        # 先验矩阵
        P_prior_tensor = torch.tensor(P_prior, dtype=torch.float32)
        if P_prior_tensor.shape != (self.d, self.d):
            raise ValueError(f"P_prior must have shape ({self.d},{self.d}), got {tuple(P_prior_tensor.shape)}")
        self.register_buffer("P_prior", P_prior_tensor)
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(self.d))

        # 预计算 logit(P_prior), 避免重复计算
        P_clamped = torch.clamp(P_prior_tensor, self.eps_prior, 1.0 - self.eps_prior)
        self.register_buffer("_prior_logits", torch.log(P_clamped) - torch.log1p(-P_clamped))

        # τ 配置
        self.learn_tau = bool(learn_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        if not (self.tau_min < self.tau_max):
            raise ValueError("tau_min must be < tau_max")

        # ---- 分组 τ ----
        self.n_tau_groups = max(1, int(n_tau_groups))
        group_indices, actual_n_groups = self._build_tau_groups(P_prior_tensor)
        self.n_tau_groups = actual_n_groups
        self.register_buffer("group_indices", group_indices)

        # 初始化: learn_tau 时从 τ_min 起步 (保守策略: 先忽略先验, 等 EB 验证后再提升)
        # 乘法参数化: τ→0 忽略先验, τ=1 完全使用先验
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
        按 P_prior 非对角元素的分位数将边分组.
        返回 (group_indices: LongTensor[d,d], actual_n_groups: int).
        """
        mask = self.off_diag_mask.bool()
        p_offdiag = P_prior_tensor[mask]

        if self.n_tau_groups <= 1:
            return torch.zeros(self.d, self.d, dtype=torch.long), 1

        # 分位点边界 (去重以处理大量相同值的情况)
        quantile_pts = torch.linspace(0.0, 1.0, self.n_tau_groups + 1)[1:-1]
        boundaries = torch.quantile(p_offdiag.float(), quantile_pts)
        boundaries = torch.unique(boundaries)
        actual_n_groups = len(boundaries) + 1

        group_indices = torch.bucketize(P_prior_tensor, boundaries)
        return group_indices.long(), actual_n_groups

    # --------------------------------------------------------
    # τ 相关
    # --------------------------------------------------------
    def _expand_tau(self) -> torch.Tensor:
        """将分组 τ 扩展为 (d, d) 矩阵."""
        return self.tau_groups[self.group_indices]

    def _expand_tau_from(self, tau_groups_var: torch.Tensor) -> torch.Tensor:
        """从给定的 τ 向量扩展 (支持 autograd)."""
        return tau_groups_var[self.group_indices]

    def get_tau(self) -> torch.Tensor:
        """返回 τ 均值 (用于日志/兼容)."""
        return self.tau_groups.mean()

    def set_tau(self, value: float):
        """将所有 τ 组设为同一值."""
        value = float(np.clip(value, self.tau_min, self.tau_max))
        self.tau_groups.fill_(value)

    def get_W0_adj(self) -> torch.Tensor:
        return self.W0 * self.off_diag_mask

    def calibrated_prior(self, tau_matrix: torch.Tensor) -> torch.Tensor:
        """
        tau_matrix: (d, d).
        乘法参数化: sigmoid(logit(P) * τ).
        τ=1 → 完全使用先验, τ→0 → P_hat→0.5 (忽略先验).
        梯度 ∂/∂τ = logit(P), 不因 P≈0.5 而消失.
        """
        return torch.sigmoid(self._prior_logits * tau_matrix)

    def omega_mask(self, tau_matrix: torch.Tensor) -> torch.Tensor:
        P_hat = self.calibrated_prior(tau_matrix)
        return (1.0 - P_hat) + self.delta

    # --------------------------------------------------------
    # 无环约束
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
    # 损失计算
    # --------------------------------------------------------
    def _compute_robust_loss(self, residuals: torch.Tensor,
                             obs_mask: torch.Tensor = None) -> torch.Tensor:
        """
        鲁棒回归损失, 返回标量.
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
        """先验调制L1: 高先验概率边减少惩罚, 低先验概率边增加惩罚."""
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
        """内层损失: τ 固定 (从 buffer 读取, 无梯度).
        obs_mask: (T, d) binary mask for missing data support. None=all observed.
        """
        tau_matrix = self._expand_tau()
        Omega = self.omega_mask(tau_matrix)

        pred = self.forward(X_t, X_lags)
        residuals = X_t - pred
        loss_mse = self._compute_robust_loss(residuals, obs_mask=obs_mask)
        loss_l1 = self._compute_prior_adjusted_l1(tau_matrix)

        W0_adj = self.get_W0_adj()

        # 先验加权 L2 (prior-weighted ridge)
        loss_prior = 0.0
        for k in range(self.K):
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (self.Wk[k] ** 2))
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega_w0 * (W0_adj ** 2))

        # 无环约束 + ALM
        h_val = self._compute_h_w0()
        loss_alm = loss_mse + loss_l1 + loss_prior + (alpha * h_val) + 0.5 * rho * (h_val ** 2)

        if not torch.isfinite(loss_alm):
            raise RuntimeError("loss_alm is NaN/Inf. Try smaller lr / stronger clamp / smaller init_scale.")

        tau_mean = self.tau_groups.mean()
        return loss_alm, loss_mse, loss_l1, loss_prior, h_val, tau_mean

    # --------------------------------------------------------
    # Prior quality assessment (训练前快速评估)
    # --------------------------------------------------------
    def calibrate_tau_from_data(self, X_t: torch.Tensor, X_lags):
        """
        训练前用快速 Lasso VAR 估计边, 与 P_prior 比较, 设置 τ.

        原理: OLS/Lasso 回归无需 DAG 约束, 毫秒级完成. 得到的
        系数绝对值反映变量间关联强度. 将此与 P_prior 做 Spearman
        相关: 正相关 → 先验可信 (τ=1.0), 否则 → 先验不可靠 (τ_max).
        """
        from scipy.stats import spearmanr

        with torch.no_grad():
            X_t_np = X_t.detach().cpu().numpy()
            d = self.d

            # 简单成对相关作为边强度代理 (快速, 对 d≤100 足够)
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

            # Smooth mapping: corr → τ (乘法参数化)
            # High corr → high τ (trust prior), low/negative corr → low τ (ignore prior)
            # Piecewise linear:
            #   corr ≤ -0.05  → τ = tau_min  (bad prior, ignore)
            #   corr ≥  0.20  → τ = tau_max  (good prior, full trust)
            #   in between    → linear interpolation
            # 注: 阈值从0.40降为0.20, 让中等质量先验也能获得较高tau
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
    # 经验贝叶斯: Laplace 近似边际似然 (保留备用)
    # --------------------------------------------------------
    def compute_eb_objective(
        self, X_t: torch.Tensor, X_lags, tau_groups_var: torch.Tensor
    ) -> torch.Tensor:
        """
        计算经验贝叶斯目标 (负对数边际似然的 Laplace 近似).

        L_EB(τ) = L_prior(W*, τ) + L_l1(W*, τ) + (1/2) Σ log H_ii(τ)

        其中 W* 固定 (detach), τ_groups_var 带梯度.
        数据拟合项 L_data(W*) 不依赖 τ, 已省略.

        Hessian 对角近似 (线性 SVAR 解析形式):
          H_{W0[i,j]} = ‖X_t[:,i]‖²/(T·d) + λ₂·Ω_{ij}(τ)
          H_{Wk[i,j]} = ‖X_lag_k[:,i]‖²/(T·d) + λ₂·Ω_{ij}(τ)

        Parameters
        ----------
        tau_groups_var : (n_tau_groups,) tensor, requires_grad=True
        """
        dev = X_t.device
        T, d = X_t.shape

        # τ 矩阵 (梯度流经 tau_groups_var → group_indices 索引)
        tau_matrix = self._expand_tau_from(tau_groups_var)
        Omega = self.omega_mask(tau_matrix)

        W0_adj = self.get_W0_adj().detach()

        # --- 先验 L2 项 ---
        loss_prior = torch.tensor(0.0, device=dev)
        for k in range(self.K):
            Wk_det = self.Wk[k].detach()
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega * (Wk_det ** 2))
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            loss_prior = loss_prior + 0.5 * self.lambda2 * torch.sum(Omega_w0 * (W0_adj ** 2))

        # --- 先验调制 L1 项 ---
        if self.prior_l1_weight:
            P_hat = self.calibrated_prior(tau_matrix)
            coeff = torch.clamp(1.5 - P_hat, 0.1, 1.5) * self.off_diag_mask
            loss_l1 = self.lambda1 * torch.sum(coeff * torch.abs(W0_adj))
        else:
            loss_l1 = self.lambda1 * torch.sum(torch.abs(W0_adj))

        # --- Laplace 近似: (1/2) Σ log H_ii ---
        X_t_det = X_t.detach()
        # 数据项 Hessian 对角 (MSE 近似, 对 Huber/Laplace 也合理)
        data_hess_row = (X_t_det ** 2).sum(0) / (T * d)  # (d,) — 第 i 行的贡献

        log_det_term = torch.tensor(0.0, device=dev)

        # W0 贡献 (仅非对角)
        if self.apply_prior_to_w0:
            Omega_w0 = Omega * self.off_diag_mask
            H_w0 = data_hess_row.unsqueeze(1).expand(d, d) + self.lambda2 * Omega_w0
            log_det_term = log_det_term + torch.sum(
                torch.log(H_w0.clamp(min=1e-10)) * self.off_diag_mask
            )

        # Wk 贡献
        for k_idx in range(self.K):
            X_lag_det = X_lags[k_idx].detach()
            data_hess_k = (X_lag_det ** 2).sum(0) / (T * d)
            H_wk = data_hess_k.unsqueeze(1).expand(d, d) + self.lambda2 * Omega
            log_det_term = log_det_term + torch.sum(torch.log(H_wk.clamp(min=1e-10)))

        # --- Agreement 项: 先验与 W* 的一致性交叉熵 ---
        # 如果先验好, P_hat 与 |W*| 的边模式一致, 交叉熵低 → 推 τ 高
        # 如果先验差, 交叉熵高 → 推 τ 低 (P_hat→0.5, 交叉熵趋于 log2)
        # 这比 loss_prior+loss_l1 更直接: 它显式衡量先验预测数据的能力
        P_hat = self.calibrated_prior(tau_matrix)
        W0_abs = torch.abs(W0_adj)
        # soft targets: 将 |W*| 映射到 [0,1] 概率
        w_max = W0_abs.max().clamp(min=1e-6)
        W0_prob = (W0_abs / w_max).clamp(1e-6, 1.0 - 1e-6) * self.off_diag_mask
        P_hat_safe = P_hat.clamp(1e-6, 1.0 - 1e-6)
        # 二元交叉熵 (soft)
        agreement_loss = -torch.sum(
            (W0_prob * torch.log(P_hat_safe) + (1.0 - W0_prob) * torch.log(1.0 - P_hat_safe))
            * self.off_diag_mask
        )

        # --- τ 正则: 弱二次惩罚防止极端值 ---
        tau_reg = torch.tensor(0.0, device=dev)
        if self.tau_prior_sigma > 0:
            tau_reg = 0.5 * torch.sum((tau_groups_var - 0.5) ** 2) / (self.tau_prior_sigma ** 2)

        return agreement_loss + 0.5 * log_det_term + tau_reg


# ============================================================
# OLS Warm-Start
# ============================================================
def _ols_warm_start(model: PRCD_MAP_Model, X_t: torch.Tensor, X_lags, verbose=False):
    """
    用 Ridge 回归为 W0, Wk 提供初始值, 避免随机初始化陷入局部最优.
    标准做法: NOTEARS (Zheng et al. 2018) 也使用类似 warm-start.
    """
    d = model.d
    K = model.K
    T = X_t.shape[0]

    with torch.no_grad():
        # 构建回归矩阵: Y = X_t, Regressors = [X_t, X_lag1, ..., X_lagK]
        # 对每个目标变量 j, 解: x_t[:,j] = X_t @ w0[:,j] + sum_k X_lag_k @ wk[:,j]
        # 为简化, 用全局 Ridge: W = (R^T R + lambda I)^{-1} R^T Y
        parts = [X_t]
        for lag in X_lags:
            parts.append(lag)
        R = torch.cat(parts, dim=1)  # (T, d*(K+1))
        ridge_lam = 0.1  # 较强正则防止过拟合
        RtR = R.T @ R + ridge_lam * torch.eye(R.shape[1], device=R.device)
        RtY = R.T @ X_t
        try:
            W_all = torch.linalg.solve(RtR, RtY)  # (d*(K+1), d)
        except torch.linalg.LinAlgError:
            if verbose:
                print(">>> OLS warm-start: solve failed, using random init")
            return

        # 提取 W0 (前d行), Wk (后续d行)
        W0_init = W_all[:d, :]  # (d, d)
        # 清除对角线 (自回归不计入 W0)
        W0_init.fill_diagonal_(0.0)
        # 缩放: 防止初始值过大导致 DAG constraint 爆炸
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
# 训练函数
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
    ALM 训练:

    训练前: OLS warm-start + 快速 Ridge VAR 估计先验质量, 设置 τ
    外层 (Augmented Lagrangian): 更新 α, ρ 以强制无环约束
      内层 (Adam): 固定 τ, 优化 W₀, W₁:K (with early stopping)
    支持 lambda scheduling: 前1/3用大lambda促进稀疏, 后2/3降低精调
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
        """执行 ALM 外循环, 返回 (rho, alpha, h_now)."""
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

    # === 训练前 τ 校准: 快速 Ridge VAR 估计 + 与 P_prior 相关性 ===
    if model.learn_tau:
        model.calibrate_tau_from_data(X_t, X_lags)
        if verbose:
            tau_str = ", ".join(f"{t:.3f}" for t in model.tau_groups.tolist())
            print(f">>> τ pre-calibrated: [{tau_str}]")

    # === Lambda scheduling: 前1/3用大lambda促稀疏, 后2/3降低精调 ===
    lambda1_orig = model.lambda1
    lambda1_warmup = lambda1_orig * 5.0  # Phase 1: 5x lambda for sparsity
    phase1_end = max_iter // 3

    # === 主训练循环: 三层优化 (外层ALM → 内层Adam优化W → 中间层EB更新τ) ===
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

    # 恢复原始 lambda1
    model.lambda1 = lambda1_orig

    # 提取结果
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
