"""
run_dycast_standalone.py — Run DyCAST baseline in an isolated conda environment.

DyCAST: Learning Dynamic Causal Structure from Time Series (Cheng et al., ICLR 2025)
  Paper:  https://openreview.net/forum?id=WjDjem8mWE
  Code:   https://github.com/Cyue0316/DyCAST

This script is self-contained (no exp_utils dependency) and mirrors the
data-generation logic of run_rhino_standalone.py to ensure identical datasets.

Dependencies (install in a fresh conda env):
  conda create -n dycast python=3.10
  conda activate dycast
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install torchdiffeq geotorch scikit-learn pandas numpy

Usage (on GPU server):
  conda activate dycast
  python run_dycast_standalone.py --sub noise   --seeds 0 1 2
  python run_dycast_standalone.py --sub noise   --seeds 0 1 2 --gpu 0

After running, copy the output CSV back and merge:
  python run_dycast_standalone.py --merge
"""

import os, sys, time, warnings, argparse, traceback
import numpy as np
import pandas as pd

# =====================================================================
# Minimal data generation (self-contained, identical to run_rhino_standalone)
# =====================================================================

def set_seed(seed):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def make_er_dag(d, edge_prob=0.15, w_range=(0.3, 0.8), seed=0):
    rng = np.random.RandomState(seed)
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.rand() < edge_prob:
                B[i, j] = 1
    perm = rng.permutation(d)
    B = B[perm][:, perm]
    W = B * (rng.uniform(*w_range, size=(d, d)) * rng.choice([-1, 1], size=(d, d)))
    return W


def make_lag_matrices(d, K, edge_prob=0.10, scale=0.25, seed=0):
    rng = np.random.RandomState(seed + 1000)
    Wk_list = []
    for k in range(K):
        mask = (rng.rand(d, d) < edge_prob).astype(float)
        W = mask * rng.randn(d, d) * scale
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        if sr > 0.8:
            W *= 0.75 / (sr + 1e-8)
        Wk_list.append(W)
    return Wk_list


def simulate_svar_linear(T, W0, Wk_list, noise_type="gaussian", seed=0):
    rng = np.random.RandomState(seed)
    d = W0.shape[0]
    K = len(Wk_list)
    A = np.eye(d) - W0
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    burn = 200
    X = np.zeros((T + burn, d))
    for t in range(K, T + burn):
        if noise_type == "gaussian":
            eps = rng.randn(d)
        elif noise_type == "laplace":
            eps = rng.laplace(0, 1, d)
        elif noise_type == "student_t":
            eps = rng.standard_t(5, d)
        elif noise_type == "heteroscedastic":
            scale = 0.5 + np.abs(X[t - 1]) * 0.3
            eps = rng.randn(d) * scale
        else:
            eps = rng.randn(d)
        lag_contrib = sum(X[t - k - 1] @ Wk_list[k] for k in range(K))
        X[t] = (lag_contrib + eps) @ A_inv.T
    X = X[burn:]
    if not np.all(np.isfinite(X)) or X.std() < 1e-10:
        return None
    return standardize(X)


def gen_prior(W0_true, Wk_true, acc, mode="random", seed=0):
    rng = np.random.RandomState(seed)
    d = W0_true.shape[0]
    B_true = (np.abs(W0_true) > 1e-10).astype(float)
    for Wk in Wk_true:
        B_true = np.maximum(B_true, (np.abs(Wk) > 1e-10).astype(float))
    np.fill_diagonal(B_true, 0)
    P = np.full((d, d), 0.5)
    mask = np.ones((d, d), dtype=bool)
    np.fill_diagonal(mask, False)
    n_off = mask.sum()
    n_correct = int(acc * n_off)
    indices = np.argwhere(mask)
    chosen = rng.choice(len(indices), n_correct, replace=False)
    for idx in chosen:
        i, j = indices[idx]
        if B_true[i, j] > 0:
            P[i, j] = rng.uniform(0.7, 0.95)
        else:
            P[i, j] = rng.uniform(0.05, 0.3)
    np.fill_diagonal(P, 0.5)
    return P


# =====================================================================
# Metrics (self-contained)
# =====================================================================

def compute_metrics(B_true, W_est):
    from sklearn.metrics import roc_auc_score, average_precision_score
    d = B_true.shape[0]
    mask = ~np.eye(d, dtype=bool)
    y_true = B_true[mask].ravel().astype(int)
    y_score = np.abs(W_est[mask]).ravel()
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        auroc = 0.5
        auprc = float(y_true.mean())
    else:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    best_f1, best_thr = 0.0, 0.0
    for pct in range(1, 100):
        thr = np.percentile(y_score, pct)
        y_pred = (y_score > thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    y_pred_best = (y_score > best_thr).astype(int)
    shd = int(np.sum(y_pred_best != y_true))
    return dict(auroc=auroc, auprc=auprc, f1_opt=best_f1, shd=shd)


def compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est):
    B_w0 = (np.abs(W0_true) > 1e-10).astype(float)
    met_w0 = compute_metrics(B_w0, W0_est)
    res = {f"w0_{k}": v for k, v in met_w0.items()}

    B_comb = (np.abs(W0_true) > 1e-10).astype(int)
    for Wk_t in Wk_true:
        B_comb = np.maximum(B_comb, (np.abs(Wk_t) > 1e-10).astype(int))
    W_comb_est = np.abs(W0_est).copy()
    for Wk_e in Wk_est:
        W_comb_est = np.maximum(W_comb_est, np.abs(Wk_e))
    met_comb = compute_metrics(B_comb.astype(float), W_comb_est)
    res.update({f"comb_{k}": v for k, v in met_comb.items()})
    res.update({k: v for k, v in met_comb.items()})
    return res


# =====================================================================
# DyCAST model (from https://github.com/Cyue0316/DyCAST)
# =====================================================================

def _build_dycast_model(dims, k_dims=64, lag=1):
    """Import and build CausalODE model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchdiffeq import odeint_adjoint as odeint

    # --- Inline DyCAST model code (Cheng et al. ICLR 2025) ---

    class SineLayer(nn.Module):
        def __init__(self, in_features, out_features, bias=True,
                     is_first=False, omega_0=30):
            super().__init__()
            self.omega_0 = omega_0
            self.is_first = is_first
            self.in_features = in_features
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.init_weights()

        def init_weights(self):
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features,
                                                 1 / self.in_features)
                else:
                    self.linear.weight.uniform_(
                        -np.sqrt(6 / self.in_features) / self.omega_0,
                         np.sqrt(6 / self.in_features) / self.omega_0)

        def forward(self, input):
            return torch.sin(self.omega_0 * self.linear(input))

    class Func(torch.nn.Module):
        def __init__(self, input_dims, hidden_dims, dims):
            super().__init__()
            self.input_dims = input_dims
            self.hidden_dims = hidden_dims
            self.dims = dims
            self.linear = nn.Sequential(
                nn.Linear(input_dims, 2 * input_dims, bias=True),
                nn.Tanh(),
                nn.Linear(input_dims * 2, input_dims, bias=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(input_dims + 1, 512),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, dims * dims),
            )

        def constraint(self, z, t):
            def _h_fun(z, s=1.0, t=1.0):
                z = F.tanh(z)
                t_tensor = torch.tensor(t).unsqueeze(0).unsqueeze(1).to(z)
                z = torch.cat((z, t_tensor), dim=1)
                w_est = self.decoder(z).reshape(self.dims, self.dims)
                h = torch.trace(torch.matrix_exp(w_est ** 2)) - self.dims
                return h
            z = z.detach().requires_grad_(True)
            h = _h_fun(z, t=t)
            jacobian = torch.autograd.functional.jacobian(_h_fun, z, create_graph=True)
            F_ = jacobian.t() @ torch.linalg.pinv(jacobian @ jacobian.t()).to(z.device)
            return -1 * F_ * h

        def forward(self, t, z):
            x = self.linear(z)
            constrain = self.constraint(x, t)
            return x + constrain.t().reshape(1, -1)

    class LowRankLinear(nn.Module):
        def __init__(self, in_features, k):
            super().__init__()
            self.d = in_features
            self.k = k
            self.u = nn.Conv1d(self.d, self.k, kernel_size=1, bias=False)
            self.v = nn.Conv1d(self.k, self.d, kernel_size=1, bias=False)

        def forward(self, x):
            self.E_est = self.u.weight.squeeze(2)
            self.A_est = self.v.weight.squeeze(2)
            w_est = (self.E_est.transpose(0, 1) @ self.A_est.transpose(0, 1))
            out = self.u(x)
            out = self.v(out)
            return out, w_est

    class encoder(nn.Module):
        def __init__(self, in_features, k):
            super().__init__()
            self.line1 = nn.Linear(in_features, k, bias=True)

        def forward(self, x):
            x = self.line1(x)
            x = F.relu(x)
            return x

    class CausalODE(nn.Module):
        def __init__(self, dims=5, k_dims=4, lag=2):
            super().__init__()
            self.d = dims
            self.k = k_dims
            self.lag = lag
            self.encoder = encoder(2 * dims, k_dims)
            self.func = Func(hidden_dims=k_dims, input_dims=dims * k_dims, dims=dims)
            self.init_intra_t = nn.Parameter(torch.randn(dims, k_dims), requires_grad=True)
            self.init_intra_s = nn.Parameter(torch.randn(k_dims, dims), requires_grad=True)
            nn.init.kaiming_uniform_(self.init_intra_t)
            nn.init.kaiming_uniform_(self.init_intra_s)
            self.layers = nn.ModuleList([LowRankLinear(self.d, self.k) for _ in range(self.lag)])

        def terminal_value_CDE(self, w0, length):
            t = torch.linspace(1, length, length).to(w0.device)
            intra_t = odeint(self.func, w0, t, method="rk4", atol=1e-9, rtol=1e-7).permute(1, 0, 2)
            return intra_t

        def h_func(self, s=1.0):
            h = 0
            for t in range(self.west_t.size(0)):
                h += torch.trace(torch.matrix_exp(self.west_t[t] * self.west_t[t])) - self.d
            return h

        def l1_reg(self):
            west_t = self.west_t.permute(2, 1, 0)
            loss = torch.norm(self.p_est, p=1, dim=(0, 1)).sum() + \
                   torch.norm(west_t, p=1, dim=(0, 1)).sum()
            return loss

        def diag_zero(self):
            diag_loss = 0
            for i in range(self.west_t.size(0)):
                diag_loss += torch.trace(self.west_t[i] * self.west_t[i])
            return diag_loss

        def laplacian_loss(self):
            loss = 0.0
            for t in range(1, self.west_t.size(0) - 1):
                laplacian = self.west_t[t - 1] + self.west_t[t + 1] - 2 * self.west_t[t]
                loss += torch.sum(laplacian ** 2)
            return loss

        def forward(self, x):
            x_t = x
            length = x.size(1)
            self.init_intra = torch.matmul(self.init_intra_t, self.init_intra_s)
            patchs = [torch.cat((self.init_intra[i, :], self.init_intra[:, i]), dim=0)
                       for i in range(self.d)]
            patchs = torch.stack(patchs, dim=0)
            t = torch.linspace(1, length, length).unsqueeze(0).unsqueeze(2).to(x.device)
            z0 = self.encoder(patchs).reshape(1, -1)
            self.west_h = F.tanh(self.terminal_value_CDE(z0, length))
            self.west_h = torch.cat((self.west_h, t), dim=2)
            self.west_t = self.func.decoder(self.west_h).reshape(length, self.d, self.d)
            intra_output = torch.einsum('btd, tdk -> btk', x_t, self.west_t)
            output = []
            causal = []
            for i, layer in enumerate(self.layers.to(x.device)):
                out, A_est = layer(x[:, :-(i + 1), :].permute(0, 2, 1))
                output.append(out)
                causal.append(A_est)
            output = torch.stack(output, dim=2)
            self.p_est = torch.stack(causal, dim=2)
            output = torch.sum(output, dim=2).permute(0, 2, 1)
            output = torch.cat([torch.zeros_like(output[:, :1, :]), output], dim=1).to(x.device)
            out = intra_output + output
            return out

    return CausalODE(dims=dims, k_dims=k_dims, lag=lag)


# =====================================================================
# DyCAST training & extraction
# =====================================================================

def run_dycast(X, d, K=1, k_dims=64, epochs=300, lr=1e-3,
               lam_sparse=0.01, lam_dag=1.0, lam_diag=0.1, lam_smooth=0.01,
               batch_size=64, seed=0, device=None):
    """
    Train DyCAST on time series X (T, d) and extract causal adjacency.

    DyCAST learns *dynamic* (time-varying) causal structure. Since the
    ground truth in our benchmark is static, we aggregate the learned
    time-varying adjacency by taking the mean absolute value over time.

    Returns (W0_est, Wk_est_list) matching exp_utils convention.
    """
    import torch
    import torch.nn.functional as F

    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_dycast_model(dims=d, k_dims=k_dims, lag=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_std = standardize(X)
    X_tensor = torch.tensor(X_std, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, d)
    T_len = X_tensor.size(1)

    # For batch training: slide windows of length `window_len`
    # DyCAST needs full sequence for ODE solve; use mini-batch of sub-sequences
    window_len = min(T_len, 100)  # cap window to avoid ODE memory blow-up
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 30

    for epoch in range(epochs):
        model.train()

        # Random sub-sequence for this epoch
        if T_len > window_len:
            start = np.random.randint(0, T_len - window_len)
            x_batch = X_tensor[:, start:start + window_len, :]
        else:
            x_batch = X_tensor

        try:
            x_pred = model(x_batch)
        except Exception as e:
            warnings.warn(f"DyCAST forward failed at epoch {epoch}: {e}")
            continue

        # Reconstruction loss
        loss_recon = F.mse_loss(x_pred, x_batch)

        # Sparsity
        loss_l1 = model.l1_reg()

        # Acyclicity
        loss_dag = model.h_func()

        # Diagonal zero (no self-loops)
        loss_diag = model.diag_zero()

        # Temporal smoothness
        loss_smooth = model.laplacian_loss() if x_batch.size(1) > 2 else 0.0

        loss = (loss_recon
                + lam_sparse * loss_l1
                + lam_dag * loss_dag
                + lam_diag * loss_diag
                + lam_smooth * loss_smooth)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss_recon.detach())

        # NaN guard: stop training, use best checkpoint
        if not np.isfinite(loss_val):
            print(f"    DyCAST NaN at epoch {epoch}, rolling back to best checkpoint")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience and epoch > 50:
            break

        if (epoch + 1) % 50 == 0:
            print(f"    DyCAST epoch {epoch+1}/{epochs}  "
                  f"recon={loss_val:.4f}  "
                  f"h={float(loss_dag):.4f}  "
                  f"l1={float(loss_l1):.4f}")

    # Load best checkpoint if available
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Extract causal adjacency ---
    model.eval()
    with torch.no_grad():
        _ = model(X_tensor)  # full-sequence forward to populate west_t and p_est

        # Instantaneous: west_t has shape (T, d, d)
        # Aggregate dynamic graph to static by taking mean |W| over time
        west_t = model.west_t.detach().cpu().numpy()  # (T, d, d)
        W0_est = np.mean(np.abs(west_t), axis=0)       # (d, d)
        np.fill_diagonal(W0_est, 0.0)

        # Lagged: p_est has shape (d, d, K)
        p_est = model.p_est.detach().cpu().numpy()      # (d, d, K)
        Wk_est = []
        for k in range(K):
            Wk = np.abs(p_est[:, :, k])
            np.fill_diagonal(Wk, 0.0)
            Wk_est.append(Wk)

    # Final NaN guard
    if not np.all(np.isfinite(W0_est)):
        W0_est = np.zeros((d, d))
        Wk_est = [np.zeros((d, d)) for _ in range(K)]
        warnings.warn("DyCAST output contains NaN, returning zeros")

    return W0_est, Wk_est


# =====================================================================
# Experiment settings (matching exp1 sub=noise exactly)
# =====================================================================

PRIOR_ACCS = [0.0, 0.4, 0.6, 0.9]


def get_unique_settings(seeds, noise_types=None):
    """Only unique (noise, seed) combos — DyCAST is prior-agnostic."""
    if noise_types is None:
        noise_types = ["gaussian", "laplace", "student_t", "heteroscedastic"]
    settings = []
    for noise in noise_types:
        for seed in seeds:
            settings.append(dict(d=20, T=500, K=1, graph="ER", noise=noise, seed=seed))
    return settings


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="DyCAST standalone runner")
    parser.add_argument("--sub", type=str, default="noise",
                        help="Sub-experiment: noise")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index (default: auto)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--k_dims", type=int, default=64,
                        help="DyCAST latent rank (default 64)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_types", type=str, nargs="+", default=None,
                        help="Subset of noise types to run (for parallel split)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge DyCAST results into main exp1_noise CSV")
    parser.add_argument("--output_dir", type=str, default="dycast_results")
    args = parser.parse_args()

    if args.merge:
        _merge_results(args.output_dir)
        return

    import torch
    if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sub != "noise":
        raise ValueError(f"Unknown sub-experiment: {args.sub}")

    unique_settings = get_unique_settings(args.seeds, noise_types=args.noise_types)

    csv_path = os.path.join(args.output_dir, f"dycast_{args.sub}.csv")
    # Resume: check which (noise, seed) are already fully done
    done_keys = set()
    if os.path.exists(csv_path):
        df_prev = pd.read_csv(csv_path)
        for (noise, seed), grp in df_prev.groupby(["noise", "seed"]):
            if len(grp) >= len(PRIOR_ACCS):
                done_keys.add(f"{noise}_{seed}")
        print(f"Resuming: {len(done_keys)} unique (noise,seed) combos already done.")
    else:
        df_prev = pd.DataFrame()

    all_rows = list(df_prev.to_dict("records")) if not df_prev.empty else []
    n_total = len(unique_settings)
    t_global = time.time()

    for idx, st in enumerate(unique_settings):
        d, T, K = st["d"], st["T"], st["K"]
        noise, seed = st["noise"], st["seed"]
        skey = f"{noise}_{seed}"

        if skey in done_keys:
            print(f"  [{idx+1}/{n_total}] {noise} s={seed}  (skip, already done)")
            continue

        elapsed = time.time() - t_global
        done_so_far = max(idx - len(done_keys), 1)
        eta = (n_total - idx - 1) * elapsed / done_so_far if elapsed > 10 else 0
        eta_str = f"  ETA {eta/60:.0f}min" if eta > 0 else ""
        print(f"  [{idx+1}/{n_total}] d={d} T={T} {noise} s={seed}{eta_str}")

        # Generate data (identical to exp1)
        W0_true = make_er_dag(d, 0.15, seed=seed)
        Wk_true = make_lag_matrices(d, K, 0.10, seed=seed)
        X = simulate_svar_linear(T, W0_true, Wk_true, noise, seed=seed)

        if X is None or not np.all(np.isfinite(X)) or X.std() < 1e-10:
            print(f"    Skipping (unstable simulation)")
            done_keys.add(skey)
            continue

        # Run DyCAST ONCE per (noise, seed)
        try:
            t0 = time.time()
            W0_est, Wk_est = run_dycast(
                X, d, K=K, k_dims=args.k_dims, epochs=args.epochs,
                lr=args.lr, seed=seed, device=device)
            met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
            dt = time.time() - t0

            # Replicate across all prior_accs (DyCAST is prior-agnostic)
            for acc in PRIOR_ACCS:
                row = dict(d=d, T=T, graph="ER", noise=noise, K=K,
                           prior_acc=acc, prior_mode="random", seed=seed,
                           nonlinear=False, missing_ratio=0.0,
                           method="DyCAST", tau=np.nan, time=dt, **met)
                all_rows.append(row)

            print(f"    DyCAST done in {dt:.1f}s  "
                  f"AUROC={met['auroc']:.3f}  F1={met['f1_opt']:.3f}  "
                  f"SHD={met['shd']}")
        except Exception as e:
            print(f"    DyCAST FAILED: {e}")
            traceback.print_exc()

        done_keys.add(skey)

        # Save after every run (each run is expensive)
        if all_rows:
            pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    # Final save
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df)} rows ({len(df)//len(PRIOR_ACCS)} unique runs "
              f"x {len(PRIOR_ACCS)} prior_accs) to {csv_path}")

        # Print summary table
        print("\n" + "=" * 70)
        print("DyCAST Results Summary")
        print("=" * 70)
        for metric in ["auroc", "f1_opt", "shd"]:
            if metric in df.columns:
                print(f"\n--- {metric.upper()} by noise x prior_acc ---")
                pivot = df.pivot_table(index="noise", columns="prior_acc",
                                       values=metric, aggfunc=["mean", "std"])
                print(pivot.to_string())
    else:
        print("No results generated.")


def _merge_results(dycast_dir):
    """Merge DyCAST CSVs (possibly from multiple GPU splits) into one,
    then optionally into exp1_noise/all_results.csv."""
    # Collect all dycast_noise.csv from dycast_dir and its subdirs
    csvs = []
    for root, dirs, files in os.walk(dycast_dir):
        for f in files:
            if f.startswith("dycast_") and f.endswith(".csv"):
                csvs.append(os.path.join(root, f))
    if not csvs:
        print(f"No DyCAST CSVs found under {dycast_dir}/")
        return

    dfs = [pd.read_csv(c) for c in csvs]
    df_dycast = pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset=["noise", "prior_acc", "seed"], keep="last")
    print(f"DyCAST results: {len(df_dycast)} rows from {len(csvs)} file(s)")

    # Save combined
    combined_csv = os.path.join(dycast_dir, "dycast_noise_combined.csv")
    df_dycast.to_csv(combined_csv, index=False)
    print(f"Combined → {combined_csv}")

    # Merge into main exp1 results if exists
    main_csv = "exp1_noise/all_results.csv"
    if os.path.exists(main_csv):
        df_main = pd.read_csv(main_csv)
        df_main = df_main[df_main["method"] != "DyCAST"]
        df_merged = pd.concat([df_main, df_dycast], ignore_index=True)
        df_merged.to_csv(main_csv, index=False)
        print(f"Merged into {main_csv}: {len(df_merged)} total rows")
    else:
        print(f"Main CSV not found: {main_csv}")
        print(f"DyCAST results saved standalone in {combined_csv}")


if __name__ == "__main__":
    main()
