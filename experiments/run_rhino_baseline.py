"""
run_rhino_standalone.py — Run RHINO baseline in an isolated conda environment.

This script is designed to run in a separate conda environment (e.g., 'rhino')
with causica v0.0.0 installed, independent of the main PRCD-MAP environment.

It generates the same synthetic/Lorenz-96 datasets used by exp1/exp2,
runs RHINO on each, computes metrics, and outputs CSV files that can be
merged into the main experiment results.

Usage (in the 'rhino' conda env):
  conda activate rhino
  python run_rhino_standalone.py --exp exp1 --sub prior
  python run_rhino_standalone.py --exp exp1 --sub noise
  python run_rhino_standalone.py --exp exp1 --sub graph
  python run_rhino_standalone.py --exp exp1 --sub nonlinear
  python run_rhino_standalone.py --exp exp1 --sub scale
  python run_rhino_standalone.py --exp exp2 --bench lorenz96
  python run_rhino_standalone.py --exp exp2 --bench netsim
  python run_rhino_standalone.py --exp exp2 --bench causaltime

After running, merge results:
  python run_rhino_standalone.py --merge
"""

import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd

# =====================================================================
# Minimal data generation (self-contained, no exp_utils dependency)
# =====================================================================

def set_seed(seed):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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


# Lorenz-96
def lorenz96_ground_truth(d):
    B = np.zeros((d, d), dtype=int)
    for i in range(d):
        B[(i - 2) % d, i] = 1
        B[(i - 1) % d, i] = 1
        B[(i + 1) % d, i] = 1
    return B


def generate_lorenz96(d=10, T=2000, F=10.0, dt=0.05, seed=0):
    from scipy.integrate import solve_ivp
    rng = np.random.RandomState(seed)

    def rhs(t, x):
        dxdt = np.zeros(d)
        for i in range(d):
            dxdt[i] = (x[(i+1)%d] - x[(i-2)%d]) * x[(i-1)%d] - x[i] + F
        return dxdt

    x0 = rng.randn(d) * 0.01 + F
    burn = 2000
    total_steps = T + burn
    t_span = (0, total_steps * dt)
    t_eval = np.arange(0, total_steps) * dt
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45", max_step=dt)
    X = sol.y.T[burn:]
    B_true = lorenz96_ground_truth(d)
    return standardize(X), B_true


def gen_prior_from_truth(B_true, acc, mode="random", seed=0):
    rng = np.random.RandomState(seed)
    d = B_true.shape[0]
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
    """Compute AUROC, F1, SHD from ground truth binary B and continuous W."""
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
    # Best F1 via threshold sweep
    best_f1 = 0.0
    best_thr = 0.0
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
    # SHD at best threshold
    y_pred_best = (y_score > best_thr).astype(int)
    shd = int(np.sum(y_pred_best != y_true))
    return dict(auroc=auroc, auprc=auprc, f1_opt=best_f1, shd=shd)


# =====================================================================
# RHINO runner (causica v0.0.0 API)
# =====================================================================

def run_rhino_v0(X, d, K, seed=0, max_epochs=200, batch_size=256):
    """
    Run RHINO via causica v0.0.0 API.
    Returns W0 (d, d) continuous adjacency matrix.
    """
    import torch
    set_seed(seed)

    from causica.models.deci.rhino import Rhino
    from causica.datasets.variables import Variables, Variable

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Variables
    variables = Variables([
        Variable(f"x{i}", True, "continuous",
                 lower=float("-inf"), upper=float("inf"))
        for i in range(d)
    ])

    X_std = standardize(X) if X.std() > 2.0 else X

    save_dir = f"/tmp/rhino_{os.getpid()}_{seed}"
    os.makedirs(save_dir, exist_ok=True)

    # Build TemporalDataset (causica v0.0.0 API)
    from causica.datasets.dataset import TemporalDataset
    train_data = X_std.astype(np.float32)
    train_mask = np.ones_like(train_data, dtype=np.float32)
    T_len = train_data.shape[0]
    dataset = TemporalDataset(
        train_data=train_data,
        train_mask=train_mask,
        transition_matrix=None,
        adjacency_data=None,
        intervention_data=None,
        counterfactual_data=None,
        variables=variables,
        train_segmentation=[(0, T_len - 1)],
    )

    # Create RHINO model (matching official config defaults)
    model = Rhino(
        model_id=f"rhino_{seed}",
        variables=variables,
        save_dir=save_dir,
        device=device,
        lag=K,
        allow_instantaneous=True,
        tau_gumbel=0.25,
        lambda_dag=100.0,
        lambda_sparse=1.0,
        lambda_prior=100000,
        base_distribution_type="conditional_spline",
        spline_bins=8,
        var_dist_A_mode="temporal_three",
        norm_layers=True,
        res_connection=True,
        prior_A_confidence=0.5,
        init_logits=[0, 0],
        conditional_spline_order="quadratic",
        additional_spline_flow=0,
        disable_diagonal_eval=True,
    )

    # Train — full config dict matching causica v0.0.0 defaults
    train_config = {
        "learning_rate": 1e-2,
        "likelihoods_learning_rate": 1e-3,
        "batch_size": batch_size,
        "standardize_data_mean": False,
        "standardize_data_std": False,
        "rho": 1.0,
        "safety_rho": 1e13,
        "alpha": 0.0,
        "safety_alpha": 1e13,
        "tol_dag": -1,
        "progress_rate": 0.65,
        "max_steps_auglag": max_epochs,
        "max_auglag_inner_epochs": 1000,
        "max_p_train_dropout": 0.0,
        "reconstruction_loss_factor": 1.0,
        "anneal_entropy": "noanneal",
    }
    model.run_train(dataset=dataset, train_config_dict=train_config)

    # Extract adjacency
    # RHINO returns shape (1, 1, (K+1)*d, d) or (samples, 1, (K+1)*d, d)
    adj = model.get_adj_matrix(samples=100, do_round=False)
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    # Squeeze batch/sample dims, take mean if needed
    while adj.ndim > 2:
        if adj.shape[0] > 1:
            adj = adj.mean(axis=0)
        else:
            adj = adj.squeeze(0)
    # adj is now ((K+1)*d, d): first d rows = instantaneous, rest = lag
    W0 = adj[:d, :d].copy()
    np.fill_diagonal(W0, 0.0)
    return W0


# =====================================================================
# Experiment runners
# =====================================================================

def run_exp1_sub(sub, output_dir="rhino_results"):
    """Run RHINO on exp1 sub-experiment settings."""
    os.makedirs(output_dir, exist_ok=True)

    if sub == "prior":
        dims, Ts = [20], [500]
        accs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        modes = ["random", "adversarial"]
        seeds = [0, 1, 2]
        noises = ["gaussian"]
    elif sub == "noise":
        dims, Ts = [20], [500]
        accs = [0.0, 0.4, 0.6, 0.9]
        modes = ["random"]
        seeds = [0, 1, 2]
        noises = ["gaussian", "laplace", "student_t", "heteroscedastic"]
    elif sub == "graph":
        dims, Ts = [20], [500]
        accs = [0.0, 0.4, 0.6, 0.9]
        modes = ["random"]
        seeds = [0, 1, 2]
        noises = ["gaussian"]
    elif sub == "nonlinear":
        dims, Ts = [10, 20], [500]
        accs = [0.2, 0.6, 1.0]
        modes = ["random"]
        seeds = [0, 1, 2]
        noises = ["gaussian"]
    elif sub == "scale":
        dims, Ts = [10, 20, 50], [500]
        accs = [0.6]
        modes = ["random"]
        seeds = [0, 1, 2]
        noises = ["gaussian"]
    else:
        raise ValueError(f"Unknown sub: {sub}")

    K = 1
    rows = []
    total = len(dims) * len(Ts) * len(noises) * len(accs) * len(modes) * len(seeds)
    done = 0

    for d in dims:
        for T in Ts:
            for noise in noises:
                for acc in accs:
                    for mode in modes:
                        for seed in seeds:
                            done += 1
                            print(f"  [{done}/{total}] d={d} T={T} {noise} "
                                  f"acc={acc} {mode} s={seed}", end="", flush=True)

                            W0_true = make_er_dag(d, 0.15, seed=seed)
                            Wk_true = make_lag_matrices(d, K, seed=seed)
                            X = simulate_svar_linear(T, W0_true, Wk_true, noise, seed=seed)
                            if X is None:
                                print(" SKIP")
                                continue

                            P_prior = gen_prior(W0_true, Wk_true, acc, mode, seed=seed+999)
                            B_true = (np.abs(W0_true) > 1e-10).astype(int)
                            for Wk in Wk_true:
                                B_true = np.maximum(B_true, (np.abs(Wk) > 1e-10).astype(int))

                            try:
                                t0 = time.time()
                                W0_est = run_rhino_v0(X, d, K, seed=seed)
                                elapsed = time.time() - t0
                                met = compute_metrics(B_true, W0_est)
                                rows.append(dict(
                                    d=d, T=T, graph="ER", noise=noise, K=K,
                                    prior_acc=acc, prior_mode=mode, seed=seed,
                                    nonlinear=False,
                                    method="RHINO", tau=np.nan,
                                    time=elapsed, **met,
                                ))
                                print(f" auroc={met['auroc']:.3f} {elapsed:.1f}s")
                            except Exception as e:
                                print(f" FAILED: {e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"rhino_exp1_{sub}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {csv_path}")
    return df


def run_exp2_bench(bench, output_dir="rhino_results"):
    """Run RHINO on exp2 benchmark settings."""
    os.makedirs(output_dir, exist_ok=True)
    K = 1
    seeds = [0, 1, 2]
    accs = [0.3, 0.6, 0.9]
    rows = []

    if bench == "lorenz96":
        settings = [(10, 500), (20, 500), (20, 200), (40, 500)]
        for lorenz_d, lorenz_T in settings:
            for seed in seeds:
                print(f"  Lorenz96 d={lorenz_d} T={lorenz_T} s={seed}", end="", flush=True)
                X, B_true = generate_lorenz96(d=lorenz_d, T=lorenz_T, seed=seed)
                try:
                    t0 = time.time()
                    W0_est = run_rhino_v0(X, lorenz_d, K, seed=seed)
                    elapsed = time.time() - t0
                    met = compute_metrics(B_true, W0_est)
                    rows.append(dict(
                        bench=f"Lorenz96_d{lorenz_d}_T{lorenz_T}",
                        method="RHINO", seed=seed,
                        prior_acc=np.nan, prior_mode="none",
                        tau=np.nan, time=elapsed, **met,
                    ))
                    print(f" auroc={met['auroc']:.3f} {elapsed:.1f}s")
                except Exception as e:
                    print(f" FAILED: {e}")

    elif bench == "netsim":
        netsim_dir = "data/netsim"
        for sim_id in [3, 4, 15]:
            from scipy.io import loadmat
            mat_path = os.path.join(netsim_dir, f"sim{sim_id}.mat")
            if not os.path.exists(mat_path):
                print(f"  sim{sim_id}.mat not found, skipping")
                continue
            data = loadmat(mat_path)
            X = standardize(data.get("ts", data.get("Ysim")).astype(np.float64))
            net = data.get("net", data.get("Adj"))
            # Squeeze extra dimensions (some .mat files have 3D arrays)
            if net.ndim == 3:
                net = net[:, :, 0]
            B_true = (np.abs(net) > 0.5).astype(int)
            # Ensure square before fill_diagonal
            if B_true.shape[0] != B_true.shape[1]:
                print(f"  sim{sim_id} B_true shape {B_true.shape} not square, skipping")
                continue
            np.fill_diagonal(B_true, 0)
            d = X.shape[1]
            for seed in seeds:
                print(f"  Netsim sim{sim_id} d={d} s={seed}", end="", flush=True)
                try:
                    t0 = time.time()
                    W0_est = run_rhino_v0(X, d, K, seed=seed)
                    elapsed = time.time() - t0
                    met = compute_metrics(B_true, W0_est)
                    rows.append(dict(
                        bench=f"Netsim_sim{sim_id}_d{d}",
                        method="RHINO", seed=seed,
                        prior_acc=np.nan, prior_mode="none",
                        tau=np.nan, time=elapsed, **met,
                    ))
                    print(f" auroc={met['auroc']:.3f} {elapsed:.1f}s")
                except Exception as e:
                    print(f" FAILED: {e}")

    elif bench == "causaltime":
        ct_dir = "data/causaltime"
        name_map = {"AQI": "pm25", "Traffic": "traffic", "Medical": "medical"}
        for ds_name, folder in name_map.items():
            ds_dir = os.path.join(ct_dir, folder)
            npy_graph = os.path.join(ds_dir, "graph.npy")
            npy_data = os.path.join(ds_dir, "gen_data.npy")
            if not os.path.exists(npy_graph):
                print(f"  {ds_name} not found, skipping")
                continue
            B_true = (np.abs(np.load(npy_graph)) > 0.5).astype(int)
            np.fill_diagonal(B_true, 0)
            d = B_true.shape[0]
            raw = np.load(npy_data)
            X = standardize(raw[0, :, :d].astype(np.float64) if raw.ndim == 3 else raw[:, :d])
            for seed in seeds:
                print(f"  CausalTime {ds_name} d={d} s={seed}", end="", flush=True)
                try:
                    t0 = time.time()
                    W0_est = run_rhino_v0(X, d, K, seed=seed)
                    elapsed = time.time() - t0
                    met = compute_metrics(B_true, W0_est)
                    rows.append(dict(
                        bench=f"CausalTime_{ds_name}_d{d}",
                        method="RHINO", seed=seed,
                        prior_acc=np.nan, prior_mode="none",
                        tau=np.nan, time=elapsed, **met,
                    ))
                    print(f" auroc={met['auroc']:.3f} {elapsed:.1f}s")
                except Exception as e:
                    print(f" FAILED: {e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"rhino_exp2_{bench}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Saved {len(df)} rows -> {csv_path}")
    return df


# =====================================================================
# Merge RHINO results into main experiment CSVs
# =====================================================================

def merge_results(rhino_dir="rhino_results"):
    """Merge RHINO CSVs into main experiment result files."""
    print(">>> Merging RHINO results into main experiment CSVs...")

    # Exp1 subs
    for sub in ["prior", "noise", "graph", "nonlinear", "scale"]:
        rhino_csv = os.path.join(rhino_dir, f"rhino_exp1_{sub}.csv")
        main_csv = os.path.join(f"exp1_{sub}", "all_results.csv")
        if os.path.exists(rhino_csv) and os.path.exists(main_csv):
            df_rhino = pd.read_csv(rhino_csv)
            df_main = pd.read_csv(main_csv)
            # Remove old RHINO rows if any
            df_main = df_main[df_main["method"] != "RHINO"]
            df_merged = pd.concat([df_main, df_rhino], ignore_index=True)
            df_merged.to_csv(main_csv, index=False)
            print(f"  {main_csv}: +{len(df_rhino)} RHINO rows "
                  f"(total {len(df_merged)})")
        else:
            if not os.path.exists(rhino_csv):
                print(f"  {rhino_csv} not found, skipping")

    # Exp2 benches
    for bench in ["lorenz96", "netsim", "causaltime"]:
        rhino_csv = os.path.join(rhino_dir, f"rhino_exp2_{bench}.csv")
        main_csv = os.path.join("exp2_results", f"exp2_{bench}_results.csv")
        # Also try the combined file
        if not os.path.exists(main_csv):
            main_csv = os.path.join("exp2_results",
                                     "exp2_lorenz96_results.csv" if bench == "lorenz96"
                                     else f"exp2_{bench}_results.csv")
        if os.path.exists(rhino_csv) and os.path.exists(main_csv):
            df_rhino = pd.read_csv(rhino_csv)
            df_main = pd.read_csv(main_csv)
            df_main = df_main[df_main["method"] != "RHINO"]
            df_merged = pd.concat([df_main, df_rhino], ignore_index=True)
            df_merged.to_csv(main_csv, index=False)
            print(f"  {main_csv}: +{len(df_rhino)} RHINO rows")
        else:
            if not os.path.exists(rhino_csv):
                print(f"  {rhino_csv} not found, skipping")

    print(">>> Merge complete.")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="RHINO standalone runner")
    parser.add_argument("--exp", type=str, choices=["exp1", "exp2"],
                        help="Which experiment")
    parser.add_argument("--sub", type=str, default=None,
                        choices=["noise", "sample_size", "nonlinear", "scale"],
                        help="Exp1 sub-experiment")
    parser.add_argument("--bench", type=str, default=None,
                        choices=["causaltime"],
                        help="Exp2 benchmark")
    parser.add_argument("--output", type=str, default="rhino_results")
    parser.add_argument("--merge", action="store_true",
                        help="Merge RHINO CSVs into main results")
    parser.add_argument("--all", action="store_true",
                        help="Run all exp1 subs + exp2 benches")
    args = parser.parse_args()

    if args.merge:
        merge_results(args.output)
        return

    if args.all:
        for sub in ["prior", "noise", "graph", "nonlinear", "scale"]:
            print(f"\n{'='*60}")
            print(f"  RHINO — exp1 --sub {sub}")
            print(f"{'='*60}")
            run_exp1_sub(sub, args.output)
        for bench in ["lorenz96", "netsim", "causaltime"]:
            print(f"\n{'='*60}")
            print(f"  RHINO — exp2 --bench {bench}")
            print(f"{'='*60}")
            run_exp2_bench(bench, args.output)
        print("\n>>> All RHINO runs complete. Run with --merge to combine.")
        return

    if args.exp == "exp1":
        if not args.sub:
            parser.error("--sub required for exp1")
        run_exp1_sub(args.sub, args.output)
    elif args.exp == "exp2":
        if not args.bench:
            parser.error("--bench required for exp2")
        run_exp2_bench(args.bench, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
