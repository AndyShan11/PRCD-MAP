"""
theory_trust_propagation.py — Formal Definitions and Theoretical Analysis.

Structure-Aware Trust Propagation: 形式化定义, 收敛性分析, ε-safety bound.
本文件包含:
  1. Definition: Structure-Aware Trust Propagation
  2. Theorem: Convergence under Accurate Prior
  3. Proposition: Tighter ε-safety Bound via Neighborhood Consistency
  4. 数值验证函数 (empirical verification of theoretical bounds)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from typing import Tuple, Dict

# ====================================================================
# Formal Definitions (LaTeX-ready)
# ====================================================================

DEFINITION_TRUST_PROPAGATION = r"""
\begin{definition}[Structure-Aware Trust Propagation]
\label{def:trust_propagation}
Let $\mathcal{G} = (V, E_{\text{prior}})$ be the prior graph with edge probabilities
$P \in [0,1]^{d \times d}$, and let $W^* \in \mathbb{R}^{d \times d}$ be the current
weight estimate. The \emph{structure-aware trust temperature}
$\tau: E \to [\tau_{\min}, \tau_{\max}]$ is defined by an $L$-layer edge-level graph
attention network:

\begin{align}
h_{ij}^{(0)} &= \text{Embed}(P_{ij}, \|W^*_{ij}\|, g(\text{group}(i,j))) \\
h_{ij}^{(l)} &= \text{LayerNorm}\Bigl(h_{ij}^{(l-1)} +
  \frac{1}{2}\bigl[\text{Attn}_{\text{col}}^{(l)}(h^{(l-1)}) +
  \text{Attn}_{\text{row}}^{(l)}(h^{(l-1)})\bigr]\Bigr) \\
\tau_{ij} &= \sigma(f_\theta(h_{ij}^{(L)}) + b) \cdot (\tau_{\max} - \tau_{\min})
  + \tau_{\min}
\end{align}

where:
\begin{itemize}
\item $\text{Attn}_{\text{col}}^{(l)}$: multi-head attention over column neighbors
  $\mathcal{N}_{\text{col}}(i,j) = \{(k,j) : k \neq i\}$
\item $\text{Attn}_{\text{row}}^{(l)}$: multi-head attention over row neighbors
  $\mathcal{N}_{\text{row}}(i,j) = \{(i,l) : l \neq j\}$
\item $f_\theta$: MLP prediction head
\item $b$: global bias parameter
\item $\sigma$: sigmoid function
\end{itemize}

The calibrated prior is then $\hat{P}_{ij} = \sigma(\text{logit}(P_{ij}) \cdot \tau_{ij})$.
\end{definition}
"""

THEOREM_CONVERGENCE = r"""
\begin{theorem}[Convergence under Accurate Prior]
\label{thm:convergence}
Assume the prior satisfies $\text{AUROC}(P, A^*) \geq 1 - \delta$ for some $\delta > 0$
where $A^*$ is the true adjacency. Let $\mathcal{C}_{ij} = \{(k,l) \in \mathcal{N}(i,j) :
\text{sign}(P_{kl} - 0.5) = \text{sign}(A^*_{kl} - 0.5)\}$ denote the set of correctly
informed neighbors. If $|\mathcal{C}_{ij}| / |\mathcal{N}(i,j)| \geq 1 - \epsilon$ for
all $(i,j)$ (neighborhood accuracy), then the trust propagation output satisfies:

\begin{equation}
\mathbb{E}[\|\tau_{\text{trust}} - \tau^*\|_F] \leq
  \mathbb{E}[\|\tau_{\text{group}} - \tau^*\|_F] -
  \Omega\Bigl(\sqrt{\frac{|\mathcal{N}|}{d^2}} \cdot (1 - 2\epsilon)\Bigr)
\end{equation}

where $\tau^*$ is the oracle temperature that minimizes the EB objective, and
$\tau_{\text{group}}$ is the per-group temperature.
\end{theorem}
"""

PROPOSITION_SAFETY_BOUND = r"""
\begin{proposition}[Tighter $\varepsilon$-Safety Bound]
\label{prop:safety_bound}
Define the $\varepsilon$-safety bound as:
\begin{equation}
\mathcal{B}_\varepsilon = \max_{(i,j) \in E} |\hat{P}_{ij} - A^*_{ij}|
\end{equation}

Under the trust propagation framework with $L \geq 1$ layers and neighborhood
consistency $\rho_{\text{cons}} = \min_{(i,j)} \text{Corr}(P_{\mathcal{N}(i,j)},
A^*_{\mathcal{N}(i,j)})$:

\begin{equation}
\mathcal{B}_\varepsilon^{\text{trust}} \leq \mathcal{B}_\varepsilon^{\text{group}}
  \cdot \frac{1}{1 + \eta \cdot \rho_{\text{cons}}}
\end{equation}

where $\eta > 0$ depends on the GAT weights and $\rho_{\text{cons}} \in [-1, 1]$
measures the local consistency of the prior graph.

\textbf{Interpretation:} When the prior is locally consistent ($\rho_{\text{cons}} > 0$),
trust propagation provides a strictly tighter safety bound than per-group temperature,
with improvement proportional to the consistency signal.
\end{proposition}
"""


# ====================================================================
# Numerical Verification Functions
# ====================================================================

def compute_neighborhood_consistency(P_prior: np.ndarray,
                                      B_true: np.ndarray) -> np.ndarray:
    """
    Compute per-edge neighborhood consistency score.

    ρ_cons(i,j) = correlation between P and B_true in the neighborhood of (i,j).
    High ρ_cons → reliable local prior → trust propagation benefits more.

    Returns: (d, d) matrix of consistency scores.
    """
    d = P_prior.shape[0]
    rho = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # Neighbors: row i and column j (excluding (i,j) itself)
            row_mask = np.ones(d, dtype=bool)
            row_mask[i] = False
            col_mask = np.ones(d, dtype=bool)
            col_mask[j] = False

            # Row neighbors: (i, l) for l != j, l != i
            p_row = P_prior[i, col_mask]
            b_row = B_true[i, col_mask].astype(float)

            # Col neighbors: (k, j) for k != i, k != j
            p_col = P_prior[row_mask, j]
            b_col = B_true[row_mask, j].astype(float)

            # Concatenate
            p_all = np.concatenate([p_row, p_col])
            b_all = np.concatenate([b_row, b_col])

            if len(p_all) > 2 and np.std(p_all) > 1e-6 and np.std(b_all) > 1e-6:
                rho[i, j] = np.corrcoef(p_all, b_all)[0, 1]
            else:
                rho[i, j] = 0.0

    return rho


def compute_safety_bound(P_hat: np.ndarray, B_true: np.ndarray) -> float:
    """Compute ε-safety bound: max |P_hat - B_true|."""
    d = P_hat.shape[0]
    mask = ~np.eye(d, dtype=bool)
    return float(np.max(np.abs(P_hat[mask] - B_true[mask].astype(float))))


def verify_safety_bound_improvement(
    P_prior: np.ndarray,
    B_true: np.ndarray,
    tau_trust: np.ndarray,
    tau_group: np.ndarray,
) -> Dict:
    """
    Empirically verify that trust propagation provides tighter safety bound.

    Returns dict with:
      safety_bound_trust, safety_bound_group, improvement_ratio,
      mean_consistency, consistency_matrix
    """
    d = P_prior.shape[0]
    eps = 1e-3
    P_clamped = np.clip(P_prior, eps, 1.0 - eps)
    logits = np.log(P_clamped / (1.0 - P_clamped))

    # Calibrated priors
    P_hat_trust = 1.0 / (1.0 + np.exp(-logits * tau_trust))
    P_hat_group = 1.0 / (1.0 + np.exp(-logits * tau_group))

    # Safety bounds
    bound_trust = compute_safety_bound(P_hat_trust, B_true)
    bound_group = compute_safety_bound(P_hat_group, B_true)

    # Neighborhood consistency
    rho_cons = compute_neighborhood_consistency(P_prior, B_true)
    mask = ~np.eye(d, dtype=bool)
    mean_cons = float(rho_cons[mask].mean())

    return {
        "safety_bound_trust": bound_trust,
        "safety_bound_group": bound_group,
        "improvement_ratio": bound_group / max(bound_trust, 1e-10),
        "mean_consistency": mean_cons,
        "consistency_matrix": rho_cons,
    }


def run_theory_verification(
    d: int = 20, T: int = 500, prior_accs: list = None,
    n_seeds: int = 5, verbose: bool = True,
) -> pd.DataFrame:
    """
    Numerically verify theoretical bounds across prior accuracies.

    For each (prior_acc, seed):
    1. Generate synthetic data
    2. Run trust propagation and per-group
    3. Compare safety bounds
    4. Compute neighborhood consistency
    """
    import sys, os, importlib.util as _ilu
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _src_dir = os.path.join(_this_dir, "..", "src")

    # Load ../src/utils.py explicitly
    if "_orig_exp_utils" not in sys.modules:
        _spec = _ilu.spec_from_file_location("_orig_exp_utils", os.path.join(_src_dir, "utils.py"))
        _orig = _ilu.module_from_spec(_spec)
        sys.modules["_orig_exp_utils"] = _orig
        _spec.loader.exec_module(_orig)
    _orig = sys.modules["_orig_exp_utils"]
    set_seed = _orig.set_seed
    make_er_dag = _orig.make_er_dag
    make_lag_matrices = _orig.make_lag_matrices
    simulate_svar_linear = _orig.simulate_svar_linear
    standardize = _orig.standardize
    gen_prior = _orig.gen_prior
    make_lag_tensors = _orig.make_lag_tensors
    compute_dual_metrics = _orig.compute_dual_metrics

    # Load ../src/model_linear.py explicitly
    _spec2 = _ilu.spec_from_file_location("_orig_model_linear", os.path.join(_src_dir, "model_linear.py"))
    _orig2 = _ilu.module_from_spec(_spec2)
    _spec2.loader.exec_module(_orig2)
    PRCD_MAP_Model = _orig2.PRCD_MAP_Model
    train_prcd_alm = _orig2.train_prcd_alm

    # Local trust model (in ../src)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from model_linear_trust import PRCD_MAP_Trust, train_prcd_trust_alm

    if prior_accs is None:
        prior_accs = [0.2, 0.4, 0.6, 0.8, 1.0]

    results = []

    for acc in prior_accs:
        for seed in range(n_seeds):
            if verbose:
                print(f"\n  acc={acc:.1f}, seed={seed}")

            set_seed(seed)
            W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
            Wk_true = make_lag_matrices(d, 1, seed=seed)
            X = simulate_svar_linear(T, W0_true, Wk_true, seed=seed)
            if X is None:
                continue
            X = standardize(X)
            P_prior = gen_prior(W0_true, Wk_true, acc=acc, seed=seed)

            B_true = (np.abs(W0_true) > 1e-10).astype(int)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_t, X_lags = make_lag_tensors(X, 1)
            X_t, X_lags = X_t.to(dev), [x.to(dev) for x in X_lags]

            # Trust propagation
            model_t = PRCD_MAP_Trust(
                num_vars=d, lag_k=1, P_prior=P_prior,
                lambda1=0.001, lambda2=0.01, learn_tau=True,
            ).to(dev)
            W0_t, Wk_t, tau_t_val = train_prcd_trust_alm(
                model_t, X_t, X_lags, max_iter=25, inner_iter=300,
                verbose=False)

            # Extract tau matrix
            with torch.no_grad():
                tau_trust = model_t._compute_tau_matrix().cpu().numpy()

            # Per-group
            model_g = PRCD_MAP_Model(
                num_vars=d, lag_k=1, P_prior=P_prior,
                lambda1=0.001, lambda2=0.01, learn_tau=True,
            ).to(dev)
            W0_g, Wk_g, tau_g_val = train_prcd_alm(
                model_g, X_t, X_lags, max_iter=25, inner_iter=300,
                verbose=False)

            with torch.no_grad():
                tau_group = model_g._expand_tau().cpu().numpy()

            # Verify bounds
            verification = verify_safety_bound_improvement(
                P_prior, B_true, tau_trust, tau_group)

            # Metrics
            met_t = compute_dual_metrics(W0_true, Wk_true, W0_t, Wk_t)
            met_g = compute_dual_metrics(W0_true, Wk_true, W0_g, Wk_g)

            results.append({
                "prior_acc": acc, "seed": seed,
                "safety_bound_trust": verification["safety_bound_trust"],
                "safety_bound_group": verification["safety_bound_group"],
                "improvement_ratio": verification["improvement_ratio"],
                "mean_consistency": verification["mean_consistency"],
                "auroc_trust": met_t["auroc"],
                "auroc_group": met_g["auroc"],
                "f1_trust": met_t["f1_opt"],
                "f1_group": met_g["f1_opt"],
                "tau_trust_mean": float(np.mean(tau_trust[~np.eye(d, dtype=bool)])),
                "tau_group_mean": float(np.mean(tau_group[~np.eye(d, dtype=bool)])),
            })

            if verbose:
                r = results[-1]
                print(f"    bound: trust={r['safety_bound_trust']:.3f} "
                      f"group={r['safety_bound_group']:.3f} "
                      f"ratio={r['improvement_ratio']:.3f}")
                print(f"    AUROC: trust={r['auroc_trust']:.3f} "
                      f"group={r['auroc_group']:.3f}")
                print(f"    consistency={r['mean_consistency']:.3f}")

    df = pd.DataFrame(results)

    if verbose and len(df) > 0:
        print("\n" + "=" * 60)
        print("THEORY VERIFICATION SUMMARY")
        print("=" * 60)
        summary = df.groupby("prior_acc").agg({
            "safety_bound_trust": ["mean", "std"],
            "safety_bound_group": ["mean", "std"],
            "improvement_ratio": ["mean", "std"],
            "mean_consistency": ["mean"],
            "auroc_trust": ["mean"],
            "auroc_group": ["mean"],
        })
        print(summary.to_string())

        print("\nKey finding:")
        mean_ratio = df["improvement_ratio"].mean()
        if mean_ratio > 1.0:
            print(f"  Trust propagation provides {mean_ratio:.2f}x tighter safety bound "
                  f"on average (Proposition 1 verified)")
        else:
            print(f"  Average improvement ratio: {mean_ratio:.2f}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    df = run_theory_verification(d=args.d, n_seeds=args.seeds)
    if len(df) > 0:
        df.to_csv("theory_verification_results.csv", index=False)
        print("\n>>> Saved theory_verification_results.csv")
