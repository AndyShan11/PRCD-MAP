"""Generate 5 data-driven priors per CausalTime dataset (AQI/Traffic/Medical).
Variable-anonymous datasets — LLM domain knowledge inapplicable. We use 5 distinct
classical statistical measures as priors:
  style 0: |Pearson|
  style 1: |Spearman|
  style 2: |Pearson| top-20% (sparse)
  style 3: Pearson^2 (R^2-style)
  style 4: |partial correlation| (inverse covariance)
"""
import os, sys, json
import numpy as np
from scipy import stats
from pathlib import Path

sys.path.insert(0, './src')
sys.path.insert(0, './experiments')
from utils_trust import load_causaltime

CACHE = Path('./experiments/llm_prior_cache')
CACHE.mkdir(parents=True, exist_ok=True)

def s_pearson(X):
    R = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
    P = np.abs(R); np.fill_diagonal(P, 0.0)
    return np.clip(P, 0.01, 0.99)

def s_spearman(X):
    rho, _ = stats.spearmanr(X)
    if np.ndim(rho) == 0:
        rho = np.array([[1.0, rho], [rho, 1.0]])
    rho = np.nan_to_num(rho, nan=0.0)
    P = np.abs(rho); np.fill_diagonal(P, 0.0)
    return np.clip(P, 0.01, 0.99)

def s_sparse(X, frac=0.20):
    R = np.abs(np.nan_to_num(np.corrcoef(X.T), nan=0.0))
    np.fill_diagonal(R, 0.0)
    d = R.shape[0]
    n_keep = max(1, int(frac * d * (d - 1)))
    flat = R[~np.eye(d, dtype=bool)]
    thr = np.partition(flat, -n_keep)[-n_keep] if n_keep < flat.size else 0
    P = np.where(R >= thr, R, 0.5)
    np.fill_diagonal(P, 0.0)
    return np.clip(P, 0.01, 0.99)

def s_squared(X):
    R = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
    P = R ** 2; np.fill_diagonal(P, 0.0)
    return np.clip(P, 0.01, 0.99)

def s_partial(X):
    cov = np.cov(X.T) + 1e-3 * np.eye(X.shape[1])
    prec = np.linalg.inv(cov)
    d = prec.shape[0]
    P = np.zeros_like(prec)
    for i in range(d):
        for j in range(d):
            if i != j:
                denom = np.sqrt(prec[i, i] * prec[j, j])
                if denom > 1e-10:
                    P[i, j] = -prec[i, j] / denom
    P = np.abs(P); np.fill_diagonal(P, 0.0)
    return np.clip(P, 0.01, 0.99)

STYLES = [
    ("style0", "abs_pearson",         s_pearson),
    ("style1", "abs_spearman",        s_spearman),
    ("style2", "sparse_pearson_top20", s_sparse),
    ("style3", "pearson_squared",     s_squared),
    ("style4", "partial_correlation", s_partial),
]

manifest = {}
for ds in ['AQI', 'Traffic', 'Medical']:
    print(f"=== {ds} ===")
    X, B = load_causaltime('./data/causaltime', ds, n_samples=10)
    d = X.shape[1]
    print(f"  X shape: {X.shape}, B shape: {B.shape}")
    manifest[ds] = {'d': int(d), 'styles': {}}
    for tag, name, fn in STYLES:
        P = fn(X.astype(float))
        assert P.shape == (d, d), f"shape mismatch: {P.shape} vs ({d},{d})"
        out = CACHE / f"{ds}_prior_{tag}.npy"
        np.save(out, P)
        m = {
            "method": name,
            "max":  float(P.max()),
            "min":  float(P[P > 0.011].min()) if (P > 0.011).any() else 0,
            "mean": float(P.mean()),
            "n_strong": int((P > 0.6).sum()),
        }
        manifest[ds]['styles'][tag] = m
        print(f"  {tag} {name:25s}  max={m['max']:.3f}  mean={m['mean']:.3f}  strong={m['n_strong']}")

with open(CACHE / '_corr_prior_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved manifest to {CACHE/'_corr_prior_manifest.json'}")
print("All 15 priors (3 datasets × 5 styles) ready.")
