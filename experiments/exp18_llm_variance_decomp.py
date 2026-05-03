"""
=============================================================================
Experiment 18 — LLM-prior variance decomposition
=============================================================================
Reviewer concern (Q10): the LLM-prior pipeline reports aggregate gain, but
the variance attributable to LLM model (GPT-4o, Claude, Gemini) versus
prompt style (conservative-textbook, mechanism-first, literature-anchored,
permissive, adversarial) is not decomposed.

Mapping (from llm_prior_cache/_electricity_llm_manifest.json and analogous
manifests for AQI / Medical / Traffic):
  style0 → GPT-4o, conservative textbook
  style1 → Claude Sonnet, mechanism-first reasoning
  style2 → Gemini 1.5 Pro, literature-anchored
  style3 → GPT-4o, permissive (dense)
  style4 → Claude Sonnet, adversarial / reverse-checking

This script:
  (1) Re-runs PRCD-MAP (trust) on each (dataset, style) cell with the same
      seeds used in Table 2, OR re-uses exp9 outputs if present.
  (2) Decomposes AUROC variance via two-way ANOVA:
          AUROC = LLM_model + prompt_style + LLM_model × style + ε
  (3) Reports model-attributable std, style-attributable std, and per-cell
      AUROC ± std for the appendix table.

Datasets: AQI, Medical, Traffic (CausalTime; T=400, ground truth available).

Usage:
  python exp18_llm_variance_decomp.py --datasets AQI Medical Traffic \
        --seeds 0 1 2 3 4
=============================================================================
"""
import os, sys, time, argparse, traceback, json
import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_DL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data_loaders")
if os.path.isdir(_DL) and _DL not in sys.path:
    sys.path.insert(0, _DL)

from utils import (set_seed, standardize, run_prcd_map, ensure_dir, fmt_time,
                   compute_dual_metrics)


# Mapping from style index → (LLM model, prompt style)
STYLE_MAP = {
    0: ("GPT-4o",   "conservative_textbook"),
    1: ("Claude",   "mechanism_first"),
    2: ("Gemini",   "literature_anchored"),
    3: ("GPT-4o",   "permissive"),
    4: ("Claude",   "adversarial"),
}


def load_dataset_with_truth(name, t_target=400):
    """Load CausalTime dataset; returns (X, B_true, d)."""
    try:
        from utils import load_causaltime  # may exist in some forks
        X, B_true = load_causaltime(name, n_samples=t_target)
        if X is not None:
            d = X.shape[1]
            return X, B_true, d
    except Exception:
        pass
    # Fallback: try data_loaders/load_causaltime
    try:
        from load_causaltime import load_causaltime
        X, B_true = load_causaltime(name, n_samples=t_target)
        d = X.shape[1]
        return X, B_true, d
    except Exception:
        traceback.print_exc()
        return None, None, None


def load_prior_for_style(dataset, style_idx, cache_dir="llm_prior_cache"):
    path = os.path.join(cache_dir, f"{dataset}_prior_style{style_idx}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["AQI", "Medical", "Traffic"])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--styles", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--cache-dir", type=str, default="llm_prior_cache")
    parser.add_argument("--T", type=int, default=400)
    args = parser.parse_args()

    output_dir = "exp18_llm_variance_decomp"
    ensure_dir(output_dir)

    results = []
    t_global = time.time()

    for ds in args.datasets:
        X, B_true, d = load_dataset_with_truth(ds, t_target=args.T)
        if X is None:
            print(f"[SKIP] cannot load {ds}")
            continue
        X = standardize(X)
        K = 1
        # Construct W0/Wk targets from B_true (assume B_true is combined)
        # For CausalTime evaluation, we report combined-AUROC against B_true
        # by treating B_true as the combined adjacency.
        for style_idx in args.styles:
            P_prior = load_prior_for_style(ds, style_idx, args.cache_dir)
            if P_prior is None:
                print(f"[SKIP] missing prior {ds}/style{style_idx}")
                continue
            llm_model, prompt_style = STYLE_MAP.get(style_idx, ("?", "?"))

            for seed in args.seeds:
                t0 = time.time()
                try:
                    set_seed(seed)
                    W0, Wk, tau = run_prcd_map(
                        X, P_prior, d, K,
                        lambda1=0.001, lambda2=0.01,
                        learn_tau=True,
                        max_iter=35, inner_iter=400, lr=8e-3, seed=seed,
                        score_calibration=True)
                    # Build a "combined" estimate matching B_true shape.
                    # If B_true is W0-only (cross-sectional), use W0.
                    # If B_true is (d, d) assume W0+sum(Wk).
                    W0a = np.abs(W0)
                    if W0a.shape == B_true.shape:
                        W_combined = W0a + sum(np.abs(w) for w in Wk)
                    else:
                        W_combined = W0a
                    # Generic AUROC against B_true:
                    from sklearn.metrics import roc_auc_score
                    y = (np.abs(B_true) > 1e-8).astype(int).ravel()
                    s = W_combined.ravel()
                    if y.sum() > 0 and y.sum() < y.size:
                        auroc = float(roc_auc_score(y, s))
                    else:
                        auroc = float("nan")

                    results.append({
                        "dataset": ds, "style": style_idx,
                        "llm_model": llm_model,
                        "prompt_style": prompt_style,
                        "seed": seed, "auroc": auroc,
                        "tau_mean": float(tau) if tau is not None else np.nan,
                    })
                    print(f"  {ds}/style{style_idx}/{llm_model}/{prompt_style} "
                          f"seed={seed}: AUROC={auroc:.3f} "
                          f"[{fmt_time(time.time()-t0)}]")
                except Exception:
                    traceback.print_exc()

                if results:
                    pd.DataFrame(results).to_csv(
                        os.path.join(output_dir, "_intermediate.csv"),
                        index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "exp18_results.csv"), index=False)

    # Variance decomposition
    print("\n" + "=" * 60)
    print("Variance decomposition (two-way ANOVA per dataset)")
    print("=" * 60)
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds].dropna(subset=["auroc"]).copy()
            if len(sub) < 6:
                continue
            print(f"\n{ds}:")
            print(f"  per-cell mean AUROC ± std (across seeds):")
            for s in sorted(sub["style"].unique()):
                ss = sub[sub["style"] == s]
                print(f"    style{s} ({ss['llm_model'].iloc[0]:6s}, "
                      f"{ss['prompt_style'].iloc[0]:25s}): "
                      f"{ss['auroc'].mean():.3f}±{ss['auroc'].std():.3f}")
            # ANOVA: AUROC ~ llm_model + prompt_style
            try:
                model = ols("auroc ~ C(llm_model) + C(prompt_style)",
                            data=sub).fit()
                anova = sm.stats.anova_lm(model, typ=2)
                print(f"  ANOVA:")
                print(anova.to_string())
            except Exception:
                # Variance breakdown without statsmodels
                m_std = sub.groupby("llm_model")["auroc"].mean().std()
                p_std = sub.groupby("prompt_style")["auroc"].mean().std()
                print(f"  across-LLM std (mean per LLM): {m_std:.3f}")
                print(f"  across-prompt std (mean per style): {p_std:.3f}")
    except ImportError:
        print("statsmodels not available; reporting marginal stds only.")
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds].dropna(subset=["auroc"]).copy()
            if len(sub) < 4:
                continue
            print(f"\n{ds}:")
            m_std = sub.groupby("llm_model")["auroc"].mean().std()
            p_std = sub.groupby("prompt_style")["auroc"].mean().std()
            print(f"  across-LLM std: {m_std:.3f}")
            print(f"  across-prompt std: {p_std:.3f}")

    print(f"\n>>> Total time: {fmt_time(time.time() - t_global)}")
    print(f">>> Results: {output_dir}/exp18_results.csv")


if __name__ == "__main__":
    main()
