"""
=============================================================================
Experiment 4 — LLM Prior Auto-Construction Pipeline
=============================================================================
Automatic prior construction: variable names + descriptions -> LLM produces a P_prior matrix -> PRCD-MAP -> causal graph.

Pipeline:
  1. For each dataset (CausalTime AQI/Traffic/Medical + Electricity), build variable descriptions
  2. Call the Claude API (or fall back to a local LLM) to produce the causal prior matrix
  3. Generate 3 independent priors per dataset (different prompts)
  4. Run PRCD-MAP(trust) and PRCD-MAP(per-group)
  5. Assess robustness: variance of metrics across the 3 priors

Usage:
  python exp4_llm_prior.py --dataset AQI --seeds 0 1 2
  python exp4_llm_prior.py --dataset all --seeds 0 1 2
  python exp4_llm_prior.py --use-cached  # use previously generated priors
=============================================================================
"""

import os, sys, time, warnings, argparse, traceback, json, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_trust import *


# ====================================================================
# Variable Descriptions for CausalTime & Electricity
# ====================================================================

DATASET_DESCRIPTIONS = {
    "AQI": {
        "domain": "Air Quality Index monitoring at urban stations",
        "variables": {
            "PM2.5": "Fine particulate matter concentration (μg/m³)",
            "PM10": "Coarse particulate matter concentration (μg/m³)",
            "SO2": "Sulfur dioxide concentration (μg/m³)",
            "NO2": "Nitrogen dioxide concentration (μg/m³)",
            "CO": "Carbon monoxide concentration (mg/m³)",
            "O3": "Ozone concentration (μg/m³)",
            "temperature": "Ambient temperature (°C)",
            "pressure": "Atmospheric pressure (hPa)",
            "humidity": "Relative humidity (%)",
            "wind_speed": "Wind speed (m/s)",
            "wind_dir": "Wind direction (degrees)",
            "rain": "Precipitation (mm)",
        },
    },
    "Traffic": {
        "domain": "Urban traffic flow monitoring at sensor locations",
        "variables": {
            f"sensor_{i}": f"Traffic flow at sensor location {i} (vehicles/hour)"
            for i in range(12)
        },
    },
    "Medical": {
        "domain": "Clinical patient monitoring in ICU",
        "variables": {
            "heart_rate": "Heart rate (bpm)",
            "blood_pressure_sys": "Systolic blood pressure (mmHg)",
            "blood_pressure_dia": "Diastolic blood pressure (mmHg)",
            "respiratory_rate": "Respiratory rate (breaths/min)",
            "SpO2": "Blood oxygen saturation (%)",
            "temperature": "Body temperature (°C)",
        },
    },
    "Electricity": {
        "domain": "Regional electricity consumption by industrial sector",
        "variables": {
            "Large_Industrial": "Large industrial electricity consumption (MWh)",
            "Non_Std_Industrial": "Non-standard industrial consumption (MWh)",
            "Residential": "Residential electricity consumption (MWh)",
            "Commercial": "Commercial electricity consumption (MWh)",
            "Agriculture": "Agricultural electricity consumption (MWh)",
            "Ferrous_Metal": "Ferrous metal smelting consumption (MWh)",
            "Chemicals": "Chemical manufacturing consumption (MWh)",
            "Textiles": "Textile industry consumption (MWh)",
        },
    },
}


# ====================================================================
# LLM Prior Generation
# ====================================================================

def build_prompt(dataset_name: str, var_names: List[str],
                 var_descriptions: Dict[str, str],
                 domain: str, prompt_style: int = 0) -> str:
    """Build prompt for LLM to generate causal prior matrix."""

    var_list = "\n".join(f"  {i}. {name}: {var_descriptions.get(name, 'Unknown')}"
                         for i, name in enumerate(var_names))

    if prompt_style == 0:
        prompt = f"""You are an expert in causal inference and {domain}.

Given the following {len(var_names)} time-series variables:
{var_list}

For each ordered pair (i → j), estimate the probability that variable i has a direct
causal effect on variable j at the same time step. Output a {len(var_names)}×{len(var_names)}
matrix where entry [i][j] is the probability P(i→j) ∈ [0, 1].

Rules:
- Diagonal entries should be 0 (no self-causation)
- Consider domain knowledge about physical/economic mechanisms
- High probability (>0.7) only for well-established causal links
- Low probability (<0.3) for unlikely or reverse causal directions
- Medium probability (0.3-0.7) when uncertain

Output ONLY the matrix as a JSON array of arrays, no explanation."""

    elif prompt_style == 1:
        prompt = f"""As a domain expert in {domain}, analyze potential causal relationships
between these variables:
{var_list}

Think step by step:
1. What are the known physical/economic mechanisms?
2. What direct causal links are established in the literature?
3. What links are implausible?

Output a {len(var_names)}×{len(var_names)} causal prior probability matrix as JSON.
Entry [i][j] = P(variable_i → variable_j). Diagonal = 0.
Output ONLY the JSON matrix."""

    else:  # prompt_style == 2
        prompt = f"""Domain: {domain}
Variables:
{var_list}

Generate a causal prior matrix ({len(var_names)}×{len(var_names)}).
[i][j] = probability of direct causal effect from variable i to variable j.
Use conservative estimates. When very uncertain, use 0.5.
Diagonal entries = 0. JSON format only."""

    return prompt


def call_llm_for_prior(prompt: str, api_key: str,
                       model: str = "claude-sonnet-4-20250514",
                       max_retries: int = 3) -> np.ndarray:
    """Call Claude API to generate prior matrix. Raises on failure."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    last_err = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            matrix = _parse_matrix_from_text(response_text)
            if matrix is not None:
                return matrix
            last_err = f"Failed to parse matrix from response (attempt {attempt+1})"
            print(f"  [RETRY] {last_err}")
        except Exception as e:
            last_err = str(e)
            print(f"  [RETRY] API call attempt {attempt+1} failed: {e}")
            import time as _time
            _time.sleep(5)

    raise RuntimeError(f"LLM prior generation failed after {max_retries} retries: {last_err}")


def _parse_matrix_from_text(text: str) -> Optional[np.ndarray]:
    """Parse a matrix from LLM text output (JSON array of arrays)."""
    # Try direct JSON parse
    try:
        # Find JSON array in text
        match = re.search(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)
        if match:
            matrix = json.loads(match.group())
            return np.array(matrix, dtype=float)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try to extract numbers
    lines = text.strip().split('\n')
    rows = []
    for line in lines:
        nums = re.findall(r'[\d.]+', line)
        if nums:
            rows.append([float(x) for x in nums])
    if rows and all(len(r) == len(rows[0]) for r in rows):
        return np.array(rows, dtype=float)

    return None


def generate_fallback_prior(d: int, seed: int = 0,
                            sparsity: float = 0.2) -> np.ndarray:
    """Generate a random structured prior when LLM is unavailable."""
    rng = np.random.default_rng(seed + 7777)
    P = np.full((d, d), 0.5)
    # Random structure with domain-inspired blocks
    for i in range(d):
        for j in range(d):
            if i == j:
                P[i, j] = 0.0
                continue
            r = rng.random()
            if r < sparsity:
                P[i, j] = rng.uniform(0.6, 0.9)
            elif r < 2 * sparsity:
                P[i, j] = rng.uniform(0.1, 0.3)
            # else stays at 0.5
    return P


def generate_llm_priors(dataset_name: str, var_names: List[str], d: int,
                        n_priors: int = 3, api_key: str = "",
                        cache_dir: str = "llm_prior_cache") -> List[np.ndarray]:
    """Generate multiple independent LLM priors."""
    ensure_dir(cache_dir)
    priors = []

    desc = DATASET_DESCRIPTIONS.get(dataset_name, {})
    domain = desc.get("domain", f"{dataset_name} domain")
    var_descriptions = desc.get("variables", {})

    # Map variable names to descriptions
    if not var_descriptions:
        var_descriptions = {name: f"Variable {name}" for name in var_names}

    for style in range(n_priors):
        cache_file = os.path.join(cache_dir, f"{dataset_name}_prior_style{style}.npy")

        if os.path.exists(cache_file):
            P = np.load(cache_file)
            if P.shape == (d, d):
                priors.append(P)
                print(f"  Loaded cached prior style {style}: {cache_file}")
                continue

        if not api_key:
            raise RuntimeError(
                f"No cached prior for {dataset_name} style {style} and no API key. "
                f"Run generate_cached_priors.py first to create cached priors."
            )

        prompt = build_prompt(dataset_name, var_names, var_descriptions,
                              domain, prompt_style=style)

        P = call_llm_for_prior(prompt, api_key=api_key)
        if P.shape != (d, d):
            raise RuntimeError(
                f"LLM returned matrix shape {P.shape}, expected ({d},{d}). "
                f"Check variable descriptions for dataset '{dataset_name}'."
            )

        # Ensure valid prior
        P = np.clip(P, 0.01, 0.99)
        np.fill_diagonal(P, 0.0)

        np.save(cache_file, P)
        priors.append(P)
        print(f"  Generated prior style {style}: {cache_file}")

    return priors


# ====================================================================
# End-to-End Pipeline
# ====================================================================

def run_e2e_pipeline(X, B_true, var_names, dataset_name, priors, cfg, seeds):
    """Run end-to-end: LLM prior → PRCD-MAP → evaluate."""
    d = X.shape[1]
    K = 1
    W0_true = B_true.astype(float)
    Wk_true = [np.zeros_like(W0_true)]
    results = []

    for prior_idx, P_prior in enumerate(priors):
        for seed in seeds:
            # Trust propagation
            try:
                t0 = time.time()
                W0_t, Wk_t, tau_t = run_prcd_trust(
                    X, P_prior, d, K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, seed=seed)
                dt = time.time() - t0
                met = compute_dual_metrics(W0_true, Wk_true, W0_t, Wk_t)
                met.update({
                    "method": "PRCD-MAP(trust)", "seed": seed,
                    "prior_idx": prior_idx, "dataset": dataset_name,
                    "runtime": dt, "prior_type": "LLM",
                })
                results.append(met)
            except Exception as e:
                warnings.warn(f"trust failed: {e}")

            # Per-group baseline
            try:
                t0 = time.time()
                W0_p, Wk_p, tau_p = run_prcd_map(
                    X, P_prior, d, K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, seed=seed)
                dt = time.time() - t0
                met = compute_dual_metrics(W0_true, Wk_true, W0_p, Wk_p)
                met.update({
                    "method": "PRCD-MAP(per-group)", "seed": seed,
                    "prior_idx": prior_idx, "dataset": dataset_name,
                    "runtime": dt, "prior_type": "LLM",
                })
                results.append(met)
            except Exception as e:
                warnings.warn(f"per-group failed: {e}")

            # No-prior baseline (uniform P=0.5)
            try:
                P_uniform = np.full((d, d), 0.5)
                np.fill_diagonal(P_uniform, 0.0)
                W0_u, Wk_u, tau_u = run_prcd_trust(
                    X, P_uniform, d, K, lambda1=cfg.lambda1, lambda2=cfg.lambda2,
                    max_iter=cfg.max_iter, inner_iter=cfg.inner_iter,
                    lr=cfg.lr, seed=seed, learn_tau=False)
                met = compute_dual_metrics(W0_true, Wk_true, W0_u, Wk_u)
                met.update({
                    "method": "No-Prior", "seed": seed,
                    "prior_idx": prior_idx, "dataset": dataset_name,
                    "prior_type": "uniform",
                })
                results.append(met)
            except Exception:
                pass

    return results


@dataclass
class Cfg:
    seeds:          List[int] = field(default_factory=lambda: [0, 1, 2])
    lambda1:        float = 0.001
    lambda2:        float = 0.01
    max_iter:       int   = 35
    inner_iter:     int   = 400
    lr:             float = 1e-2
    n_priors:       int   = 3
    api_key:        str   = None
    causaltime_dir: str   = "./data/causaltime"
    electricity_xlsx: str = "./data/electricity.xlsx"
    electricity_prior: str = "./data/electricity_prior.csv"
    output_dir:     str   = "exp4_llm_prior_results"
    cache_dir:      str   = "llm_prior_cache"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["AQI", "Traffic", "Medical", "Electricity", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--use-cached", action="store_true",
                        help="Force offline mode: use cache only, never call any LLM API.")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--n-priors", type=int, default=3,
                        help="Number of independent priors per dataset (style0..style{N-1}).")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override cache directory (default: llm_prior_cache).")
    args = parser.parse_args()

    cfg = Cfg()
    if args.seeds:
        cfg.seeds = args.seeds
    cfg.n_priors = args.n_priors
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    # API key: --use-cached forces offline; otherwise CLI arg > env var > allowed empty only if the cache is complete
    if args.use_cached:
        api_key = None
        print(">>> --use-cached set: offline mode, will NOT call any API.")
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    cfg.api_key = api_key
    if api_key:
        print(f">>> API key loaded (ends with ...{api_key[-6:]})")
    elif not args.use_cached:
        print(">>> No API key — will use cached priors from llm_prior_cache/")
        print("    (Run generate_cached_priors.py first if cache is empty)")
    print(f">>> n_priors={cfg.n_priors}, cache_dir={cfg.cache_dir}")

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir)
    t_global = time.time()
    all_results = []

    datasets_to_run = (["AQI", "Traffic", "Medical", "Electricity"]
                       if args.dataset == "all" else [args.dataset])

    for ds_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load data
        if ds_name in ("AQI", "Traffic", "Medical"):
            X, B_true = load_causaltime(cfg.causaltime_dir, ds_name, n_samples=10)
            if X is None:
                print(f"  [SKIP] CausalTime {ds_name} not found")
                continue
            d = X.shape[1]
            var_names = [f"var_{i}" for i in range(d)]  # generic names
            desc = DATASET_DESCRIPTIONS.get(ds_name, {})
            if desc.get("variables"):
                var_names = list(desc["variables"].keys())[:d]
        elif ds_name == "Electricity":
            try:
                df_diff, P_existing, split, col_names = load_electricity(
                    cfg.electricity_xlsx, cfg.electricity_prior)
                X = standardize(df_diff.values[:split])
                d = X.shape[1]
                var_names = col_names
                # No ground truth for electricity — use existing prior as "reference"
                B_true = (P_existing > 0.5).astype(int)
                np.fill_diagonal(B_true, 0)
            except Exception as e:
                print(f"  [SKIP] Electricity data load failed: {e}")
                continue
        else:
            continue

        print(f"  d={d}, T={X.shape[0]}, vars={var_names[:5]}...")

        # Generate LLM priors
        priors = generate_llm_priors(
            ds_name, var_names, d,
            n_priors=cfg.n_priors, api_key=cfg.api_key,
            cache_dir=cfg.cache_dir)

        # Run pipeline
        results = run_e2e_pipeline(X, B_true, var_names, ds_name, priors,
                                    cfg, cfg.seeds)
        all_results.extend(results)

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(cfg.output_dir, "exp4_llm_prior_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> Saved {len(df)} rows -> {csv_path}")

        # Robustness analysis: variance across LLM priors
        print("\n" + "=" * 80)
        print("LLM PRIOR ROBUSTNESS ANALYSIS")
        print("=" * 80)
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            print(f"\n--- {ds} ---")
            for method in sub["method"].unique():
                m = sub[sub["method"] == method]
                auroc_by_prior = m.groupby("prior_idx")["auroc"].mean()
                print(f"  {method:30s}: AUROC per-prior={auroc_by_prior.values}, "
                      f"mean={m['auroc'].mean():.3f}, std={m['auroc'].std():.3f}")

    elapsed = time.time() - t_global
    print(f"\n>>> Exp4 complete in {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()
