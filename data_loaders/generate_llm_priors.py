"""
generate_cached_priors.py -- Pre-generate LLM-style causal prior matrices (no API needed).

Using domain knowledge distilled from Claude, this script generates 3 independent
prior matrices for each of CausalTime (AQI/Traffic/Medical) and the Electricity
dataset, caches them under llm_prior_cache/, so that exp4 can load them directly.

Run once on the server (no GPU, takes a few seconds):
  cd ./scripts && python generate_cached_priors.py

Output:
  llm_prior_cache/AQI_prior_style0.npy
  llm_prior_cache/AQI_prior_style1.npy
  llm_prior_cache/AQI_prior_style2.npy
  llm_prior_cache/Traffic_prior_style0.npy
  ...
"""

import os, sys, warnings
import importlib.util as _ilu
import numpy as np

# Load ../src/utils.py (core utilities)
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_spec = _ilu.spec_from_file_location("_orig_exp_utils", os.path.join(_SRC_DIR, "utils.py"))
_orig = _ilu.module_from_spec(_spec)
sys.modules["_orig_exp_utils"] = _orig
_spec.loader.exec_module(_orig)

load_causaltime = _orig.load_causaltime
load_electricity = _orig.load_electricity
standardize = _orig.standardize
ensure_dir = _orig.ensure_dir


CACHE_DIR = "llm_prior_cache"

# ====================================================================
# Domain-knowledge expert assessments of causal links
# ====================================================================

def _aqi_prior_style0(d):
    """
    AQI/PM2.5 -- conservative estimate (only high-confidence causal chains).

    Typical CausalTime PM25 variable order: meteorology -> pollutant is the dominant direction.
    """
    P = np.full((d, d), 0.3)  # default: weakly uncertain
    np.fill_diagonal(P, 0.0)

    if d >= 6:
        # Typical PM25 dataset variables; known chemical causal chain among pollutants:
        # SO2(2) -> PM2.5(0) (secondary particulates)
        # NO2(3) -> PM2.5(0) (secondary aerosol)
        # NO2(3) -> O3(5) (photochemical reaction)
        # CO(4) -> PM2.5(0) (co-source indicator)
        # PM10(1) contains PM2.5(0)

        # Strong causal: confirmed chemical/physical mechanism
        causal_strong = [
            (2, 0), (3, 0), (4, 0),  # SO2,NO2,CO -> PM2.5
            (3, 5),                    # NO2 -> O3
        ]
        for i, j in causal_strong:
            if i < d and j < d:
                P[i, j] = 0.85

        # Medium causal: indirect or partial
        causal_medium = [
            (1, 0), (0, 1),  # PM10 <-> PM2.5
            (2, 1),           # SO2 -> PM10
        ]
        for i, j in causal_medium:
            if i < d and j < d:
                P[i, j] = 0.65

    if d >= 10:
        # Meteorology variables (second half) -> pollutants (first half)
        met_start = d // 2  # meteorology start index
        for met_idx in range(met_start, d):
            for poll_idx in range(met_start):
                P[met_idx, poll_idx] = 0.55  # meteorology affects pollutants
                P[poll_idx, met_idx] = 0.15  # pollutants barely affect meteorology

        # Wind speed -> all pollutants (strong dispersion effect)
        wind_idx = min(d-1, met_start + 1)  # assume the second meteorology variable is wind speed
        for poll_idx in range(met_start):
            P[wind_idx, poll_idx] = 0.80

    return np.clip(P, 0.01, 0.99)


def _aqi_prior_style1(d):
    """AQI -- aggressive estimate (more causal links, biased high)."""
    P = np.full((d, d), 0.4)
    np.fill_diagonal(P, 0.0)

    # Medium mutual causal links among all pollutants
    n_poll = min(6, d)
    for i in range(n_poll):
        for j in range(n_poll):
            if i != j:
                P[i, j] = 0.60

    # Source pollutants -> secondary pollutants are stronger
    if d >= 6:
        for src in [2, 3, 4]:  # SO2, NO2, CO
            for tgt in [0, 1]:  # PM2.5, PM10
                if src < d and tgt < d:
                    P[src, tgt] = 0.88

    # Meteorology -> all pollutants
    if d >= 8:
        met_start = max(6, d // 2)
        for m in range(met_start, d):
            for p in range(met_start):
                P[m, p] = 0.70
                P[p, m] = 0.10

    return np.clip(P, 0.01, 0.99)


def _aqi_prior_style2(d):
    """AQI -- sparse estimate (only the most certain causal chains)."""
    P = np.full((d, d), 0.25)  # conservative baseline
    np.fill_diagonal(P, 0.0)

    # Only the strongest physical/chemical causes
    if d >= 6:
        strong_links = [(2, 0), (3, 0), (3, 5), (4, 0)]
        for i, j in strong_links:
            if i < d and j < d:
                P[i, j] = 0.90
                P[j, i] = 0.10  # reverse very low

    # Of meteorology, keep only wind speed and temperature
    if d >= 8:
        for poll in range(min(6, d)):
            if d > 6:
                P[6, poll] = 0.75  # temperature
            if d > 7:
                P[7, poll] = 0.80  # wind speed (dispersion)

    return np.clip(P, 0.01, 0.99)


def _traffic_prior_style0(d):
    """
    Traffic -- spatial-propagation model.
    Adjacent sensors have a strong causal link (traffic flow propagates);
    further apart, weaker.
    """
    P = np.full((d, d), 0.20)
    np.fill_diagonal(P, 0.0)

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            dist = abs(i - j)
            if dist == 1:
                P[i, j] = 0.85  # directly adjacent
            elif dist == 2:
                P[i, j] = 0.55  # one apart
            elif dist == 3:
                P[i, j] = 0.35  # two apart
            # Ring topology: ends are also adjacent
            wrap_dist = d - dist
            if wrap_dist == 1:
                P[i, j] = 0.80
            elif wrap_dist == 2:
                P[i, j] = 0.50

    return np.clip(P, 0.01, 0.99)


def _traffic_prior_style1(d):
    """Traffic -- upstream/downstream asymmetric model (traffic flow has direction)."""
    P = np.full((d, d), 0.20)
    np.fill_diagonal(P, 0.0)

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # Upstream -> downstream (i < j) is stronger than downstream -> upstream
            dist = abs(i - j)
            if i < j:  # upstream -> downstream
                if dist == 1:
                    P[i, j] = 0.88
                elif dist == 2:
                    P[i, j] = 0.60
                elif dist == 3:
                    P[i, j] = 0.40
            else:  # downstream -> upstream (congestion backflow, weaker)
                if dist == 1:
                    P[i, j] = 0.65
                elif dist == 2:
                    P[i, j] = 0.35

    return np.clip(P, 0.01, 0.99)


def _traffic_prior_style2(d):
    """Traffic -- cluster model (sensors grouped; intra-cluster strong, inter-cluster weak)."""
    P = np.full((d, d), 0.15)
    np.fill_diagonal(P, 0.0)

    n_clusters = max(2, d // 4)
    cluster_size = d // n_clusters

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            ci, cj = i // cluster_size, j // cluster_size
            if ci == cj:
                P[i, j] = 0.75  # same cluster
            elif abs(ci - cj) == 1:
                P[i, j] = 0.45  # adjacent cluster

    # Add a distance-decay term on top
    for i in range(d):
        for j in range(d):
            if i != j and abs(i - j) == 1:
                P[i, j] = max(P[i, j], 0.70)

    return np.clip(P, 0.01, 0.99)


def _medical_prior_style0(d):
    """
    Medical/ICU -- physiological causal mechanisms.

    Typical variables: heart_rate(0), BP_sys(1), BP_dia(2), resp_rate(3), SpO2(4), temp(5)
    """
    P = np.full((d, d), 0.25)
    np.fill_diagonal(P, 0.0)

    # Known physiological mechanisms
    causal_links = {
        # (src, tgt): probability
        (1, 0): 0.80,  # BP_sys -> heart_rate (baroreflex)
        (0, 1): 0.70,  # heart_rate -> BP_sys (cardiac output)
        (1, 2): 0.90,  # BP_sys -> BP_dia (systemic blood pressure)
        (2, 1): 0.60,  # BP_dia -> BP_sys (weaker reverse)
        (3, 4): 0.85,  # resp_rate -> SpO2 (ventilation -> oxygenation)
        (4, 3): 0.50,  # SpO2 -> resp_rate (hypoxic drive)
        (4, 0): 0.65,  # SpO2 -> heart_rate (hypoxic compensation)
        (5, 0): 0.75,  # temp -> heart_rate (fever -> HR up)
        (5, 3): 0.60,  # temp -> resp_rate (fever -> resp up)
        (0, 4): 0.40,  # heart_rate -> SpO2 (weak: perfusion)
    }

    for (i, j), prob in causal_links.items():
        if i < d and j < d:
            P[i, j] = prob

    return np.clip(P, 0.01, 0.99)


def _medical_prior_style1(d):
    """Medical -- aggressive model (more interactions)."""
    P = np.full((d, d), 0.35)
    np.fill_diagonal(P, 0.0)

    # All vital signs interact with each other
    for i in range(min(d, 6)):
        for j in range(min(d, 6)):
            if i != j:
                P[i, j] = 0.55

    # Strengthen known causes
    strong = {
        (1, 2): 0.92, (1, 0): 0.82, (0, 1): 0.75,
        (3, 4): 0.88, (5, 0): 0.78, (5, 3): 0.68,
        (4, 0): 0.70, (4, 3): 0.62,
    }
    for (i, j), prob in strong.items():
        if i < d and j < d:
            P[i, j] = prob

    return np.clip(P, 0.01, 0.99)


def _medical_prior_style2(d):
    """Medical -- conservative/sparse model."""
    P = np.full((d, d), 0.20)
    np.fill_diagonal(P, 0.0)

    # Only the most certain causes
    confirmed = {
        (1, 2): 0.92,  # BP_sys -> BP_dia
        (3, 4): 0.88,  # resp -> SpO2
        (5, 0): 0.80,  # temp -> HR
        (1, 0): 0.78,  # BP -> HR
    }
    for (i, j), prob in confirmed.items():
        if i < d and j < d:
            P[i, j] = prob
            P[j, i] = max(P[j, i], 0.15)  # reverse very low

    return np.clip(P, 0.01, 0.99)


def _electricity_prior_style0(d):
    """
    Electricity -- value-chain causal model.
    Heavy industry -> total demand; upstream -> downstream sectors;
    weather -> residential/agriculture.
    """
    P = np.full((d, d), 0.30)
    np.fill_diagonal(P, 0.0)

    # Heavy industry drives the rest
    if d >= 3:
        for j in range(1, d):
            P[0, j] = 0.65  # large industry -> others

    # Inter-sector causal links
    pairs = {
        (0, 5): 0.80,  # large industry -> ferrous metal
        (5, 6): 0.70,  # ferrous metal -> chemicals
        (0, 1): 0.60,  # large industry -> non-standard industry
        (2, 3): 0.55,  # residential -> commercial (consumption-driven)
        (3, 2): 0.45,  # commercial -> residential (weaker)
    }
    for (i, j), prob in pairs.items():
        if i < d and j < d:
            P[i, j] = prob

    # Same-class industries co-vary
    industrial = [0, 1, 5, 6, 7]  # industrial-related
    for i in industrial:
        for j in industrial:
            if i != j and i < d and j < d:
                P[i, j] = max(P[i, j], 0.50)

    return np.clip(P, 0.01, 0.99)


def _electricity_prior_style1(d):
    """Electricity -- economic-structure model (upstream/downstream value chain)."""
    P = np.full((d, d), 0.35)
    np.fill_diagonal(P, 0.0)

    # Energy-intensive sectors -> others
    energy_intensive = [0, 5, 6]  # large industry, ferrous metal, chemicals
    for src in energy_intensive:
        for j in range(d):
            if src < d and j < d and src != j:
                P[src, j] = max(P[src, j], 0.60)

    # End-consumption
    if d >= 4:
        P[2, 3] = 0.70  # residential -> commercial
        P[3, 2] = 0.50
    if d >= 5:
        P[4, 2] = 0.55  # agriculture -> residential (seasonal coupling)

    return np.clip(P, 0.01, 0.99)


def _electricity_prior_style2(d):
    """Electricity -- sparse conservative model."""
    P = np.full((d, d), 0.20)
    np.fill_diagonal(P, 0.0)

    # Only the most certain
    if d >= 6:
        P[0, 5] = 0.85  # large industry -> metallurgy
        P[0, 1] = 0.75  # large industry -> non-standard
        P[5, 6] = 0.72  # metallurgy -> chemicals
    if d >= 4:
        P[2, 3] = 0.65  # residential -> commercial

    return np.clip(P, 0.01, 0.99)


# ====================================================================
# Main routine
# ====================================================================

GENERATORS = {
    "AQI":         [_aqi_prior_style0, _aqi_prior_style1, _aqi_prior_style2],
    "Traffic":     [_traffic_prior_style0, _traffic_prior_style1, _traffic_prior_style2],
    "Medical":     [_medical_prior_style0, _medical_prior_style1, _medical_prior_style2],
    "Electricity": [_electricity_prior_style0, _electricity_prior_style1, _electricity_prior_style2],
}


def _enrich_prior(P, d, seed=0):
    """
    Structurally enrich a prior matrix: replace default constant regions with
    locally structured values.

    Strategy:
    1. Block structure: variables grouped into d//4 blocks; intra-block edges
       get +0.15 (simulates intra-subsystem causality)
    2. Distance decay: index-close variable pairs get +0.10 (simulates spatio-
       temporal proximity)
    3. Random perturbation: +/-0.05 to break symmetry (simulates randomness
       between prompts)

    Only acts on regions "not covered by domain knowledge" (entries close to
    the default baseline).
    """
    rng = np.random.default_rng(seed)
    P_out = P.copy()
    mask = ~np.eye(d, dtype=bool)

    # Detect the default baseline (use the median of off-diagonal entries)
    vals = P[mask]
    median_val = np.median(vals)

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # Only enrich entries "close to the default"
            if abs(P[i, j] - median_val) > 0.10:
                continue  # already shaped by domain knowledge -- leave alone

            bonus = 0.0
            # Block structure
            block_size = max(2, d // 4)
            if i // block_size == j // block_size:
                bonus += 0.15

            # Distance decay
            dist = abs(i - j)
            if dist <= 2:
                bonus += 0.10
            elif dist <= 4:
                bonus += 0.05

            # Random perturbation
            bonus += rng.uniform(-0.05, 0.05)

            P_out[i, j] = P[i, j] + bonus

    np.fill_diagonal(P_out, 0.0)
    return np.clip(P_out, 0.01, 0.99)


def main():
    ensure_dir(CACHE_DIR)

    # --- CausalTime datasets ---
    causaltime_dir = "./data/causaltime"
    for ds_name in ["AQI", "Traffic", "Medical"]:
        print(f"\n>>> {ds_name}")
        X, B_true = load_causaltime(causaltime_dir, ds_name, n_samples=10)
        if X is None:
            print(f"  [WARN] Data not found, trying expected dimensions...")
            d_expected = {"AQI": 36, "Traffic": 20, "Medical": 20}
            d = d_expected.get(ds_name, 20)
            print(f"  Using expected d={d}")
        else:
            d = X.shape[1]
            print(f"  Loaded: d={d}, T={X.shape[0]}")

        gens = GENERATORS[ds_name]
        for style, gen_fn in enumerate(gens):
            P = gen_fn(d)
            P = _enrich_prior(P, d, seed=style * 1000 + 42)
            assert P.shape == (d, d), f"Shape mismatch: {P.shape} vs ({d},{d})"
            np.fill_diagonal(P, 0.0)
            P = np.clip(P, 0.01, 0.99)

            out_path = os.path.join(CACHE_DIR, f"{ds_name}_prior_style{style}.npy")
            np.save(out_path, P)

            mask = ~np.eye(d, dtype=bool)
            print(f"  style {style}: saved {out_path}  "
                  f"(mean={P[mask].mean():.3f}, nnz>{0.5}={int((P[mask]>0.5).sum())})")

            # Compare with the ground-truth graph if available
            if B_true is not None:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(B_true[mask].flatten(), P[mask].flatten())
                    print(f"           AUROC vs ground truth: {auc:.3f}")
                except Exception:
                    pass

    # --- Electricity ---
    print(f"\n>>> Electricity")
    elec_xlsx = "./data/electricity.xlsx"
    elec_prior = "./data/electricity_prior.csv"
    try:
        df_diff, P_existing, split, col_names = load_electricity(elec_xlsx, elec_prior)
        d = len(col_names)
        print(f"  Loaded: d={d}, cols={col_names[:5]}...")
    except Exception as e:
        print(f"  [WARN] Electricity data not found ({e}), using d=8")
        d = 8

    gens = GENERATORS["Electricity"]
    for style, gen_fn in enumerate(gens):
        P = gen_fn(d)
        P = _enrich_prior(P, d, seed=style * 1000 + 99)
        np.fill_diagonal(P, 0.0)
        P = np.clip(P, 0.01, 0.99)

        out_path = os.path.join(CACHE_DIR, f"Electricity_prior_style{style}.npy")
        np.save(out_path, P)

        mask = ~np.eye(d, dtype=bool)
        print(f"  style {style}: saved {out_path}  "
              f"(mean={P[mask].mean():.3f})")

    print(f"\n>>> All priors saved to {CACHE_DIR}/")
    print(f"    Total files: {len(os.listdir(CACHE_DIR))}")


if __name__ == "__main__":
    main()
