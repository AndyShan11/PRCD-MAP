"""
=============================================================================
Experiment 6 — Structural-Community Heterogeneity Validation
=============================================================================
诊断 exp5 失败的原因: 社区按变量索引划分, TrustPropagationLite 的 row/col
均值特征会混合两社区的边, 无法区分.

新思路: 让社区按图结构角色划分 (hub vs peripheral), 这样 row/col 均值天然
区分不同社区, TrustPropagationLite 的现有特征就能利用.

设计:
  - BA 图 (m=2): 产生明显的 hub 结构, 少数节点有大量连接
  - Hub 节点 (度数 top 30%): acc_high 先验 (物理定律级)
  - Peripheral 节点 (度数 bottom 70%): acc_low 先验 (LLM 猜测级)
  - P_prior 值范围重叠, 但 row/col 均值差异大

Variants (4种, 4个GPU并行):
  V1: BA 图 + 结构异质, d=20, linear/nonlinear
  V2: BA 图 + 结构异质, d=30, linear/nonlinear
  V3: ER 图 + 度数异质 (high-degree edges get high acc), d=20
  V4: 极端异质 (acc_high=1.0, acc_low=0.0) + BA, d=20

Usage (单 GPU 全跑, ~2h):
  python exp6_structural_community.py --variant all --seeds 0 1 2

Usage (4 GPU 并行, ~30min):
  CUDA_VISIBLE_DEVICES=0 python exp6_structural_community.py --variant v1 --seeds 0 1 2
  CUDA_VISIBLE_DEVICES=1 python exp6_structural_community.py --variant v2 --seeds 0 1 2
  CUDA_VISIBLE_DEVICES=2 python exp6_structural_community.py --variant v3 --seeds 0 1 2
  CUDA_VISIBLE_DEVICES=3 python exp6_structural_community.py --variant v4 --seeds 0 1 2
=============================================================================
"""

import os, sys, time, warnings, argparse, traceback
from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_trust import *


def gen_structural_prior(W0_true, Wk_true, acc_hub, acc_periph,
                         hub_ratio=0.3, seed=0):
    """
    按节点度数划分社区: hub 节点 (度 top hub_ratio) 边用高精度先验,
    peripheral 节点边用低精度先验.

    关键: 一条边 (i,j) 的"社区归属"由 max(degree(i), degree(j)) 决定.
    Hub 相关的边 → 精准先验, 不涉及 hub 的边 → 噪声先验.
    """
    rng = np.random.default_rng(seed + 6666)
    d = W0_true.shape[0]

    B_all = (np.abs(W0_true) > 1e-10).astype(int)
    for Wk in Wk_true:
        B_all = np.maximum(B_all, (np.abs(Wk) > 1e-10).astype(int))

    # Compute node degrees (total in + out on true graph)
    degrees = B_all.sum(axis=0) + B_all.sum(axis=1)
    n_hubs = max(1, int(d * hub_ratio))
    hub_threshold = np.sort(degrees)[::-1][n_hubs - 1]
    is_hub = degrees >= hub_threshold

    P = np.full((d, d), 0.5)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            true_edge = B_all[i, j] == 1
            # Edge community: hub if EITHER endpoint is a hub
            is_hub_edge = is_hub[i] or is_hub[j]
            if is_hub_edge:
                acc = acc_hub
                if rng.random() < acc:
                    P[i, j] = rng.uniform(0.75, 0.95) if true_edge else rng.uniform(0.08, 0.28)
                else:
                    P[i, j] = rng.uniform(0.08, 0.28) if true_edge else rng.uniform(0.75, 0.95)
            else:
                acc = acc_periph
                if rng.random() < acc:
                    P[i, j] = rng.uniform(0.60, 0.85) if true_edge else rng.uniform(0.20, 0.45)
                else:
                    P[i, j] = rng.uniform(0.20, 0.45) if true_edge else rng.uniform(0.60, 0.85)
    return P, is_hub


def gen_degree_prior(W0_true, Wk_true, acc_high, acc_low, seed=0):
    """
    按边度数(两端点度之和)划分社区.
    度数 top 30% 的边用高精度, 其余用低精度.
    """
    rng = np.random.default_rng(seed + 7777)
    d = W0_true.shape[0]
    B_all = (np.abs(W0_true) > 1e-10).astype(int)
    for Wk in Wk_true:
        B_all = np.maximum(B_all, (np.abs(Wk) > 1e-10).astype(int))

    degrees = B_all.sum(axis=0) + B_all.sum(axis=1)
    # Edge "degree" = sum of endpoints
    edge_deg = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                edge_deg[i, j] = degrees[i] + degrees[j]

    # Top 30% edges by degree get high accuracy
    thr = np.percentile(edge_deg[edge_deg > 0], 70)

    P = np.full((d, d), 0.5)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            true_edge = B_all[i, j] == 1
            is_high_deg = edge_deg[i, j] >= thr
            acc = acc_high if is_high_deg else acc_low
            if rng.random() < acc:
                P[i, j] = rng.uniform(0.70, 0.95) if true_edge else rng.uniform(0.10, 0.35)
            else:
                P[i, j] = rng.uniform(0.10, 0.35) if true_edge else rng.uniform(0.70, 0.95)
    return P


# ====================================================================
# Variant configurations
# ====================================================================

VARIANTS = {
    "v1": {
        "name": "BA_d20",
        "graph": "BA", "d": 20, "T": 500,
        "prior_type": "structural",
        "settings": [(0.9, 0.3), (0.95, 0.2), (0.85, 0.35)],
        "nonlinear_options": [False, True],
    },
    "v2": {
        "name": "BA_d30",
        "graph": "BA", "d": 30, "T": 500,
        "prior_type": "structural",
        "settings": [(0.9, 0.3), (0.95, 0.2)],
        "nonlinear_options": [False, True],
    },
    "v3": {
        "name": "ER_degree_d20",
        "graph": "ER", "d": 20, "T": 500,
        "prior_type": "degree",
        "settings": [(0.9, 0.3), (0.95, 0.2), (0.85, 0.35)],
        "nonlinear_options": [False, True],
    },
    "v4": {
        "name": "BA_extreme_d20",
        "graph": "BA", "d": 20, "T": 500,
        "prior_type": "structural",
        "settings": [(1.0, 0.0), (0.98, 0.1), (1.0, 0.2)],
        "nonlinear_options": [False, True],
    },
}


def run_variant(variant_name, variant_cfg, seeds, do_baselines=True):
    """Run one variant configuration."""
    results = []
    d = variant_cfg["d"]
    T = variant_cfg["T"]
    K = 1

    for is_nonlinear in variant_cfg["nonlinear_options"]:
        for acc_high, acc_low in variant_cfg["settings"]:
            for seed in seeds:
                setting = (f"{variant_cfg['name']}_h{int(acc_high*100)}_l{int(acc_low*100)}"
                           f"_{'NL' if is_nonlinear else 'LIN'}")
                print(f"\n  {setting} seed={seed}")
                t0 = time.time()

                try:
                    set_seed(seed)
                    if variant_cfg["graph"] == "BA":
                        W0_true = make_ba_dag(d, m=2, seed=seed)
                    else:
                        W0_true = make_er_dag(d, edge_prob=0.15, seed=seed)
                    Wk_true = make_lag_matrices(d, K, edge_prob=0.10, seed=seed)

                    if is_nonlinear:
                        X = simulate_svar_nonlinear(T, W0_true, Wk_true, seed=seed)
                    else:
                        X = simulate_svar_linear(T, W0_true, Wk_true, seed=seed)
                    if X is None:
                        print("    [SKIP] simulation failed")
                        continue
                    X = standardize(X)

                    # Structural heterogeneous prior
                    if variant_cfg["prior_type"] == "structural":
                        P_prior, _ = gen_structural_prior(W0_true, Wk_true,
                                                          acc_high, acc_low, seed=seed)
                    else:
                        P_prior = gen_degree_prior(W0_true, Wk_true,
                                                   acc_high, acc_low, seed=seed)

                    # Methods
                    def _eval(name, W0_est, Wk_est):
                        if W0_est is None:
                            return
                        met = compute_dual_metrics(W0_true, Wk_true, W0_est, Wk_est)
                        met.update({
                            "method": name, "seed": seed, "setting": setting,
                            "variant": variant_name, "d": d,
                            "acc_high": acc_high, "acc_low": acc_low,
                            "nonlinear": is_nonlinear,
                        })
                        results.append(met)

                    # 1. trust
                    W0, Wk, tau = run_prcd_trust(
                        X, P_prior, d, K,
                        lambda1=0.001, lambda2=0.01,
                        max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
                    _eval("trust", W0, Wk)

                    # 2. per-group
                    W0, Wk, tau = run_prcd_map(
                        X, P_prior, d, K,
                        lambda1=0.001, lambda2=0.01,
                        max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
                    _eval("per-group", W0, Wk)

                    # 3. fixed tau=1
                    W0, Wk, tau = run_prcd_map(
                        X, P_prior, d, K,
                        lambda1=0.001, lambda2=0.01,
                        learn_tau=False, tau0=1.0,
                        max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
                    _eval("fixed-tau", W0, Wk)

                    # 4. no-prior
                    P_flat = np.full((d, d), 0.5)
                    np.fill_diagonal(P_flat, 0.0)
                    W0, Wk, tau = run_prcd_trust(
                        X, P_flat, d, K,
                        lambda1=0.001, lambda2=0.01,
                        learn_tau=False,
                        max_iter=35, inner_iter=400, lr=8e-3, seed=seed)
                    _eval("no-prior", W0, Wk)

                    # 5. baselines
                    if do_baselines:
                        try:
                            W0, Wk = run_dynotears(X, d, K, seed=seed)
                            _eval("DYNOTEARS", W0, Wk)
                        except Exception:
                            pass
                        try:
                            W0, Wk = run_pcmci_plus(X, d, K, seed=seed)
                            _eval("PCMCI+", W0, Wk)
                        except Exception:
                            pass

                    # Print comparison
                    trust = [r["auroc"] for r in results[-7:] if r.get("method") == "trust"]
                    pg = [r["auroc"] for r in results[-7:] if r.get("method") == "per-group"]
                    if trust and pg:
                        print(f"    [{fmt_time(time.time()-t0)}] trust={trust[0]:.3f} "
                              f"pg={pg[0]:.3f} Δ={trust[0]-pg[0]:+.4f}")

                except Exception as e:
                    traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="all",
                        choices=["all", "v1", "v2", "v3", "v4"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    output_dir = "exp6_structural"
    ensure_dir(output_dir)

    variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    t_global = time.time()
    all_results = []

    for v in variants_to_run:
        print(f"\n{'='*60}\nVariant {v}: {VARIANTS[v]['name']}\n{'='*60}")
        results = run_variant(v, VARIANTS[v], args.seeds, not args.skip_baselines)
        all_results.extend(results)

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, f"exp6_{args.variant}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> Saved {len(df)} rows -> {csv_path}")

        # Summary
        print("\n" + "=" * 80)
        print("STRUCTURAL COMMUNITY VALIDATION — SUMMARY")
        print("=" * 80)

        print("\n--- trust vs per-group by setting ---")
        for setting in sorted(df["setting"].unique()):
            sub = df[df["setting"] == setting]
            t = sub[sub["method"] == "trust"]["auroc"]
            p = sub[sub["method"] == "per-group"]["auroc"]
            if len(t) > 0 and len(p) > 0:
                delta_auroc = t.mean() - p.mean()
                t_f1 = sub[sub["method"] == "trust"]["f1_opt"].mean()
                p_f1 = sub[sub["method"] == "per-group"]["f1_opt"].mean()
                delta_f1 = t_f1 - p_f1
                print(f"  {setting:40s}: AUROC Δ={delta_auroc:+.4f}  F1 Δ={delta_f1:+.4f}")

        print("\n--- Method means by variant (nonlinear only if present) ---")
        for v in df["variant"].unique():
            for nl in [False, True]:
                sub = df[(df["variant"] == v) & (df["nonlinear"] == nl)]
                if len(sub) == 0:
                    continue
                nl_label = "NL" if nl else "LIN"
                print(f"\n  {v} {nl_label}:")
                for method in ["trust", "per-group", "fixed-tau", "no-prior",
                               "DYNOTEARS", "PCMCI+"]:
                    m = sub[sub["method"] == method]
                    if len(m) > 0:
                        print(f"    {method:15s}: AUROC={m['auroc'].mean():.3f}±{m['auroc'].std():.3f}"
                              f"  F1={m['f1_opt'].mean():.3f}±{m['f1_opt'].std():.3f}")

    elapsed = time.time() - t_global
    print(f"\n>>> Exp6 complete in {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()
