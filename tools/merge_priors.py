"""
merge_priors.py — Combine GPT / Claude / Gemini cached priors into a single
unified llm_prior_cache/ for exp9. The 5 final priors per dataset are sourced
from 3 different frontier LLMs × different prompting strategies, so the paper
can claim "distribution-level positive shift across diverse LLM sources",
which directly answers reviewer concern about single-draw priors.

Mapping (kept fixed so paper text matches what's loaded):

  merged_style0  <-  GPT     style0   (conservative textbook)
  merged_style1  <-  Claude  style1   (step-by-step mechanism)
  merged_style2  <-  Gemini  style2   (literature-anchored)
  merged_style3  <-  GPT     style3   (permissive)
  merged_style4  <-  Claude  style4   (adversarial / reverse-checking)

Run from repo root:
  python tools/merge_priors.py
"""
import os, json, shutil
from pathlib import Path
import numpy as np


SRC = {
    "gpt":    Path("./llm_prior_cache_gpt"),
    "claude": Path("./llm_prior_cache_claude"),
    "gemini": Path("./llm_prior_cache_gemini"),
}

DST = Path(__file__).parent.parent / "experiments" / "llm_prior_cache"

DATASETS = ["AQI", "Traffic", "Medical", "Electricity"]
EXPECTED_D = {"AQI": 12, "Traffic": 12, "Medical": 6, "Electricity": 8}

# (merged_style_idx, source_llm, source_style_idx)
MAPPING = [
    (0, "gpt",    0),
    (1, "claude", 1),
    (2, "gemini", 2),
    (3, "gpt",    3),
    (4, "claude", 4),
]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    metadata = {"mapping": [], "validation": []}

    for ds in DATASETS:
        d = EXPECTED_D[ds]
        for merged_idx, llm, src_idx in MAPPING:
            src = SRC[llm] / f"{ds}_prior_style{src_idx}.npy"
            dst = DST / f"{ds}_prior_style{merged_idx}.npy"

            P = np.load(src)
            assert P.shape == (d, d), f"{src}: shape {P.shape}, expected ({d},{d})"
            assert np.allclose(np.diag(P), 0), f"{src}: diagonal not zero"

            offdiag = P[~np.eye(d, dtype=bool)]
            assert offdiag.min() >= 0.01 and offdiag.max() <= 0.99, \
                f"{src}: off-diag out of [0.01, 0.99]"

            np.save(dst, P)
            n_strong = int(((P > 0.7) | ((P < 0.3) & (P > 0))).sum())
            metadata["mapping"].append({
                "dataset": ds,
                "merged_style": merged_idx,
                "source_llm": llm,
                "source_style": src_idx,
                "shape": list(P.shape),
                "non_default_edges": int(((P > 0.55) | (P < 0.45)).sum()),
                "strong_edges": n_strong,
                "src_path": str(src),
                "dst_path": str(dst),
            })
            print(f"{ds:12s} merged_style{merged_idx} <- {llm:6s} style{src_idx}  "
                  f"shape={P.shape} edges={metadata['mapping'][-1]['non_default_edges']}")

    pairwise_disagreement = []
    for ds in DATASETS:
        d = EXPECTED_D[ds]
        Ps = [np.load(DST / f"{ds}_prior_style{i}.npy") for i in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                diff = np.abs(Ps[i] - Ps[j]).mean()
                pairwise_disagreement.append(
                    {"dataset": ds, "pair": [i, j], "mean_abs_diff": float(diff)})
    metadata["validation"] = pairwise_disagreement

    with open(DST / "_merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {len(MAPPING) * len(DATASETS)} priors to {DST}")
    print(f"Wrote metadata to {DST / '_merge_metadata.json'}")

    print("\n=== pairwise mean-abs-diff per dataset (higher = more diverse) ===")
    by_ds = {}
    for r in pairwise_disagreement:
        by_ds.setdefault(r["dataset"], []).append(r["mean_abs_diff"])
    for ds, vals in by_ds.items():
        print(f"  {ds:12s} mean={np.mean(vals):.3f}  min={min(vals):.3f}  "
              f"max={max(vals):.3f}")


if __name__ == "__main__":
    main()
