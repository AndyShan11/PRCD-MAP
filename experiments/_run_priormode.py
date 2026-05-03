"""Wrapper: run exp1_synthetic_benchmark with custom prior_mode (systematic / adversarial / random).

Used to generate the prior-corruption-robustness data in 04_corruption_robustness.
The underlying script `exp1_synthetic_benchmark.py` does not expose
`prior_modes` via CLI; this wrapper loads the Cfg, overrides it, and
invokes `run_experiment` directly.

Usage:
    python _run_priormode.py --sub sample_size --prior-mode systematic \\
        --seeds 0 1 2 3 4 --no-rhino --no-ngc --no-nam
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from exp1_synthetic_benchmark import Cfg, run_experiment, generate_summary, cfg_sub


def main():
    p = argparse.ArgumentParser(description="exp1 wrapper with custom prior_mode")
    p.add_argument("--sub", required=True,
                   choices=["noise", "sample_size", "nonlinear", "scale"])
    p.add_argument("--prior-mode", required=True,
                   choices=["random", "systematic", "adversarial"],
                   help="prior corruption model")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--dims", nargs="+", type=int, default=None)
    p.add_argument("--no-rhino", action="store_true")
    p.add_argument("--no-ngc", action="store_true")
    p.add_argument("--no-nam", action="store_true")
    p.add_argument("--no-pcmci", action="store_true")
    p.add_argument("--no-varlingam", action="store_true")
    p.add_argument("--no-dynotears", action="store_true")
    p.add_argument("--skip-baselines", action="store_true")
    args = p.parse_args()

    cfg = cfg_sub(args.sub)
    cfg.prior_modes = [args.prior_mode]
    cfg.seeds = args.seeds
    if args.dims:
        cfg.dims = args.dims
    cfg.output_dir = f"exp1_{args.sub}_{args.prior_mode}"

    if args.skip_baselines:
        cfg.do_dynotears = cfg.do_pcmci = cfg.do_varlingam = False
        cfg.do_rhino = cfg.do_ngc = cfg.do_nam = False
    if args.no_rhino:     cfg.do_rhino = False
    if args.no_ngc:       cfg.do_ngc = False
    if args.no_nam:       cfg.do_nam = False
    if args.no_pcmci:     cfg.do_pcmci = False
    if args.no_varlingam: cfg.do_varlingam = False
    if args.no_dynotears: cfg.do_dynotears = False

    print(f">>> mode={args.prior_mode} sub={args.sub} dims={cfg.dims} "
          f"T={cfg.sample_sizes} seeds={cfg.seeds}", flush=True)

    df = run_experiment(cfg)
    generate_summary(df, cfg)
    print(">>> wrapper done", flush=True)


if __name__ == "__main__":
    main()
