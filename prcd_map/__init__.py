"""
PRCD-MAP: Safe Integration of Imperfect Domain Priors for Temporal Causal Discovery.

This package provides the core PRCD-MAP model for prior-calibrated temporal causal
discovery using MAP estimation with empirical Bayes temperature learning.
"""

from prcd_map.model import PRCD_MAP_Model, run_prcd_map

__version__ = "0.1.0"
__all__ = ["PRCD_MAP_Model", "run_prcd_map"]
