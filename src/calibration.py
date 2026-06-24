"""
Confidence calibration — your bridge to the stats/Bayesian side without heavy theory.
A classifier that knows when it's unsure is directly useful to detchar workflows.

Resources:
  - Guo et al. 2017, "On Calibration of Modern Neural Networks" — THE temperature
    scaling paper. Short, readable. Implement temp scaling from it.
  - Reliability diagram + Expected Calibration Error (ECE): definitions are in
    that paper; binning predicted-confidence vs. empirical accuracy.
  - Fit the temperature on the VALIDATION set, then report ECE on TEST
    (before vs. after). Fitting on test would be cheating.
"""

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    """Bin by max-confidence, compare bin accuracy vs. bin confidence,
    weight by bin population. Return scalar ECE. TODO."""
    raise NotImplementedError


def fit_temperature(val_logits, val_labels) -> float:
    """Optimize a single scalar T>0 to minimize NLL on validation logits.
    A 1-D optimization (LBFGS or even a small grid). Return T. TODO."""
    raise NotImplementedError


def reliability_diagram(probs, labels, out_path, n_bins: int = 15) -> None:
    """Plot empirical accuracy vs. predicted confidence per bin, with the
    diagonal. Save before/after-scaling versions for the report. TODO."""
    raise NotImplementedError


def main() -> None:
    """Load the saved logits/labels from evaluate.py, compute ECE, fit T on val,
    recompute ECE on test, save both reliability diagrams. Print the delta."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
