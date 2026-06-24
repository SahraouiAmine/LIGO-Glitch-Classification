"""
Test-set evaluation. Produces the numbers and figures for your report/README.

  python -m src.evaluate --checkpoint models/best_model.pt

Resources:
  - sklearn.metrics: classification_report, confusion_matrix, f1_score(average=...).
  - Report PER-CLASS precision/recall/F1, not just overall accuracy (trap #1).
  - Benchmark context: classic ensemble baseline ~98.21% overall; also compare
    against the April 2026 arXiv eval (2604.08796) on this same Zenodo dataset.
"""

from pathlib import Path


def load_model(checkpoint: Path, model_kind: str, num_classes: int):
    """Rebuild the architecture (from src.model) and load weights. TODO."""
    raise NotImplementedError


def evaluate(model, test_loader, device) -> dict:
    """Run inference over the test set. Return predictions, labels, and
    softmax probabilities (you'll reuse the probs for calibration). TODO."""
    raise NotImplementedError


def confusion_figure(labels, preds, class_names, out_path: Path) -> None:
    """Save a confusion matrix figure. In the report, DISCUSS which classes
    confuse each other and why (morphology: Blip vs. Tomte, scattered-light
    variants, etc.). The discussion is what shows physics understanding. TODO."""
    raise NotImplementedError


def main() -> None:
    """Build test loader, load model, evaluate, dump:
      - classification_report (text/CSV)
      - confusion matrix figure
      - the raw probs+labels (npz) for calibration.py to consume
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
