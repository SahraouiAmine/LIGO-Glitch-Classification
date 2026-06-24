"""
Training entry point. Run as a CLI; this is part of the deliverable, not a notebook.

  python -m src.train --model baseline --epochs 15 --batch-size 64 ...

Resources:
  - PyTorch transfer-learning tutorial for the train/val loop shape.
  - torch.optim (Adam/AdamW), lr schedulers (StepLR / cosine).
  - Weighted CrossEntropyLoss using src.data.class_weights_from_split (trap #1).
  - Save the BEST checkpoint by val metric, not the last epoch.
"""

import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def parse_args() -> argparse.Namespace:
    """TODO: --model {baseline,multiview,multimodal}, --epochs, --batch-size,
    --lr, --out (checkpoint path), --device, seed, etc."""
    raise NotImplementedError


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Standard loop. Return mean train loss. TODO."""
    raise NotImplementedError


@torch.no_grad  # type: ignore[name-defined]  # (import torch when you implement)
def validate(model, loader, criterion, device) -> dict:
    """Return {'loss':..., 'acc':..., 'macro_f1':...}. Macro-F1 matters more
    than accuracy under imbalance — track it for model selection. TODO."""
    raise NotImplementedError


def main() -> None:
    """Wire it together:
      - build datasets/loaders (train+val) and transforms
      - build model from args
      - loss (weighted), optimizer, scheduler
      - loop: train_one_epoch -> validate -> checkpoint-if-best
      - log per-epoch metrics (print is fine; TensorBoard/W&B is a bonus)
      - export best weights to MODELS_DIR / 'best_model.pt'
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
