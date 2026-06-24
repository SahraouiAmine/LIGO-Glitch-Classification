"""
Dataset, splits, and transforms.

The two traps from your assignment live here:
  1. Use the PROVIDED train/val/test splits (sample_type column). Don't resplit.
  2. Never let the four duration-images of ONE glitch land in different splits.
     If you build your own grouping for any reason, group by gravityspy_id.

Resources:
  - torch.utils.data.Dataset / DataLoader docs.
  - torchvision.transforms (v2) for resize/normalize/augment.
  - PyTorch "Transfer Learning for Computer Vision" tutorial for the
    ImageNet normalization constants and input sizing your backbone expects.
  - Recall the PNGs include plot axes/labels — crop them out (the Gravity Spy
    docs give pixel bounds). Decide: crop in __getitem__ or once at download.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


# Fill from the dataset docs. O1/O2 had 22 classes; O3 changed the set.
CLASS_NAMES: list[str] = []  # TODO
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class Sample:
    """One classifiable unit. Decide your unit early: single 4s image, or a
    glitch (4 durations) for multi-view. This dataclass should reflect it."""
    glitch_id: str
    label: int
    image_paths: list[Path]      # one path (single-view) or four (multi-view)
    metadata: dict | None = None  # Omicron features, if doing multimodal


class GravitySpyDataset(Dataset):
    """Map-style dataset over staged Gravity Spy samples.

    __init__ args worth having: split ('train'|'val'|'test'), transform,
    a flag for multi-view, a flag for whether to attach metadata.

    TODO:
      - build the sample index by reading the folder tree and/or metadata CSV
      - filter to the requested split via sample_type
      - in __getitem__: load image(s), apply transform, return (x, y) or
        ((image, features), y) for the multimodal case
    """

    def __init__(self, split: str, *, transform=None, multiview: bool = False,
                 with_metadata: bool = False) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def build_transforms(train: bool):
    """Return the transform pipeline.

    TODO:
      - train vs. eval pipelines (augmentation only on train)
      - resize to backbone input, ToTensor, Normalize(ImageNet stats)
      - be conservative with augmentation: flips/rotations can destroy the
        time-frequency morphology that defines a class. Think before you flip.
    """
    raise NotImplementedError


def class_weights_from_split(split: str = "train") -> torch.Tensor:
    """Compute inverse-frequency weights for imbalanced-loss.

    Trap #1 from the assignment: classes are wildly imbalanced. You'll feed
    this into the loss (or use it to reason about sampling). Return a tensor
    of shape [NUM_CLASSES].
    """
    raise NotImplementedError
