"""
Model architecture. THIS is the one src/ module inference/ is allowed to import.
Keep it free of training-loop / data-loading imports so the serving unit stays lean.

Resources:
  - torchvision.models for pretrained backbones (resnet18 + ResNet18_Weights).
  - PyTorch transfer-learning tutorial: how to swap the final fc layer.
  - For multi-view/multimodal: nothing exotic — just decide where you fuse
    (concatenate features before the classifier head).
"""

import torch
import torch.nn as nn


def build_baseline(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """ResNet18 with the head replaced for `num_classes`. Your Milestone 1 model.

    TODO:
      - load backbone with ImageNet weights
      - replace .fc with a Linear to num_classes
      - decide freeze strategy (return model ready to train; let train.py
        handle freezing/unfreezing schedule)
    """
    raise NotImplementedError


class MultiViewClassifier(nn.Module):
    """Milestone 2 option A: encode each duration image, fuse, classify.

    Sketch:
      - a shared (or per-view) CNN encoder -> feature vector per view
      - fuse 4 view-features (concat / mean / attention)
      - MLP head -> logits

    TODO: implement __init__ and forward(self, views) where views is a
    stacked tensor [B, num_views, C, H, W].
    """

    def __init__(self, num_classes: int, num_views: int = 4) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MultiModalClassifier(nn.Module):
    """Milestone 2 option B: fuse spectrogram features with Omicron scalars
    (peak frequency, SNR, bandwidth). Physically motivated — e.g. scattered
    light lives at characteristic low frequencies.

    Sketch:
      - CNN encoder for the image -> image features
      - small MLP for the tabular features -> tab features
      - concat -> classifier head

    TODO: implement. Mind feature scaling on the tabular side (normalize).
    """

    def __init__(self, num_classes: int, num_tab_features: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
