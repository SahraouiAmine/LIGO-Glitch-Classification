"""
The deployable inference core. Imports ONLY src.model from the project.
No training, no data-loading, no sklearn. If this module needs the dataset
or train.py to run, the train/serve boundary is broken — fix it.

Resources:
  - Same preprocessing transform you used at eval time (resize/normalize).
    Factor it so train and serve can't drift apart.
  - pytorch-grad-cam library for the saliency overlay.
  - Apply your fitted temperature T here so the served confidences are calibrated.
"""

from pathlib import Path
from PIL import Image
import torch


class GlitchPredictor:
    """Loads weights once, predicts on demand. Designed for app.py to hold one
    instance for the process lifetime."""

    def __init__(self, checkpoint: Path, temperature: float = 1.0,
                 device: str = "cpu") -> None:
        """TODO: rebuild architecture (src.model), load weights, eval(), store
        class names + temperature + the eval transform."""
        raise NotImplementedError

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Crop axes if needed, resize, normalize, add batch dim. TODO."""
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, float]:
        """Return {class_name: calibrated_probability} sorted desc.
        Apply temperature scaling to logits before softmax. TODO."""
        raise NotImplementedError

    def explain(self, image: Image.Image):
        """Return a Grad-CAM overlay (PIL image or array) for the top class.
        Optional but high-value for the demo. TODO."""
        raise NotImplementedError
