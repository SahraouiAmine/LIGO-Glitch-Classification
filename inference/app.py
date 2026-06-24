"""
Gradio app — this is what becomes your live demo URL on Hugging Face Spaces.
A recruiter drags in a spectrogram, sees predicted class + calibrated
confidences (+ optional Grad-CAM). This file IS the deliverable's front door.

Resources:
  - Gradio "Quickstart" and the Image input + Label output components.
  - HF "Spaces overview" and the Gradio-on-Spaces guide (Space needs this file,
    src/model.py, the weights, and requirements-serve.txt).
  - For weights >a few MB: Git LFS, or push to the HF Hub and load by repo id.
"""

import gradio as gr

# from inference.predict import GlitchPredictor
# PREDICTOR = GlitchPredictor(checkpoint=..., temperature=..., device="cpu")


def classify(image):
    """Adapter between the Gradio Image and your predictor.
    TODO: call PREDICTOR.predict(image); return a dict for gr.Label, and
    optionally the Grad-CAM image for a second output."""
    raise NotImplementedError


def build_interface():
    """TODO: gr.Interface (or Blocks) with:
      - input: gr.Image(type='pil')
      - outputs: gr.Label(num_top_classes=5) [+ gr.Image for Grad-CAM]
      - title/description: one line on what this is and the dataset
      - examples: a few sample spectrograms so visitors can click-to-try
    Return the interface object."""
    raise NotImplementedError


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
