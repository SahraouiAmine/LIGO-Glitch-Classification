# LIGO Glitch Classifier

> **Live demo:** <!-- TODO: Hugging Face Space URL goes HERE, at the very top. -->

Classifies transient noise ("glitches") in LIGO data by time–frequency
morphology, using the Gravity Spy O3 dataset. Glitch identification is a core
detector-characterization task — the same kind of data-quality work that, e.g.,
the glitch subtraction behind GW200105 depended on.

<!-- Write these sections AS you build. Don't leave them all for the end. -->

## Results
<!-- TODO: table — your model(s) vs. published baselines.
     Columns: model | overall acc | macro-F1 | ECE (pre/post temp scaling).
     Reference points: classic ensemble ~98.21%; April 2026 arXiv eval (2604.08796). -->

## Data
<!-- TODO: which Zenodo records, how to get them (`python data/download.py`),
     class list, split sizes, the axis-crop note. -->

## Method
<!-- TODO: backbone + transfer learning; your Milestone 2 choice (multi-view or
     multimodal) and why; calibration via temperature scaling. -->

## Reproduce
```bash
# TODO once stable:
# pip install -r requirements.txt
# python data/download.py
# python -m src.train --model baseline ...
# python -m src.evaluate --checkpoint models/best_model.pt
# python -m src.calibration
```

## Serving
<!-- TODO: how the Space is built — requirements-serve.txt, inference/app.py,
     where the weights load from (Git LFS / HF Hub). Note the train/serve split. -->

## Repo layout
<!-- TODO: brief tree. Emphasize that inference/ imports only src/model.py. -->

## References
<!-- TODO: Zevin 2017, Glanzer 2023, Guo 2017 (calibration), the Abbott 2021
     NSBH paper for motivation. -->
