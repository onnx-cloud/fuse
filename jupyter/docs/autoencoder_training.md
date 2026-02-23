# Autoencoder training and export

This document summarizes the notebook steps: train a tiny linear autoencoder with NumPy, embed weights into a Fuse snippet, export to ONNX and validate with ReferenceEvaluator.

- Trains a 3->2->3 autoencoder for a few epochs (SGD, MSE loss)
- Writes a Fuse snippet with typed constants
- Exports to ONNX via `fuse onnx`
- Validates model via `onnx.checker` and `onnx.reference.ReferenceEvaluator`

Run the notebook with papermill:

```
papermill docs/cookbook/autoencoder_training.ipynb docs/cookbook/autoencoder_training-executed.ipynb
```

Then convert to Markdown:

```
jupyter nbconvert --to markdown docs/cookbook/autoencoder_training-executed.ipynb --output docs/cookbook/autoencoder_training.md
```
