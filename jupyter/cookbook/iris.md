# Iris classification (Cookbook)

This short, self-contained example shows how to go from the small `jupyter/data/iris.csv` dataset to a deterministic Fuse model (`iris.fuse`) and an ONNX export. The model implemented here is a simple **nearest-centroid** classifier (ArgMin of L2 distances to three centroids). It is compact, deterministic, and easy to inspect.

## Files
- `jupyter/data/iris.csv` — canonical dataset (included with the repo).
- `jupyter/cookbook/iris.fuse` — Fuse model implementing the centroid classifier with embedded centroids and `@proof` checks for the sample rows.
- `onnx/cookbook/iris.onnx` — recommended export target.
- `jupyter/cookbook/iris.ipynb` — a small notebook that demonstrates how to inspect the data, export the model and run inference with `onnxruntime`.

## Quick steps
1. Inspect the dataset:

```
python -c "import pandas as pd; print(pd.read_csv('jupyter/data/iris.csv').head())"
```

2. Export the Fuse model to ONNX:

```
./.venv/bin/fuse onnx -f jupyter/cookbook/iris.fuse -o onnx/cookbook/iris.onnx
```

3. Validate the exported ONNX model:

```
python -c "import onnx; m=onnx.load('onnx/cookbook/iris.onnx'); onnx.checker.check_model(m); print('OK')"
```

4. Run inference with `onnxruntime` (example in the notebook): the ONNX model accepts a single input named `x` (shape `f32[4]`) and returns a predicted label `i64[1]`.

## Notes
- The centroids are chosen to classify the small example CSV shipped under `jupyter/data/` and to make the example deterministic and easy to test via the `@proof` fn embedded in the Fuse model.
- For a full ML workflow (training, validation), use `examples/cookbook/iris_train.py` instead; this example is intentionally lightweight and readable for teaching purposes.
