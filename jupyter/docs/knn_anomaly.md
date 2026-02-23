# kNN Anomaly (Average Distance)

Computes average distance to a small set of centroids as a simple kNN-like anomaly signal.

## For beginners
- A common anomaly heuristic is "distance to nearest neighbors". This toy example computes distances to two fixed centroids and returns their average.
- Lower average means closer to known data, higher means more anomalous.

## For experts
- Ops used: `Reshape`, `Sub`, `Mul`, `ReduceSum`, `Div`, `Reshape`.
- Demonstrates batch-like broadcasting: centroids are `2x2` and the input is reshaped to `1x2` to compute pairwise differences.

## Run & export
- Test: `python -m pytest tests/test_cookbook.py -q -k knn`  
- Export: `./.venv/bin/fuse onnx -f jupyter/cookbook/knn_anomaly.fuse -o onnx/cookbook/knn_anomaly.onnx`

## Notes
- This is a single-sample scoring function; you can adapt it to compute distances across a dataset by adding batch axes.