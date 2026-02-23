# Isolation-Forest Inspired Score

Computes a simple anomaly score by summing scaled absolute deviations from a center.

## For beginners
- The example picks a center and computes how far a sample is from that center (elementwise), scales the deviations, sums them, and turns that into a small score.
- This mimics ideas from isolation/outlier detection: larger deviation → larger score.

## For experts
- Ops used: `Sub`, `Abs`, `Div`, `ReduceSum`, `Add`, `Reshape`.
- We use an INT64 axes tensor to drive `ReduceSum` (compatible with ONNX opset signatures) and a shape tensor for `Reshape` to produce a 1-D output.

## Run & export
- Test: `python -m pytest tests/test_cookbook.py -q -k isolation`  
- Export: `./.venv/bin/fuse onnx -f jupyter/cookbook/isolation_forest.fuse -o onnx/cookbook/isolation_forest.onnx`

## Notes
- This is purposely simple and deterministic; in realistic isolation-forest workflows you would build a forest of trees, but the scoring idea is the same: produce a scalar anomaly score per sample.