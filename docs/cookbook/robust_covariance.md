# Robust Covariance (Pooled Variance)

A toy pooled variance computation that measures deviation from a mean as a simple covariance-based anomaly signal.

## For beginners
- Compute the mean, subtract it from the sample, square deviations, and average.
- This gives a one-number summary of how variable the sample is relative to a tiny dataset.

## For experts
- Ops used: `ReduceSum`, `Div`, `Sub`, `Mul`, `Reshape`.
- Demonstrates using an `axes` tensor to drive reductions and shape tensors for `Reshape`.

## Run & export
- Test: `python -m pytest tests/test_cookbook.py -q -k robust_covariance`  
- Export: `./.venv/bin/fuse onnx -f jupyter/cookbook/robust_covariance.fuse -o onnx/cookbook/robust_covariance.onnx`

## Notes
- Helps illustrate how statistic computations (means, variances) can be expressed as ONNX graphs and validated quickly with a numeric evaluator.