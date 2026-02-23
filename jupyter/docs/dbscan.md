# DBSCAN-like Label

Minimal core-point test that labels a sample as core if its squared distance to a center is below a threshold.

## For beginners
- DBSCAN groups points by neighborhood density. This tiny example implements a single core-point check: distance < threshold → core.
- The output is an integer label (0/1) returned as a 1-D tensor.

## For experts
- Ops used: `Sub`, `Mul`, `ReduceSum`, `Less`, `Cast`, `Reshape`.
- Demonstrates mixing comparison ops (`Less`) with casting to integer outputs and sticky typed constants for thresholds.

## Run & export
- Test: `python -m pytest tests/test_cookbook.py -q -k dbscan`  
- Export: `./.venv/bin/fuse onnx -f jupyter/cookbook/dbscan.fuse -o onnx/cookbook/dbscan.onnx`

## Notes
- This shows integrating logic ops with numeric reductions; good starting point for building neighborhood-based filters.