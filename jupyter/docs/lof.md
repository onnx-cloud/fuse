# LOF-inspired Score

Local Outlier Factor-style ratio comparing sample distance to a reference sample.

## For beginners
- LOF compares the reachability of a point to that of its neighbors. Here we compare the sample distance to a fixed reference to illustrate the ratio idea.

## For experts
- Ops used: `Sub`, `Abs`, `ReduceSum`, `Div`, `Reshape`.
- The code uses explicit axes and shape INT64 tensors and returns a scalar score.

## Run & export
- Test: `python -m pytest tests/test_cookbook.py -q -k lof`  
- Export: `./.venv/bin/fuse onnx -f jupyter/cookbook/lof.fuse -o onnx/cookbook/lof.onnx`

## Notes
- The recipe is intentionally simple; real LOF uses kNN reachability distances and local densities, but the ratio captures the intuition succinctly.