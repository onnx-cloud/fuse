# GMM Posterior (Toy)

Two-component Gaussian-style posterior approximation using unnormalized scores.

## For beginners
- This example computes two Gaussian-like scores (centered at different means) and normalizes them so you get an approximate posterior for component B.
- It demonstrates how mixtures combine and are normalized: it’s like asking "how likely is this point to belong to cluster B vs A?".

## For experts
- Ops used: `Sub`, `Mul`, `ReduceSum`, `Exp`, `Add`, `Div`, `Reshape`.
- Shows numerically stable-ish float32 work and why small changes to arithmetic (e.g., extra +1 in denom) affect the final numeric result — the inline test keeps an ONNX-realistic float32 asserted value.

## Run & export
- Test: `python -m pytest tests/test_cookbook.py -q -k gmm`  
- Export: `./.venv/bin/fuse onnx -f jupyter/cookbook/gmm.fuse -o onnx/cookbook/gmm.onnx`

## Notes
- Useful when experimenting with mixture models, normalization constants, and numerics.