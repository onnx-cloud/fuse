# Elliptic Envelope (Mahalanobis-like)

Mahalanobis-like squared distance computed using a precision matrix to produce a scalar anomaly measure.

## For beginners
- This measures how far a point is from the center after accounting for the shape of the distribution (the precision matrix acts like an inverse covariance).
- Higher values indicate points that are less likely under the elliptical model.

## For experts
- Ops used: `Reshape`, `MatMul`, `Reshape`.
- Example demonstrates matrix multiplication sequence: (x-mean)^T * precision * (x-mean) to produce squared Mahalanobis distance.

## Run & export
- Test: `python -m pytest tests/test_cookbook.py -q -k elliptic`  
- Export: `./.venv/bin/fuse onnx -f jupyter/cookbook/elliptic_envelope.fuse -o onnx/cookbook/elliptic_envelope.onnx`

## Notes
- Keep the precision matrix symmetric positive-definite for genuine Mahalanobis behavior; this example uses identity precision to keep results simple.