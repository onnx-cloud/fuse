# Cookbook examples

This folder showcases small anomaly-detection and dimensionality-reduction workflows implemented directly in Fuse. Each example stays within ONNX opset 18 and carries an inline `@proof` so that the cookbook test suite can verify the outcomes deterministically.

Runnable examples
- `autoencoder.fuse` — identity-like linear autoencoder for three-element vectors.
- `pca_recon.fuse` — PCA reconstruction that projects and reprojects using an identity basis.
- `isolation_forest.fuse` — isolation-forest inspired score using `ReduceSum` to aggregate deviations.
- `one_class_svm.fuse` — simplified RBF decision function built with `Exp` and `ReduceSum`.
- `dbscan.fuse` — distance-threshold classifier that mimics a DBSCAN core-point test.
- `gmm.fuse` — two-component Gaussian posterior that uses `Exp` and mixture normalization.
- `lof.fuse` — local-outlier-factor style ratio that compares sample reachability to a reference.
- `elliptic_envelope.fuse` — Mahalanobis-like distance formed by a precision matrix.
- `knn_anomaly.fuse` — average distance to two fixed neighbors to emulate a kNN anomaly score.
- `robust_covariance.fuse` — sample-to-mean deviation normalized by a toy covariance estimate.
- `type_aliases.fuse` — type aliases and typed constants (moved from `examples/showcase`).
- `symbolic_dims.fuse` — symbolic dimensions and reduce/reshape examples (moved from `examples/showcase`).
- `golden.fuse` — example showing a small function with inline `@proof` (moved from `examples/showcase`).
- `quantize.fuse` — **moved to** `examples/golden/quantize.fuse` (demonstrates `@quantize` / `@dequantize` pragmas; requires opset >= 19 for integer quantization support).
- `dtypes.fuse` — shows small functions returning typed constants for a broad set of scalar dtypes (f32, f64, i64, i32, …, bf16, f16).
- `namespacing.fuse` — note demonstrating domain-by behavior (moved from `examples/showcase`).

Notes
- These examples rely on ONNX operators such as `ReduceSum`, `ReduceMean`, `Less`, `Cast`, and `Exp`. They are validated for opset 18 in the cookbook test harness.
- Drop-in these snippets to see how Fuse can describe simple scoring heuristics without pulling in prepared ONNX models.

Detailed docs
- For step-by-step notes, beginner-friendly explanations, and expert details for each example, see the docs in `docs/cookbook/` or open `docs/cookbook/README.md`.  
- The docs include run/export instructions and the ONNX ops each example exercises.
