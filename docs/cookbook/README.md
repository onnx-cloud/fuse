# Cookbook Documentation

Welcome to the Cookbook documentation. Each example includes both beginner-friendly intuition and expert-level details (ops used, shapes, and run/export commands).

These docs live at `docs/cookbook/` and are intended to be beginner-friendly while retaining technical accuracy for advanced users.

Examples and docs (click to view):

- [Autoencoder](autoencoder.md) — tiny identity-style autoencoder with Reshape/MatMul
- [PCA reconstruction](pca_recon.md) — projection + backprojection via MatMul
- [Isolation forest (toy)](isolation_forest.md) — sum of scaled absolute deviations
- [One-class SVM (toy)](one_class_svm.md) — RBF-like exp(-0.5 * ||x-c||^2)
- [DBSCAN core test](dbscan.md) — distance threshold classification (Less / Cast)
- [GMM posterior (toy)](gmm.md) — two-component mixture posterior approximation
- [LOF-inspired score](lof.md) — ratio comparing distance to a reference
- [Elliptic envelope](elliptic_envelope.md) — Mahalanobis-like distance via precision matrix
- [kNN anomaly (avg distance)](knn_anomaly.md) — average distance to centroids
- [Robust covariance (pooled variance)](robust_covariance.md) — mean/variance-based score
- [Autoencoder training notebook and notes](autoencoder_training.md) — step-by-step training + export guide (beginner-friendly)
- [Iris classification (ML tutorial)](14-iris.md) — train a tiny MLP, embed weights in a `.fuse`, export to ONNX and validate

## Advanced building-blocks 🔧

We imagine catalogs of higher-level, reusable blocks that are useful for trustworthy, temporal, multimodal, and robust models. 

Implementations live under `examples/advanced/` as small, documented, `@proof`-tested `.fuse` blocks.

| node Name   | Input Shape / Type | Output Shape / Type  | Behavior / Semantic Intent | Optional Params / Notes  |
| ---- | ----- | ----- | ---- | ----- |
| TemporalFusion  | `f32[seq, features]*n`   | `f32[seq, fused_features]` | Learns dynamic weighting across multiple sequences | attention mask size, fusion axis  |
| MetaResidual | `f32[N, dim], f32[N, dim]`   | `f32[N, dim]`  | Weighted residual addition per feature | gating function type  |
| StochasticDropoutMix  | `f32[N, dim]`   | `f32[N, dim]`  | Structured stochastic path selection   | dropout probability, path count   |
| EmbeddingDiffusion | `f32[N, embed_dim]`   | `f32[N, embed_dim]`  | Spreads info across embeddings via learned adjacency  | adjacency matrix, diffusion steps |
| AdaptivePrecision  | `f32[N, dim]`   | `f32[N, dim]`  | Per-channel precision scaling for efficiency | precision map or scaling factor   |
| Perturbation | `f32[N, dim]`   | `f32[N, dim]`  | Computes outputs under controlled input perturbations | noise type, magnitude |
| ReverseResidual | `f32[N, dim]`   | `f32[N, dim]`  | Subtracts predicted change to estimate alternate trajectory | prediction function   |
| ConditionalRollback   | `f32[N, dim]`   | `f32[N, dim]`  | Selects optimal outcome among multiple branches | branch count, selection criteria  |
| HypotheticalEmbedding | `f32[N, dim]`   | `f32[N, latent_dim]` | Projects inputs into “what-if” latent space  | latent_dim, projection function   |
| CrossModalAttention   | `f32[N, dim1], f32[N, dim2]` | `f32[N, out_dim]` | Attention across modalities   | attention type, output dimension  |
| DynamicLayerSelector | `f32[N, dim], i64[N]` | `f32[N, dim]` | Routes inputs through different layers based on a selector | number of layers |
| SparsePath | `f32[N, dim]` | `f32[N, dim]` | Activates a subset of pathways in a layer (MoE-style) | number of experts, gating mechanism |
| MemoryAugmented | `f32[N, dim], f32[M, mem_dim]` | `f32[N, dim], f32[M, mem_dim]` | Reads from and writes to an external memory matrix | memory size, read/write heads |
| ContrastiveDiffnode | `f32[N, dim], f32[N, dim]` | `f32[N, 1]` | Computes a score based on the difference between two embeddings | distance metric |
| AdaptiveNormalization | `f32[N, C, H, W]` | `f32[N, C, H, W]` | Normalizes features based on a conditional input | conditional vector dimension |
| MultiHorizonForecast | `f32[N, seq_in, dim]` | `f32[N, seq_out, dim]` | Predicts a sequence of multiple future time steps | input/output sequence lengths |
| GraphPropagate | `f32[num_nodes, feat], i64[2, num_edges]` | `f32[num_nodes, feat]` | Aggregates features from neighbors in a graph. Supports k-step propagation via `steps` (iter/Loop) and compile-time `A^k` folding; also supports sparse adjacency lowering (CSR/COO). | aggregation function (sum, mean); `--folds` and `--fold-fold-externalize-mb` |
| FeedbackLoop | `f32[N, dim]` | `f32[N, dim]` | Uses the output of a layer as its own input in a subsequent step | number of feedback steps |
| StochasticEnsemble | `f32[N, dim]` | `f32[N, out_dim]` | Averages predictions from multiple stochastically-selected models | number of models, selection probability |
| ResidualAttention | `f32[N, dim]` | `f32[N, dim]` | Applies attention to a residual connection | attention mechanism |
| ConditionalMix | `f32[N, dim], f32[N, dim], f32[N, 1]` | `f32[N, dim]` | Blends two tensors based on a learned mixing factor | blending function |
| PerturbedDropout | `f32[N, dim]` | `f32[N, dim]` | Applies dropout with a learnable, structured perturbation | perturbation magnitude |
| HypotheticalResidual | `f32[N, dim]` | `f32[N, dim]` | Adds a residual from a hypothetical "what-if" branch | hypothetical branch function |
| MetaScaling | `f32[N, dim]` | `f32[N, dim]` | Learns a dynamic, per-feature scaling factor from metadata | metadata input shape |
| LatentInterpolation | `f32[N, dim], f32[N, dim]` | `f32[N, dim]` | Interpolates between two points in a latent space | interpolation factor |
| Loopnode | `i64, i64, f32[N, dim]` | `f32[N, dim]` | A general-purpose loop for iterative computation | loop-carried dependencies |
| SwitchSelector | `f32[N, dim], i64[N]` | `f32[N, dim]` | A hard switch to select one of many layers (non-differentiable) | number of layers |
| Aggregationnode | `f32[N, ...]*k` | `f32[agg_shape]` | Aggregates multiple tensors into one (e.g., pooling, concat) | aggregation type, axis |
| ParameterGenerator | `f32[N, cond_dim]` | `f32[param_shape]` | Generates the parameters of another layer dynamically | target parameter shape |
| ComparatorDiff | `f32[N, dim], f32[N, dim]` | `f32[N, dim]` | Element-wise comparison and difference | comparison operator |
| GraphRelational | `f32[N, dim], f32[N, N]` | `f32[N, dim]` | Abstract message passing on arbitrary graphs | message/update functions |
| SequenceManipulator | `f32[B, S, D], i64[B, S']` | `f32[B, S', D]` | Reorders, shuffles, or sorts a sequence based on indices | index generation logic |
| NoiseGenerator | `i64[rank]` | `f32[...]` | Generates structured or random noise of a dynamic shape | noise distribution, seed |

> Invariant: add complete `.fuse` reference implementations under `jupyter/cookbook/` with `@proof` checks and a small performance/shape regression test.

Quick tips

- To run the inline tests for a single example: `python -m pytest tests/test_cookbook.py -q -k <keyword>` (e.g., `-k gmm`).
- To export an example to ONNX: `./.venv/bin/fuse onnx -f jupyter/cookbook/<example>.fuse -o onnx/cookbook/<example>.onnx`.

Tutorial

- A short overview of ML concepts for programmers is available at `docs/TUTORIAL.md`.

