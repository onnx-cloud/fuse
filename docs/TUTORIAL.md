# ML Concepts for Programmers — A Short Tutorial

This short tutorial introduces core machine learning (ML) concepts for programmers and shows how small ideas map to Fuse code. It is aimed at people who know programming well but are new to ML.

 **Scope**: this tutorial emphasizes unsupervised learning techniques (anomaly detection, clustering, dimensionality reduction); see the [Cookbook](../COOKBOOK.md) for supervised learning and advanced Fuse features. Experts will find links to deeper technical references.


## 1) The basic ingredients

- **Data**: arrays of numbers (features). Example: a vector of sensor readings.
- **Labels**: (optional) ground-truth values for supervised learning (e.g., class ids).
- **Model**: a computational graph mapping inputs to outputs.
- **Loss**: a function that measures how well the model does on labeled data.
- **Training**: adjusting model parameters to minimize loss.
- **Inference**: using a trained model to make predictions on new data.

When working with ML, think of your problem in three phases: (1) *preparation*: collect and format raw data as numeric arrays, (2) *modeling*: define a computational graph that transforms inputs to predictions, and (3) *deployment*: export the trained model to production. Fuse focuses on phases 2–3, letting you write the model graph in a type-safe, deterministic way that compiles directly to ONNX—a portable format supported by many runtime environments (TensorFlow, PyTorch, ONNX Runtime, etc.).

Practical tip: always inspect/visualize features and consider scaling (standardization) before applying distance-based methods.


## 1b) Fuse Language Primer

**What is Fuse?** Fuse is a small, typed domain-specific language that compiles to ONNX graph definitions. It lets you write inference logic cleanly without manually constructing ONNX protobufs.

Instead of writing Python code that calls ONNX operators directly (which is verbose and error-prone), Fuse lets you express your inference graph in a high-level syntax similar to function definitions. The compiler handles type-checking, shape inference, and deterministic lowering to ONNX. This means you get compile-time error detection, predictable performance, and byte-level reproducibility—the same source always produces identical binary outputs.

**Key syntax**):

| Construct | Purpose | Example |
|----|-----|-----|
| `node` | Reusable function (helper, no export) | `fn scale(x) { ... }` |
| `model` | Top-level exportable graph (exported to `.onnx`) | `model detect(x) { ... }` |
| Parameters | Mutable inputs (become ONNX graph inputs) | `param threshold: f32 = 0.5` |
| Constants | Immutable values folded into the graph | `const axis: i64 = 0` |
| Tensors | Typed arrays with shape | `f32[3, 4]` |
| `@proof` | Inline executable test (validates at compile time) | `@proof graph test_score() { ... }` |

**Minimal example**:

```fuse
const AXIS: i64[1] = [0]

fn compute_mean(x: f32[3]) -> f32[1] {
  s = ReduceSum(x, AXIS, keepdims@=0)
  Div(s, 3.0)
}

@proof graph test_mean() {
  result = compute_mean([1.0, 2.0, 3.0])
  assert result == [2.0]
}
```

For details on types, shape rules, and lowering semantics, see [SPEC.md](../SPEC.md) § Type & Shape System and § Lowering.


## 2) Supervised vs Unsupervised

- **Supervised**: learning a mapping from inputs → labels (classification/regression).
- **Unsupervised**: finding structure without labels (clustering, dimensionality reduction, anomaly detection).
*Supervised* learning requires labeled training data (e.g., images labeled "cat" or "dog"). You use that data to adjust model parameters so predictions match labels. *Unsupervised* learning has no labels—instead, you discover patterns, group similar items, or identify outliers. For example, clustering algorithms group customers by purchase behavior without pre-defined categories, and anomaly detectors flag unusual patterns in sensor data without knowing what "unusual" looks like beforehand.
This tutorial focuses on unsupervised techniques. Cookbook examples demonstrate both: see [jupyter/cookbook/](../../jupyter/cookbook/) for supervised learning (classification via pretrained models like FinBERT) and more advanced topics.


## 4) Dimensionality reduction (PCA)

Intuition: project input to a lower-dimensional basis that preserves variance, then reconstruct.

In high-dimensional data (e.g., images with millions of pixels, or sensor arrays with hundreds of readings), many features may be redundant or correlated. Dimensionality reduction compresses data while retaining the most important structure. **PCA (Principal Component Analysis)** finds the directions (principal components) where data varies most, then projects data onto those directions. You can then either use the compressed representation for downstream tasks, or reconstruct it to denoise—if a point deviates significantly when reconstructed, it may be an anomaly. For novices: think of PCA as finding the "best angle" to view your data so you lose the least information.

Fuse snippet (from [jupyter/cookbook/pca_recon.fuse](../../jupyter/cookbook/pca_recon.fuse)):

```fuse
# Project -> Reconstruct using an identity projection (toy example)
fn pca_reconstruct(x: f32[3]) -> f32[3] {
  P: f32[3,3] = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]
  shape_row: i64[2] = [1, 3]
  shape_vec: i64[1] = [3]
  A = Reshape(x, shape_row)
  Y = MatMul(A, P)
  Recon = MatMul(Y, Transpose(P))
  Reshape(Recon, shape_vec)
}
```

Why it helps: lower-dim representations can reduce noise and make downstream tasks simpler.


## 5) Autoencoders for anomaly detection

Idea: train a network to reconstruct normal data well; anomalies have higher reconstruction error.

An **autoencoder** is a neural network with a bottleneck: it compresses inputs to a low-dimensional latent code, then reconstructs them. If trained on normal data, it learns to reconstruct normal examples well but struggles with anomalies (producing high reconstruction error). This makes autoencoders useful for detecting unusual patterns: compute the reconstruction error for new data—high error suggests an anomaly. Intuitively, autoencoders learn a compact representation of "normalcy"; anything too different gets garbled. For novices: imagine training a photocopier on only high-quality photos; copies of low-quality or fake photos will look distorted, revealing the anomaly.

Loss example (reconstruction error as an ONNX subgraph):

```fuse
# Compute reconstruction loss using an existing autoencoder function
shape_row: i64[2] = [1, 3]
shape_one: i64[1] = [1]

fn recon_loss(x: f32[3]) -> f32[1] {
  out = autoencoder(x)
  diff = Sub(out, x)
  sq = Mul(diff, diff)
  axes0: i64[1] = [0]
  s = ReduceSum(sq, axes0, keepdims@=0)
  Reshape(s, shape_one)
}
```

Note: Training a model requires an optimizer; Fuse examples focus on inference graphs. To train, you would compute the loss in Python and update parameters via gradient-based steps.


## 6) Probabilistic models (GMM)

A Gaussian Mixture Model scores points by how well they fit different components and yields posterior probabilities. It's good for soft clustering and anomaly scoring via likelihoods.

A **GMM** models data as a mixture of several bell-curve-shaped (Gaussian) distributions. Instead of assigning each point to a single cluster, GMM computes probabilities: how likely is this point to belong to cluster 1, cluster 2, etc.? This "soft" assignment is useful for anomaly detection—points far from all clusters have low likelihood, flagging them as anomalies. Intuitively, GMM answers: "Given the patterns I've learned, how surprising is this new data point?" Rare or surprising points are likely anomalies. For novices: think of GMM as assuming your data comes from K different populations (with different characteristics), and scoring new points based on how well they fit those populations.

Cookbook snippet (posterior for a toy 2-component GMM): see `gmm.fuse` (uses `Exp`, `ReduceSum`, and normalization via `Div`).

Practical detail: numeric stability matters—do computations in float32 and be mindful of normalization constants.


## 7) Distance & density based methods (kNN, LOF, DBSCAN)

- **kNN**: distance to neighbors — simpler and effective for small-ish datasets.
- **LOF**: ratio comparing local reachability vs neighbor reachability — captures local density anomalies.
- **DBSCAN**: density-based clustering — identifies dense regions and marks sparse points as noise/outliers.

These methods use proximity as a signal: normal points cluster near similar points, while anomalies are isolated or in low-density regions. **k-NN** is the simplest—check how close a new point is to its k nearest neighbors; if very distant, it's anomalous. **LOF** is more sophisticated: it compares a point's local density to its neighbors' densities; a point in a sparse region surrounded by dense clusters is anomalous even if it's not globally far away. **DBSCAN** goes further, grouping points by connectivity in high-density regions and marking isolated points as outliers. Intuitively, kNN asks "Is this point isolated?", LOF asks "Is this point anomalously isolated relative to its neighbors?", and DBSCAN asks "Does this point belong to a dense cluster?" For novices: kNN is like asking "Am I surrounded by similar people?"; if yes, you're normal; if no, you're an outlier.

Fuse-friendly note: these are composed of primitive ops (Sub, Mul, ReduceSum, Less, Cast) so they map cleanly to ONNX.


## 8) Evaluation for anomaly detection

No single obvious metric — common practices:
- If labeled anomalies available, measure precision/recall or ROC-AUC.
- If not, validate using synthetic anomalies or manual inspection.
- Choose thresholds based on desired false-positive rate and operational constraints.

Once you've built an anomaly detector, how do you know if it works? Evaluation depends on what you have: if you have ground-truth labels (known anomalies), compute standard metrics like **precision** (of detected anomalies, how many are true anomalies?), **recall** (of true anomalies, how many did you find?), or **ROC-AUC** (how well does the detector rank anomalies above normal points?). If no labels exist, validate manually on sampled detections or inject synthetic anomalies (e.g., corrupted data) and check if your detector flags them. Choose your decision threshold based on operational needs: a strict threshold catches more anomalies (high recall) but may raise false alarms (low precision), while a lenient threshold minimizes false alarms but may miss real anomalies. For novices: think of precision vs. recall as a dial—push toward one end and the other suffers; your domain constraints (cost of false alarms vs. cost of missed anomalies) determine where to set the dial.


## 9) From concepts to code: an anomaly scoring model

Building on the earlier L1 distance idea, here's a more complete example—a parameterized anomaly detector that takes a dataset center and threshold:

```fuse
# Parameterized L1-style anomaly detector
const AXIS: i64[1] = [0]

model l1_anomaly(
  x: f32[3],
  param center: f32[3],
  param scale: f32 = 1.0
) -> f32[1] {
  diff = Abs(Sub(x, center))
  total = ReduceSum(diff, AXIS, keepdims@=0)
  scaled = Mul(total, scale)
  shape_out: i64[1] = [1]
  Reshape(scaled, shape_out)
}

@proof graph test_detector() {
  # Test 1: normal point (close to center)
  result1 = l1_anomaly([0.1, 0.0, 0.1], center@=[0.0, 0.0, 0.0], scale@=1.0)
  assert result1 == [0.2]
  
  # Test 2: outlier (far from center)
  result2 = l1_anomaly([10.0, 10.0, 10.0], center@=[0.0, 0.0, 0.0], scale@=1.0)
  assert result2 == [30.0]
}
```

This example demonstrates:
- **Parameters**: `center` and `scale` become graph inputs (can be adjusted at inference time without recompiling).
- **Model**: top-level exportable function with concrete signature.
- **Inline tests**: `@proof` validates asserted outputs at compile time; helps catch errors early.

To export to ONNX and inspect the graph:

```bash
fuse onnx -f path/to/file.fuse -o output_dir/
# Produces: output_dir/l1_anomaly.onnx
python -c "import onnx; m = onnx.load('output_dir/l1_anomaly.onnx'); onnx.checker.check_model(m); print('Valid!')"
```

For more complete examples, see [jupyter/cookbook/knn_anomaly.fuse](../../jupyter/cookbook/knn_anomaly.fuse), [lof.fuse](../../jupyter/cookbook/lof.fuse), and [isolation_forest.fuse](../../jupyter/cookbook/isolation_forest.fuse).


## 10) Advanced features: control flow and external imports

**Conditional logic (if/else)**: Fuse supports branching to implement adaptive inference logic. In real inference pipelines, you often want different behavior based on intermediate results—for example, apply aggressive filtering if a preliminary score is high, or route to different downstream models. Fuse allows this with conditional operators like `If` and `Where`, enabling adaptive, multi-path inference graphs.

For example, threshold-based routing:

```fuse
model adaptive_detector(x: f32[3], threshold: f32) -> f32[1] {
  score = l1_score(x)  # from earlier
  is_high = Greater(score, threshold)
  # Use Where() to conditionally scale or return different values
  high_val: f32[1] = [1.0]
  low_val: f32[1] = [0.0]
  Where(is_high, high_val, low_val)
}
```

See [examples/golden/control_flow.fuse](../../examples/golden/control_flow.fuse) for full control flow examples.

**Importing external ONNX models**: You can wrap pre-trained models (BERT, image classifiers, etc.) and compose them with Fuse logic using `@import`. This enables modular pipelines combining production models with custom inference. In practice, you often have pre-trained models from external sources (e.g., Hugging Face, TensorFlow Hub) that you want to reuse without modification. Rather than reimplementing them in Fuse, you can load them as callables and combine them with custom Fuse logic—e.g., encode text with FinBERT, then apply a custom anomaly scoring function. This composition reduces code duplication and lets different teams own different components.

```fuse
@import("../onnx/bert_encoder.onnx")
fn encode(text: i64[...]) -> f32[...]

model classify_and_score(text: i64[...]) -> f32[...] {
  embeddings = encode(text)
  # Apply Fuse logic to embeddings (e.g., compute anomaly score)
  score = l1_anomaly(embeddings, center@=[...], scale@=1.0)
  score
}
```

See [jupyter/cookbook/finbert.fuse](../../jupyter/cookbook/finbert.fuse) for a real FinBERT integration example, and [SPEC.md](../SPEC.md) § Modules & Imports for full syntax details.


## 11) Graduated next steps: from tutorial to cookbook

Having covered core ML concepts and Fuse syntax, you're ready to explore concrete examples. The roadmap below shows how tutorial topics map to cookbook examples, organized by difficulty. Start at your comfort level and explore related files incrementally—each example includes comments and tests documenting the approach.

Here's a roadmap to deeper examples organized by difficulty and topic:

| Concept | Tutorial Section | Example File | Difficulty |
|-----|-----|-----|----|
| **Language basics** | §1b | [examples/golden/params_consts.fuse](../../examples/golden/params_consts.fuse) | Beginner |
| **Simple operations** | §1b | [examples/golden/algebraic.fuse](../../examples/golden/algebraic.fuse) | Beginner |
| **Dimensionality reduction** | §4 | [jupyter/cookbook/pca_recon.fuse](../../jupyter/cookbook/pca_recon.fuse) | Beginner |
| **Autoencoders** | §5 | [jupyter/cookbook/autoencoder.fuse](../../jupyter/cookbook/autoencoder.fuse) | Beginner–Intermediate |
| **kNN anomaly scoring** | §7, §9 | [jupyter/cookbook/knn_anomaly.fuse](../../jupyter/cookbook/knn_anomaly.fuse) | Intermediate |
| **LOF (Local Outlier Factor)** | §7 | [jupyter/cookbook/lof.fuse](../../jupyter/cookbook/lof.fuse) | Intermediate |
| **DBSCAN clustering** | §7 | [jupyter/cookbook/dbscan.fuse](../../jupyter/cookbook/dbscan.fuse) | Intermediate |
| **Isolation Forest** | §8 | [jupyter/cookbook/isolation_forest.fuse](../../jupyter/cookbook/isolation_forest.fuse) | Intermediate–Advanced |
| **One-class SVM** | §8 | [jupyter/cookbook/one_class_svm.fuse](../../jupyter/cookbook/one_class_svm.fuse) | Advanced |
| **Attributes & metadata** | — | [jupyter/cookbook/attributes.fuse](../../jupyter/cookbook/attributes.fuse) | Intermediate |
| **Control flow** | §10 | [examples/golden/control_flow.fuse](../../examples/golden/control_flow.fuse) | Intermediate |
| **External ONNX integration** | §10 | [jupyter/cookbook/finbert.fuse](../../jupyter/cookbook/finbert.fuse) | Advanced |

Pick an example matching your current level and difficulty, examine the comments and `@proof` cases, try running it with `fuse test -f <file>`, then modify it slightly (e.g., change parameter values, add a test case). This hands-on approach builds intuition faster than reading alone. As you gain confidence, tackle the next difficulty level or explore related techniques.


## 12) Practical tips & further reading

**Implementation best practices**:
- Normalize or standardize input features for distance-based methods.
- Always validate on held-out data; be careful of data leakage between training and validation.
- Test models with inline `@proof` blocks during development; catch errors before compilation.
- Use `@proof` not just for final correctness, but for intermediate calculations (e.g., test that your normalization step produces asserted values)—this makes debugging easier.

**Working with Fuse files**:
- Parse a `.fuse` file and print AST: `fuse ast -f file.fuse`
- Compile to ONNX: `fuse onnx -f file.fuse -o output_dir/` (produces `.onnx` files)
- Validate ONNX graph: `python -c "import onnx; onnx.checker.check_model(onnx.load('file.onnx'))"`
- Run inline tests: `fuse test -f file.fuse`

See [CLI.md](CLI.md) for comprehensive CLI documentation with examples and workflows.

**Helpful references**:
- [SPEC_REFERENCE.md](SPEC_REFERENCE.md): Language spec overview and quick links
- [SPEC.md](../SPEC.md): Full specification (grammar, type/shape rules, lowering semantics)
- [CLI.md](CLI.md): Command-line tool reference with workflows
- [COOKBOOK.md](../COOKBOOK.md): Recipe-based examples organized by Fuse features
- Bishop, *Pattern Recognition and Machine Learning*
- Goodfellow, Bengio, Courville, *Deep Learning*
- ONNX [operator documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md) for detailed semantics

**Next level**: Once comfortable with Fuse syntax, explore the [SPEC.md](../SPEC.md) for advanced features (symbolic dimensions, type aliases, module system), or read [FUSE_TO_ONNX.md](../FUSE_TO_ONNX.md) to understand lowering details.
