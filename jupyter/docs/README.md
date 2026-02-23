# Fuse Jupyter Documentation Hub 📚

Welcome to the comprehensive documentation for **Fuse Jupyter** — the interactive notebook environment for the Fuse ONNX DSL.

## 📖 Quick Links

- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Tutorials](#tutorials)
- [Cookbook](#cookbook)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Getting Started

### Installation

```bash
# Install Fuse with Jupyter support
pip install fuse[jupyter]

# Or install from source
git clone https://github.com/yourorg/fuse.git
cd fuse
pip install -e ".[jupyter]"
```

### Quick Start

1. **Start Jupyter Lab**
   ```bash
   jupyter lab
   ```

2. **Load Fuse Extension**
   ```python
   %load_ext src.jupyter.magics
   ```

3. **Write Your First Model**
   ```python
   %%fuse
   param x: f32[3]
   const y = [1.0, 2.0, 3.0]
   output z = x + y
   ```

4. **Run It**
   ```python
   result = _.run({'x': [10.0, 20.0, 30.0]})
   print(result['z'])  # [11, 22, 33]
   ```

### Docker Quick Start

```bash
make jupyter  # Starts Jupyter in Docker
```

Access at: `http://localhost:8888`

---

## Architecture

### Components Overview

```
┌─────────────────────────────────────────────────┐
│  JupyterLab Frontend                            │
│  ┌─────────────┐  ┌──────────────┐            │
│  │ Chat Widget │  │ Admin Widget │            │
│  └─────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────┘
                     ↓ HTTP API
┌─────────────────────────────────────────────────┐
│  Jupyter Server Extension                       │
│  ┌──────────────────────────────────────────┐  │
│  │ Tornado Handlers                         │  │
│  │ • /fuse/api/ops                          │  │
│  │ • /fuse/api/completions                  │  │
│  │ • /fuse/api/llm                          │  │
│  │ • /fuse/api/llm/admin                    │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  IPython Kernel                                  │
│  ┌──────────────┐  ┌────────────────────┐      │
│  │ Cell Magics  │  │ Exception Handlers │      │
│  └──────────────┘  └────────────────────┘      │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  Fuse Core                                       │
│  Parser → Lowering → ONNX → Runtime             │
└─────────────────────────────────────────────────┘
```

### Key Technologies

- **Frontend**: TypeScript, React 18, esbuild
- **Backend**: Python, Tornado, jupyter_server
- **Core**: ONNX, onnxruntime
- **Build**: Docker multi-stage, npm/pip

---

## API Reference

### IPython Magics

#### `%%fuse` - Cell Magic

Compile Fuse code in a cell and expose as ONNX model.

```python
%%fuse --domain=my_model
param x: f32[10]
output y = Relu(x)
```

**Options:**
- `--domain=NAME` - Set model domain
- `--no-validate` - Skip ONNX validation
- `--externalize` - Write large tensors to external files

**Returns:** `_fuse_model` variable containing `Model` object

#### `%fuse_export` - Line Magic

Export compiled model to `.onnx` file.

```python
%fuse_export my_model output.onnx
```

#### `%fuse_verify` - Line Magic

Validate model structure.

```python
%fuse_verify my_model
```

### REST API Endpoints

#### `GET /fuse/api/ops`

List all available ONNX operators.

**Response:**
```json
["Abs", "Add", "Conv", "MatMul", ...]
```

#### `POST /fuse/api/completions`

Get context-aware code completions.

**Request:**
```json
{
  "prefix": "Mat",
  "context": "output = Mat"
}
```

**Response:**
```json
[
  {
    "label": "MatMul",
    "insertText": "MatMul",
    "kind": "function",
    "detail": "ONNX Op"
  }
]
```

#### `POST /fuse/api/llm`

Query LLM engine for code assistance.

**Request:**
```json
{
  "engine": "openai",
  "messages": [
    {"role": "user", "content": "How do I create a Conv2d layer?"}
  ],
  "stream": false
}
```

**Response:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Use the Conv operator with appropriate parameters..."
    }
  }]
}
```

#### `GET /fuse/api/llm/stream`

Server-Sent Events endpoint for streaming LLM responses.

**Query Parameters:**
- `engine` - LLM engine name
- `message` - User message

**Response:** SSE stream with `data:` prefix

#### `GET /fuse/api/llm/admin`

Get configured LLM engines (admin only).

**Response:**
```json
{
  "openai": {
    "url": "https://api.openai.com/v1/chat/completions",
    "model": "gpt-4",
    "label": "OpenAI GPT-4",
    "secretEnv": "OPENAI_API_KEY"
  }
}
```

#### `POST /fuse/api/llm/admin/:name`

Create or update LLM engine configuration.

**Request Body:**
```json
{
  "url": "https://api.openai.com/v1/chat/completions",
  "model": "gpt-4",
  "secretEnv": "OPENAI_API_KEY",
  "label": "OpenAI GPT-4",
  "prompt": "You are a helpful assistant."
}
```

#### `DELETE /fuse/api/llm/admin/:name`

Delete LLM engine configuration.

### Python API

#### `Model` Class

Wrapper around compiled ONNX models.

```python
from src.jupyter.magics import Model

model = Model("my_model", model_proto)

# Run inference
results = model.run({'x': [1, 2, 3]})

# Display model info
model.show()

# Export to file
model.to_onnx('output.onnx')
```

**Methods:**
- `run(inputs: Dict[str, Any], provider: str = 'reference') -> Dict[str, np.ndarray]`
- `show() -> str` - Display model structure
- `to_onnx(path: str)` - Export to file

---

## Tutorials

### Tutorial 1: Your First Fuse Model

See [interactive_tutorial.ipynb](../notebooks/interactive_tutorial.ipynb) for a hands-on 7-lesson tutorial covering:

1. Loading Fuse magics
2. Basic operations
3. Matrix multiplication
4. Type system
5. Control flow
6. Neural networks
7. Model export

### Tutorial 2: Building a CNN

```python
%%fuse --domain=cnn_example
param input: f32[1, 3, 224, 224]  # NCHW format
param conv1_w: f32[64, 3, 3, 3]
param conv1_b: f32[64]

# First conv layer
conv1 = Conv(input, conv1_w, conv1_b, 
             kernel_shape=[3, 3], 
             strides=[1, 1], 
             pads=[1, 1, 1, 1])
relu1 = Relu(conv1)
pool1 = MaxPool(relu1, kernel_shape=[2, 2], strides=[2, 2])

output features = pool1
```

### Tutorial 3: Using LLM Assistance

1. Open Chat Widget: `Cmd/Ctrl+K`
2. Ask questions:
   - "How do I implement batch normalization?"
   - "What's the difference between Conv and ConvTranspose?"
3. Insert generated code directly into notebook

### Tutorial 4: Custom Training Loops

```python
%%fuse --domain=training
trainable param weights: f32[10, 5]
trainable param bias: f32[5]
param input: f32[?, 10]
param labels: f32[?, 5]

# Forward pass
logits = Add(MatMul(input, weights), bias)
probs = Softmax(logits, axis=1)

# Loss
loss = ReduceMean(
    Neg(Mul(labels, Log(probs)))
)

output predictions = probs
output training_loss = loss
```

---

## Cookbook

### Recipe Index

#### Data Loading
- [Load NPZ weights](../cookbook/01_load_npz.ipynb)
- [Load from HuggingFace](../cookbook/02_huggingface.ipynb)
- [Custom data preprocessing](../cookbook/03_preprocessing.ipynb)

#### Model Patterns
- [ResNet block](../cookbook/10_resnet_block.ipynb)
- [Transformer layer](../cookbook/11_transformer.ipynb)
- [Attention mechanism](../cookbook/12_attention.ipynb)
- [Batch normalization](../cookbook/13_batchnorm.ipynb)

#### Advanced Topics
- [Dynamic shapes](../cookbook/20_dynamic_shapes.ipynb)
- [Control flow](../cookbook/21_control_flow.ipynb)
- [Custom operators](../cookbook/22_custom_ops.ipynb)
- [Quantization](../cookbook/23_quantization.ipynb)

#### Deployment
- [Export for ONNX Runtime](../cookbook/30_export_ort.ipynb)
- [WebAssembly deployment](../cookbook/31_wasm.ipynb)
- [Mobile optimization](../cookbook/32_mobile.ipynb)

---

## Troubleshooting

### Common Issues

#### "Module not found: src.jupyter.magics"

**Solution:**
```bash
pip install -e .  # Install in editable mode
```

#### "ONNX validation failed"

**Solution:**
```python
%%fuse --no-validate
# Your code here
```

Or check for:
- Type mismatches
- Shape incompatibilities
- Unsupported operators

#### "Rate limit exceeded"

**Solution:** Increase `FUSE_LLM_RATE_PER_MIN`:
```bash
export FUSE_LLM_RATE_PER_MIN=120
```

#### "Connection timeout"

**Solution:** Increase `FUSE_LLM_TIMEOUT`:
```bash
export FUSE_LLM_TIMEOUT=60  # seconds
```

#### Kernel crashes

**Diagnostics:**
```python
# Check ONNX Runtime
import onnxruntime as ort
print(ort.__version__)

# Validate model manually
import onnx
onnx.checker.check_model(model_proto)
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourorg/fuse.git
cd fuse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements-dev.txt
pip install -e ".[jupyter]"

# Build frontend
cd jupyter/labextensions/fuse
npm install
npm run build
```

### Running Tests

```bash
# All tests
make test-all

# Jupyter tests only
make test-jupyter

# Specific test file
pytest tests/jupyter/test_completion_provider.py -v
```

### Code Style

- **Python**: Black, isort, flake8
- **TypeScript**: ESLint, Prettier
- **Commits**: Conventional Commits

```bash
# Format code
make format

# Lint
make lint
```

### Submitting PRs

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run `make test-all`
5. Commit: `git commit -m "feat: add new feature"`
6. Push and create PR

---

## FAQ

### Q: Can I use Fuse without Jupyter?

**A:** Yes! Fuse has a CLI:
```bash
fuse compile model.fuse -o model.onnx
```

### Q: Does Fuse support training?

**A:** Yes, mark parameters as `trainable`:
```python
trainable param weights: f32[10, 5]
```

### Q: Can I import existing ONNX models?

**A:** Yes:
```python
@import from "model.onnx" as MyModel
```

### Q: Is GPU acceleration supported?

**A:** Yes, via ONNX Runtime:
```python
model.run(inputs, provider='cuda')
```

### Q: How do I visualize models?

**A:** Use `%fuse_graphviz`:
```python
%fuse_graphviz my_model
```

---

## Resources

### External Links

- [Fuse Specification](../../SPEC.md)
- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [JupyterLab Extensions](https://jupyterlab.readthedocs.io/)

### Community

- GitHub Discussions
- Discord Server
- Stack Overflow Tag: `fuse-onnx`

### Video Tutorials

- [Getting Started (10 min)](https://youtube.com/placeholder)
- [Building CNNs (20 min)](https://youtube.com/placeholder)
- [Advanced Training (30 min)](https://youtube.com/placeholder)

---

## Changelog

See [CHANGELOG.txt](../../CHANGELOG.txt) for version history.

---

## License

See [LICENSE](../../LICENSE) file.

---

**Last Updated:** February 4, 2026  
**Version:** 0.1.0  
**Maintainers:** Fuse Team
