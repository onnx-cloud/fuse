# Fuse Jupyter Docker - Quick Start Guide 🚀

**One-command deployment** of the complete Fuse ONNX DSL development environment.

---

## ⚡ Quick Start (30 seconds)

```bash
# Build and run in one command
make jupyter

# Open your browser to:
http://localhost:8888
```

That's it! You're ready to write ONNX models in Fuse.

---

## 📋 Prerequisites

- **Docker** (20.10+)
- **Make** (optional but recommended)
- **8GB RAM** minimum
- **2GB disk space**

### Install Docker

```bash
# macOS (Homebrew)
brew install docker

# Linux (Ubuntu/Debian)
sudo apt install docker.io

# Verify
docker --version
```

---

## 🎯 Usage Options

### Option 1: Makefile (Recommended)

```bash
# Build image and start container
make jupyter

# Check status
docker ps | grep fuse

# Stop container
make jupyter-stop

# Access shell
make jupyter-shell

# Clean up
make jupyter-clean
```

### Option 2: Manual Docker Commands

```bash
# Build image
docker build -f docker/jupyter/Dockerfile -t fused:local .

# Run container
docker run --rm -d \
  -p 8888:8888 \
  -v $(pwd):/fused \
  -w /fused \
  --name fuse \
  fused:local

# Stop container
docker stop fuse
```

### Option 3: Docker Compose (Future)

```yaml
# docker-compose.yml (coming soon)
version: '3.8'
services:
  fuse:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - .:/fused
```

---

## 🧪 Verify Installation

Once Jupyter is running, open http://localhost:8888 and:

1. **Check the welcome screen** - Should show "Fuse Welcome" with quick links
2. **Run the welcome notebook** - Click "Welcome Tutorial" and run all cells
3. **Try a simple example**:

```python
# In a new notebook cell
%%fuse
const x = [[1, 2, 3]] : f32[1, 3]
const y = [[4], [5], [6]] : f32[3, 1]
const result = x @ y
```

4. **Test the chat assistant** - Press `Cmd/Ctrl + K` to open Copilot chat
5. **Browse the cookbook** - Open [Cookbook Index](http://localhost:8888/fuse/cookbook)

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Check what's using port 8888
lsof -i :8888

# Use a different port
docker run --rm -d -p 9999:8888 ... fused:local

# Then open http://localhost:9999
```

### Build Fails - Node/npm Issues

```bash
# Clear Docker build cache
docker builder prune -a

# Rebuild with no cache
docker build --no-cache -f docker/jupyter/Dockerfile -t fused:local .
```

### Container Won't Start

```bash
# Check logs
docker logs fuse

# Common issues:
# 1. Port conflict (see above)
# 2. Permission issues (try: sudo chown -R $USER:$USER .)
# 3. Out of disk space (docker system prune)
```

### Extension Not Loading

```bash
# Shell into container
make jupyter-shell

# Check extension is registered
jupyter labextension list | grep fuse

# Rebuild extension
cd /fused/jupyter/labextensions/fuse
npm install
npm run build
jupyter lab build

# Restart container
exit
make jupyter-stop
make jupyter-start
```

### Python Module Import Errors

```bash
# Verify PYTHONPATH in container
make jupyter-shell
echo $PYTHONPATH  # Should include /fused

# Check src package
python -c "import src.jupyter.server; print('OK')"

# Reinstall if needed
pip install -e /fused
```

---

## 🏗️ Build Options

### Production Build (Minimal Size)

```bash
# Remove examples and run notebooks
docker build \
  --build-arg CLEAN_EXAMPLES=1 \
  --build-arg RUN_NOTEBOOKS=1 \
  -f docker/jupyter/Dockerfile \
  -t fused:production .
```

### Development Build (Fast Iteration)

```bash
# Mount notebooks as volume for instant changes (no rebuild!)
docker run --rm -d \
  -p 8888:8888 \
  -v $(pwd)/jupyter/notebooks:/fused/jupyter/notebooks \
  -v $(pwd)/examples:/fused/examples \
  -e FUSE_DEBUG=1 \
  --name fuse-dev \
  fused:local

# Edit notebooks on host - changes reflect immediately in container
```

**💡 Pro Tip:** The Dockerfile is optimized for caching. Notebook changes trigger only a 10-15 second rebuild instead of 5 minutes! See [CACHING.md](CACHING.md) for details.

### Multi-Architecture Build

```bash
# Build for ARM64 (Apple Silicon) and AMD64
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/jupyter/Dockerfile \
  -t fused:multi .
```

---

## 🎓 Next Steps

### For New Users

1. **Complete the Interactive Tutorial** - [Open tutorial.ipynb](http://localhost:8888/lab/tree/jupyter/notebooks/interactive_tutorial.ipynb)
2. **Browse the Cookbook** - 63 recipes covering common patterns
3. **Read the Documentation** - [Full docs hub](http://localhost:8888/fuse/docs)

### For Developers

1. **Review the Architecture** - See [ARCHITECTURE.md](../../jupyter/docs/README.md#architecture)
2. **Explore the API** - See [API Reference](../../jupyter/docs/README.md#api-reference)
3. **Check Examples** - Browse [examples/golden/](../../examples/golden/)

### For Contributors

1. **Read CONTRIBUTING.md** - [Contributing guide](../../CONTRIBUTING.md)
2. **Setup Development** - `make setup && source .venv/bin/activate`
3. **Run Tests** - `make test-all`

---

## 📊 What's Inside?

The Docker image includes:

- **JupyterLab 4.x** - Modern notebook interface
- **Fuse Extension** - Custom UI widgets (chat, admin, error cards)
- **Python 3.11** - Latest stable Python
- **ONNX Runtime** - Fast model execution
- **221 ONNX Operators** - Full operator coverage
- **Completion Provider** - Context-aware autocomplete
- **LLM Integration** - Copilot chat assistant (optional)
- **Tutorial Notebooks** - 7-lesson interactive tutorial
- **63 Cookbook Recipes** - Common patterns and examples

**Total Size:** ~2.5 GB (optimized layers)
**Build Time:** ~3-5 minutes (cached: ~30 seconds)
**Memory Usage:** ~800 MB idle, ~2 GB active

---

## 🔒 Security Notes

### Default Configuration

- **No authentication** by default (localhost only)
- **Token disabled** for convenience
- **CORS open** to localhost
- **LLM admin enabled** without password

### Production Recommendations

```python
# jupyter_config.py (add authentication)
c.ServerApp.token = 'your-secure-token-here'
c.ServerApp.password_required = True
c.ServerApp.allow_origin = 'https://yourdomain.com'

# Environment variables
export FUSE_LLM_ADMIN_PASSWORD="strong-password"
export FUSE_LLM_REQUIRE_AUTH=1
```

---

## 🌐 Advanced Usage

### Remote Deployment

```bash
# Deploy to cloud VM
ssh user@remote-host
git clone https://github.com/yourorg/fuse.git
cd fuse
make jupyter

# Access via SSH tunnel
ssh -L 8888:localhost:8888 user@remote-host
# Open http://localhost:8888
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml (example)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fuse-jupyter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fuse
  template:
    metadata:
      labels:
        app: fuse
    spec:
      containers:
      - name: fuse
        image: fused:local
        ports:
        - containerPort: 8888
        volumeMounts:
        - name: workspace
          mountPath: /workspace
```

### CI/CD Integration

```yaml
# .github/workflows/jupyter-test.yml
name: Jupyter Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Jupyter image
        run: make jupyter-image
      - name: Run tests in container
        run: docker run fused:local pytest tests/
```

---

## 🆘 Support

- **Documentation:** [docs/README.md](../../jupyter/docs/README.md)
- **Issues:** [GitHub Issues](https://github.com/yourorg/fuse/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourorg/fuse/discussions)
- **Chat:** [Discord Server](https://discord.gg/fuse)

---

## ✅ Success Checklist

After following this guide, you should be able to:

- [ ] Build the Docker image successfully
- [ ] Access Jupyter at http://localhost:8888
- [ ] See the Fuse welcome screen
- [ ] Run cells in the welcome notebook
- [ ] Create a new notebook and write Fuse code
- [ ] Use `%%fuse` magic commands
- [ ] Access the chat assistant (Cmd+K)
- [ ] Browse the cookbook
- [ ] View ONNX operator documentation

---

**Ready to dive deeper?** Check out:
- 📚 [Full Documentation](../../jupyter/docs/README.md)
- 🍳 [Cookbook Index](../../jupyter/cookbook/INDEX.md)
- 📖 [Tutorial Notebook](../../jupyter/notebooks/interactive_tutorial.ipynb)
- 📋 [API Reference](../../jupyter/docs/README.md#api-reference)

---

*Build time: ~5 minutes | Total size: ~2.5 GB | Memory: ~2 GB*
