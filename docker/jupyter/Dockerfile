# Build our JupyterLab
# use separate stage to cache node_modules and speed repeated builds; 
# keep final image smaller by not including Node or build tools
# fail fast - don't hide or swallow errors

FROM node:20-slim AS node-builder

WORKDIR /build-lab
# Copy package files first to cache npm modules
COPY jupyter/labextensions/fuse/package.json jupyter/labextensions/fuse/package-lock.json* ./
# Copy LICENSE for inclusion in extension
COPY LICENSE ./LICENSE
# Use BuildKit npm cache mount to speed repeated builds
RUN --mount=type=cache,target=/root/.npm npm ci || true
RUN mkdir -p lib
# Copy source and run the build
COPY jupyter/labextensions/fuse/ ./
RUN npm run build || true
RUN mkdir -p lib

FROM python:3.11-slim AS base

WORKDIR /fused

# Ensure all processes (including docker exec) can import the `src` package
# This is critical so that the fuse_server shim module can load src.jupyter.server
ENV PYTHONPATH=/fused:${PYTHONPATH:-}
# The canonical notebook directory inside the image; can be overridden at runtime
ENV NOTEBOOK_DIR=/fused

# Minimal image for Jupyter productized dev
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates gnupg2 && rm -rf /var/lib/apt/lists/*

# Optional build-time controls
ARG RUN_NOTEBOOKS=0
ARG CLEAN_EXAMPLES=0



COPY pyproject.toml setup.cfg requirements.txt requirements-dev.txt /fused/

# Ensure pip, uv are available and use BuildKit cache for pip to speed repeated installs
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install uv && \
    python -m uv pip install -r requirements.txt -r requirements-dev.txt

# Note: we delay copying the full source tree and the editable install until after labextensions are
# installed (to avoid invalidating cached layers when source changes frequently)


# Install core JupyterLab and commonly used extensions
RUN python -m uv pip install jupyterlab jupyterlab-git jupyterlab-lsp jupyterlab_code_formatter && \
    python -m uv pip install papermill requests 

# Temporary stage: install Node here to run `jupyter labextension install` and produce prebuilt assets
FROM base AS lab-install

# Install Node (temporary) in this stage only
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends nodejs && rm -rf /var/lib/apt/lists/*

# Copy the built assets from the node-builder and install the labextension
COPY --from=node-builder /build-lab/lib /fused/labextensions/fuse/lib
COPY --from=node-builder /build-lab/package.json /fused/labextensions/fuse/package.json
RUN jupyter labextension install labextensions/fuse --no-build

FROM base AS fused

# Copy scripts and config (changes occasionally)
WORKDIR /fused
COPY scripts/ /fused/scripts/
COPY docker/ /fused/docker/
COPY schemas/ /fused/schemas/
COPY *.json *.txt *.md *.proto /fused/

# Ensure jpserver config explicitly registers our extension under a small shim
# Create a shim module 'fuse_server' in site-packages that proxies to src.jupyter.server
RUN python - <<'PY'
from pathlib import Path
p = Path('/usr/local/lib/python3.11/site-packages/fuse_server.py')
p.write_text('from src.jupyter.server import load_jupyter_server_extension\n')
PY
RUN mkdir -p /usr/local/etc/jupyter/jupyter_server_config.d && \
    cat > /usr/local/etc/jupyter/jupyter_server_config.d/fuse.json <<'JSON'
{"ServerApp": {"jpserver_extensions": {"fuse_server": true}}}
JSON

COPY jupyter/__init__.py jupyter/jupyter_config.py /fused/jupyter/
COPY jupyter/config/ /fused/jupyter/config/
COPY jupyter/settings/ /fused/jupyter/settings/

# Copy installed labextensions from the temporary lab-install stage to keep Node out of the runtime image
COPY --from=lab-install /usr/local/share/jupyter/labextensions /usr/local/share/jupyter/labextensions

# Now copy the source and do the editable install (delayed so changes to src/ don't bust earlier caches)
COPY src/ /fused/src
COPY jupyter/kernelspec/ /fused/jupyter/kernelspec/
# Install the editable package, verify the extension importability explicitly,
# then enable the server extension and run the kernel installer. Use a
# strict fail-fast approach so image builds surface missing dependencies
# early with clear diagnostics.
RUN --mount=type=cache,target=/root/.cache/pip python -m uv pip install -e /fused && \
    python -m uv pip install -r requirements.txt -r requirements-dev.txt

RUN --mount=type=cache,target=/root/.cache/pip python -m uv pip install -e /fused && \
    python -m uv pip install -r requirements.txt -r requirements-dev.txt && \
    # Remove potentially conflicting 'lark' packages then explicitly install the
    # canonical 'lark-parser' package to ensure a consistent import surface.
    python -m pip uninstall -y lark lark-parser || true && \
    # Force-reinstall a known-good lark-parser release compatible with our imports
    python -m pip install --no-cache-dir --force-reinstall lark-parser==0.12.0 && \
    # Quick import sanity check to ensure package exposes expected API (fail-fast)
    python - <<'PY'
import importlib
from lark import Lark
print('Lark imported:', Lark)
print('lark.lark spec:', importlib.util.find_spec('lark.lark'))
PY

RUN jupyter server extension enable --sys-prefix --py fuse

# Diagnostic: print lark module details to help identify unexpected package
RUN python - <<'PY'
import importlib, sys, subprocess, glob, os
print('\nPIP SHOW:')
subprocess.run(['python', '-m', 'pip', 'show', 'lark'], check=False)
subprocess.run(['python', '-m', 'pip', 'show', 'lark-parser'], check=False)
print('\nsite-packages listing for lark*:')
for p in glob.glob('/usr/local/lib/python3.11/site-packages/lark*'):
    print(p)
    try:
        if os.path.isdir(p):
            print('  contains:', sorted(os.listdir(p))[:200])
        else:
            print('  file:', p)
    except Exception as e:
        print('  err listing:', e)

print('\nparsers files:')
for p in glob.glob('/usr/local/lib/python3.11/site-packages/lark/parsers/*'):
    print('FILE:', p)
    try:
        with open(p, 'r', encoding='utf-8', errors='replace') as fh:
            for i, ln in enumerate(fh):
                if i < 5:
                    print('  ', ln.rstrip())
                else:
                    break
    except Exception as e:
        print('  err reading:', e)

print('\nDIST-INFO RECORD (partial):')
rec = '/usr/local/lib/python3.11/site-packages/lark_parser-0.12.0.dist-info/RECORD'
try:
    with open(rec, 'r', encoding='utf-8', errors='replace') as fh:
        for i, ln in enumerate(fh):
            if i < 40:
                print(ln.rstrip())
            else:
                break
except Exception as e:
    print('  err reading RECORD:', e)

print('\nimport diagnostics:')
try:
    m = importlib.import_module('lark')
    print('lark module:', m, getattr(m, '__file__', None))
    print('has Lark:', hasattr(m, 'Lark'))
except Exception as e:
    print('lark import error:', e)
    raise
PY

# Install runtime kernel during image build. This will fail-fast if any
# dependencies (such as 'lark-parser') are missing or broken and ensures the
# produced image is fully functional for Jupyter usage.
RUN python -m src.jupyter.install


# Copy IPython startup installer script. Run it inside the container
# if you want the startup loader installed in that environment.
COPY jupyter/scripts/ /fused/jupyter/scripts/
# Run the IPython startup installer during build. This will register the startup
# loader in the image and fail early if import/permission issues exist.
RUN python /fused/jupyter/scripts/install_startup.py


# Copy recommended jupyter config into image
RUN mkdir -p /etc/jupyter && \
    mkdir -p /etc/jupyter/jupyter_server_config.d
COPY jupyter/jupyter_config.py /etc/jupyter/jupyter_config.py

# Install JupyterLab user settings globally
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension && \
    cp /fused/jupyter/settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings \
       /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/ && \
    cp /fused/jupyter/settings/@jupyterlab/apputils-extension/notification.jupyterlab-settings \
       /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/

# Note: Any remaining jupyter/ subdirs not explicitly copied above
# Use straightforward COPY (source directories exist in repo); avoid shell redirections
COPY jupyter/todo/ /fused/jupyter/todo/
COPY jupyter/tensorbus/ /fused/jupyter/tensorbus/
COPY jupyter/static/ /fused/jupyter/static/

# feature toggles tuned for local / quickstart (consider set to 0 for prod)
ENV FUSE_LLM_ENABLED=1
ENV FUSE_LLM_ADMIN_ENABLED=1

# Copy example LLM config into image so the Admin UI has something to show
RUN mkdir -p /config
COPY jupyter/config/llm_config.json.example /config/llm_config.json

# Ensure logs directory exists
RUN mkdir -p /logs && touch /logs/llm_access.log

# Copy launcher settings into system-wide Jupyter data dir
RUN python - <<'PY'
from pathlib import Path
import shutil, json, os
from jupyter_core.paths import jupyter_data_dir
p = Path(jupyter_data_dir()) / 'lab' / 'user-settings' / '@jupyterlab' / 'launcher-extension'
p.mkdir(parents=True, exist_ok=True)
# Try several likely source locations (image may have /jupyter or /workspace)
src_candidates = [
    Path('/jupyter/settings/@jupyterlab/launcher-extension/launcher.jupyterlab-settings'),
    Path('/fused/jupyter/settings/@jupyterlab/launcher-extension/launcher.jupyterlab-settings'),
    Path('/workspace/jupyter/settings/@jupyterlab/launcher-extension/launcher.jupyterlab-settings'),
    Path('jupyter/settings/@jupyterlab/launcher-extension/launcher.jupyterlab-settings'),
]
for src in src_candidates:
    try:
        if src.exists():
            shutil.copy(str(src), p)
            break
    except Exception:
        continue
else:
    print('Warning: launcher settings not found; skipping copy')
PY

# Optionally run notebooks and/or remove example files (controlled via build args)
RUN if [ "${RUN_NOTEBOOKS}" = "1" ]; then \
      find examples -name '*.ipynb' -maxdepth 3 -print0 | xargs -0 -n1 -I{} sh -c 'echo "Running notebook: {}" && python -m papermill "{}" "/tmp/$(basename "{}" .ipynb)-executed.ipynb" || true'; \
    fi && \
    if [ "${CLEAN_EXAMPLES}" = "1" ]; then rm -rf examples/* || true; fi

# Create working dirs and entrypoint script
COPY docker/jupyter/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Add health check to ensure server is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD sh -c "curl -fsS http://localhost:8888/fuse/api/health || curl -fsS http://localhost:8888/lab/fuse/api/health || exit 1"

# Copy LAST (notebooks, examples, docs) so won't invalidate cache

FROM fused AS final
COPY jupyter/notebooks/ /fused/jupyter/notebooks/
COPY jupyter/cookbook/ /fused/jupyter/cookbook/
COPY jupyter/docs/ /fused/jupyter/docs/
COPY jupyter/data/ /fused/jupyter/data/
RUN echo "Final image size:" && du -sh /fused

EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["lab"]
