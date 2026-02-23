#!/usr/bin/env python3
"""Microbench: compare GraphPropagate pow-fold vs Loop implementations.

Produces CSV rows: size,n_nnz,k,method,time_ms

Usage examples:
  ./scripts/bench_graph_propagate.py --sizes 128 256 --ks 1 4 8 --repeat 5
"""

import argparse
import time

import numpy as np

# Prefer running via the project's virtualenv Python if available
try:
    import os, sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parents[1]
    _venv_py = _here / ".venv" / "bin" / "python"
    if _venv_py.exists():
        try:
            if _Path(sys.executable).resolve() != _venv_py.resolve():
                os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
        except Exception:
            pass
except Exception:
    pass

try:
    import onnxruntime as ort
except Exception:
    ort = None

from src.lowering import FuseLowerer

TEMPLATE = """@fuse 0.7
@opset onnx 18
@version 0.7.0
node test(features: f32[{N}, {D}], adj: f32[{N},{N}]) -> f32[{N},{D}] {{
  out = GraphPropagate(features, adj, steps={K}, method="{METHOD}")
}}"""


def make_model_text(N, D, K, method):
    return TEMPLATE.format(N=N, D=D, K=K, METHOD=method)


def run_once(N, D, K, method, folds=8):
    src = make_model_text(N, D, K, method)
    lowerer = FuseLowerer(max_pow_folds=folds)
    model = None
    for name in ("lower_from_string", "lower_from_text", "lower"):
        node = getattr(lowerer, name, None)
        if node:
            model = node(src)
            break
    if hasattr(model, "model"):
        model = model.model
    if ort is None:
        raise RuntimeError("onnxruntime not available; cannot run benchmark")
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    X = np.random.randn(N, D).astype(np.float32)
    # random sparse adjacency approx
    A = (np.random.rand(N, N) < 0.05).astype(np.float32)
    # ensure some connectivity
    if N > 1:
        for i in range(N):
            A[i, i] = 0
            A[i, (i + 1) % N] = 1.0
    # warmup
    sess.run(None, {"features": X, "adj": A})
    t0 = time.time()
    _out = sess.run(None, {"features": X, "adj": A})
    dt = (time.time() - t0) * 1000.0
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 256])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--folds", type=int, default=8)
    args = p.parse_args()

    print("size,k,method,ms")
    for N in args.sizes:
        for K in args.ks:
            for method in ("pow", "iter"):
                times = []
                for _ in range(args.repeat):
                    try:
                        dt = run_once(
                            N,
                            min(64, max(8, N // 4)),
                            K,
                            method,
                            folds=args.folds,
                        )
                        times.append(dt)
                    except Exception as e:
                        print(f"# error running {N},{K},{method}: {e}")
                        break
                if times:
                    print(f"{N},{K},{method},{np.median(times):.2f}")


if __name__ == "__main__":
    main()
