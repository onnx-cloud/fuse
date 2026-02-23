"""src.sandbox — runtimes and sandbox orchestration for ONNX-first experiments.

Provides:
- RuntimeAdapter abstraction + Reference/OnnxRuntime adapters
- LocalSandbox (file/ModelProto) and ZooSandbox (load by canonical id)
- best-effort timeout and simple numeric-compare helpers

Design: small, explicit, testable. Timeouts are best-effort (Python-level);
C-level runtimes may not be interruptible — callers should use process isolation
for hard safety requirements.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import onnx


class SandboxError(RuntimeError):
    pass


# ---- Runtime adapters ----
class RuntimeAdapter:
    """Minimal adapter interface for executing ONNX models.

    Implementations must provide `load` and `run`.
    """

    def load(self, model: onnx.ModelProto):
        raise NotImplementedError

    def run(
        self, session: Any, feeds: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class ReferenceAdapter(RuntimeAdapter):
    def load(self, model: onnx.ModelProto):
        try:
            from onnx.reference import ReferenceEvaluator
        except Exception as e:
            raise SandboxError(f"ReferenceEvaluator not available: {e}")
        return ReferenceEvaluator(model)

    def run(
        self, session: Any, feeds: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        # If we have a proto, try to call the evaluator with local (un-prefixed)
        # output names derived from the proto to avoid ReferenceEvaluator
        # mismatches where graph-level names (e.g., 'id.x') don't match the
        # internal node names (e.g., 'x'). If that fails, fall back to asking
        # for all outputs (None) and perform defensive name selection.
        if hasattr(session, "proto_") and getattr(session, "proto_") is not None:
            proto_outs = [o.name for o in session.proto_.graph.output]
            local_outs = [p.split(".", 1)[-1] for p in proto_outs]
            try:
                outs = session.run(local_outs, feeds)
                # Map returned in-order outputs back to canonical proto names
                return {n: np.asarray(v) for n, v in zip(proto_outs, outs)}
            except Exception:
                # fall through to more forgiving behavior
                pass

        outs = session.run(None, feeds)
        # ReferenceEvaluator returns a list in output order. Different ONNX
        # versions expose the output names differently; be forgiving and
        # prefer public attrs when available.
        if hasattr(session, "output_names"):
            out_names = list(session.output_names)
        elif hasattr(session, "output_names_"):
            out_names = list(session.output_names_)
        elif (
            hasattr(session, "proto_")
            and getattr(session, "proto_") is not None
        ):
            out_names = [o.name for o in session.proto_.graph.output]
        elif (
            hasattr(session, "_model")
            and getattr(session, "_model") is not None
        ):
            out_names = [o.name for o in session._model.graph.output]
        else:
            # fallback: produce numeric keys
            out_names = [str(i) for i in range(len(outs))]

        # If output names look invalid (empty names, length mismatch, or
        # mismatch with proto outputs) and we have a proto, prefer the
        # canonical proto output names.
        if (
            hasattr(session, "proto_")
            and getattr(session, "proto_") is not None
        ):
            proto_outs = [o.name for o in session.proto_.graph.output]
            if any(not n for n in out_names) or len(out_names) != len(proto_outs) or set(out_names) != set(proto_outs):
                out_names = proto_outs

        return {n: np.asarray(v) for n, v in zip(out_names, outs)}


class OnnxRuntimeAdapter(RuntimeAdapter):
    def load(self, model: onnx.ModelProto):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise SandboxError(f"onnxruntime not available: {e}")
        # Create an in-memory session from serialized bytes
        try:
            data = model.SerializeToString(deterministic=True)
        except TypeError:
            # Older protobufs may not accept deterministic kwargs; fall back
            data = model.SerializeToString()
        sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        return sess

    def run(
        self, session: Any, feeds: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        # onnxruntime expects name->ndarray mapping and returns list of outputs
        out_names = [o.name for o in session.get_outputs()]
        res = session.run(out_names, feeds)
        return {n: np.asarray(v) for n, v in zip(out_names, res)}


# ---- Sandbox orchestration ----
@dataclass
class RunResult:
    outputs: Dict[str, np.ndarray]
    runtime: str
    duration_s: float


class LocalSandbox:
    """Run a local ONNX ModelProto or .onnx file against available runtimes.

    Behavior: best-effort timeout (Python-level), supports 'reference' and
    'onnxruntime' runtimes. Returns a RunResult with numpy outputs.
    """

    ADAPTERS = {
        "reference": ReferenceAdapter(),
        "onnxruntime": OnnxRuntimeAdapter(),
    }

    def __init__(self, *, allow_external_files: bool = True):
        self.allow_external_files = bool(allow_external_files)

    def _resolve_model(self, model_or_path: Any) -> onnx.ModelProto:
        if isinstance(model_or_path, onnx.ModelProto):
            return model_or_path
        p = Path(model_or_path)
        if not p.exists():
            raise FileNotFoundError(f"model path not found: {model_or_path}")
        return onnx.load(str(p))

    def run(
        self,
        model_or_path: Any,
        feeds: Dict[str, np.ndarray],
        *,
        runtime: str = "reference",
        timeout_s: Optional[float] = None,
    ) -> RunResult:
        if runtime not in self.ADAPTERS:
            raise ValueError(f"unknown runtime: {runtime}")
        model = self._resolve_model(model_or_path)
        adapter = self.ADAPTERS[runtime]
        session = adapter.load(model)

        result: Dict[str, np.ndarray] = {}
        exc: Optional[BaseException] = None

        def _worker():
            nonlocal result, exc
            try:
                start = time.time()
                out = adapter.run(session, feeds)
                dur = time.time() - start
                result = out
                result_meta["duration"] = dur
            except BaseException as e:
                exc = e

        result_meta: Dict[str, float] = {"duration": 0.0}
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            raise TimeoutError(
                "runtime execution exceeded timeout (best-effort)"
            )
        if exc:
            raise SandboxError(f"runtime error: {exc}")

        return RunResult(
            outputs={k: np.asarray(v) for k, v in result.items()},
            runtime=runtime,
            duration_s=float(result_meta["duration"]),
        )

    def bench(
        self,
        model_or_path: Any,
        feeds: Dict[str, np.ndarray],
        *,
        runtime: str = "onnxruntime",
        repeats: int = 5,
    ) -> Tuple[float, float]:
        # returns (median_ms, mean_ms)
        times = []
        for _ in range(max(1, repeats)):
            r = self.run(model_or_path, feeds, runtime=runtime)
            times.append(r.duration_s * 1000.0)
        times = sorted(times)
        mean = float(np.mean(times))
        med = float(np.median(times))
        return med, mean

    def compare(
        self,
        model_or_path: Any,
        feeds: Dict[str, np.ndarray],
        *,
        runtimes: Optional[list[str]] = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> Dict[str, Any]:
        runtimes = runtimes or ["reference", "onnxruntime"]
        results: Dict[str, RunResult] = {}
        for r in runtimes:
            try:
                results[r] = self.run(model_or_path, feeds, runtime=r)
            except Exception as e:
                results[r] = e  # propagate for the report
        # If any runtime raised, include that in report
        report: Dict[str, Any] = {"runtimes": {}, "allclose": None}
        baseline = None
        baseline_name = None
        for name, val in results.items():
            if isinstance(val, RunResult):
                report["runtimes"][name] = {
                    "duration_s": val.duration_s,
                    "outputs": {k: v.tolist() for k, v in val.outputs.items()},
                }
                if baseline is None:
                    baseline = val.outputs
                    baseline_name = name
            else:
                report["runtimes"][name] = {"error": str(val)}
        if baseline is None:
            report["allclose"] = False
            return report
        # compare others to baseline
        diffs = {}
        all_ok = True
        for name, val in results.items():
            if not isinstance(val, RunResult):
                diffs[name] = {"ok": False, "reason": str(val)}
                all_ok = False
                continue
            per_out = {}
            for k, arr in val.outputs.items():
                ref = baseline.get(k)
                if ref is None:
                    per_out[k] = {"ok": False, "reason": "missing in baseline"}
                    all_ok = False
                    continue
                ok = np.allclose(
                    np.asarray(ref), np.asarray(arr), rtol=rtol, atol=atol
                )
                per_out[k] = {"ok": bool(ok)}
                if not ok:
                    all_ok = False
            diffs[name] = per_out
        report["allclose"] = bool(all_ok)
        report["diffs"] = diffs
        report["baseline"] = baseline_name
        return report


class ZooSandbox(LocalSandbox):
    """Convenience sandbox that can load models by canonical id from a Zoo.

    Requires a `zoo` implementing `read(id)` -> OnnxEntry.
    """

    def __init__(self, zoo, **kw):
        super().__init__(**kw)
        self.zoo = zoo

    def run(
        self,
        model_or_id: Any,
        feeds: Dict[str, np.ndarray],
        *,
        runtime: str = "reference",
        timeout_s: Optional[float] = None,
    ) -> RunResult:
        if isinstance(model_or_id, str) and not Path(model_or_id).exists():
            entry = self.zoo.read(model_or_id)
            model = entry.load()
            return super().run(
                model, feeds, runtime=runtime, timeout_s=timeout_s
            )
        return super().run(
            model_or_id, feeds, runtime=runtime, timeout_s=timeout_s
        )


__all__ = [
    "LocalSandbox",
    "ZooSandbox",
    "RuntimeAdapter",
    "ReferenceAdapter",
    "OnnxRuntimeAdapter",
]
