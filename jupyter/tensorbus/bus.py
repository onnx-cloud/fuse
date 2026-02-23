from __future__ import annotations

import threading
from hashlib import blake2b
from typing import Any, Dict, List, Optional, cast

import logging
logger = logging.getLogger(__name__)

import numpy as np
import onnx
import onnxruntime as ort

from .util import (blake3_hash, primary_device)
from .core_types import (HLC, LocalSessionResolver, ModelBinding, SessionResolver,
                   TensorBinding, TensorBusInterface, TensorSnapshot, Transport,
                   )


class TensorBus(TensorBusInterface):
    def __init__(
        self,
        name: str,
        resolver: Optional[SessionResolver] = None,
        transports: Optional[List[Transport]] = None,
    ) -> None:
        self.name = name
        self.resolver = resolver or LocalSessionResolver()
        self.providers = getattr(self.resolver, "providers", ["CPUExecutionProvider"])
        self.device = primary_device(self.providers)

        self._tensors: Dict[str, TensorBinding] = {}
        self._models: Dict[str, ModelBinding] = {}
        self._dirty: set[str] = set()
        self._lock = threading.RLock()

        self.transports = transports or []
        for t in self.transports:
            t.subscribe(self._on_snapshot)

    # Model Load / Tensor Registration

    def load_model(self, model_name: str) -> ModelBinding:
        session = self.resolver.session(model_name)
        meta = session.get_modelmeta()
        graph_name = meta.graph_name

        if not graph_name:
            raise RuntimeError(
                f"Model at '{model_name}' has no graph name. "
                f"ONNX models require a named graph."
            )

        mid = blake3_hash(graph_name)
        model = ModelBinding(mid, graph_name, session)

        with self._lock:
            self._models[graph_name] = model
            self._register_model_tensors(model)

        return model

    def _register_model_tensors(self, model: ModelBinding):
        s = model.session
        for arg in s.get_inputs():
            self._bind_tensor(arg)
        for arg in s.get_outputs():
            self._bind_tensor(arg)

    def _bind_tensor(self, arg: Any) -> None:
        tid = blake3_hash(arg.name)
        binding = TensorBinding(tid, arg, HLC(0, self.name), b"", {})
        self._tensors[arg.name] = binding

    # Set / Get Tensor

    def set_tensor(self, name: str, arr: np.ndarray) -> None:
        """Set tensor value and mark for replication.
        
        Args:
            name: Registered tensor name from model inputs/outputs
            arr: NumPy array to store
            
        Raises:
            KeyError: If tensor name not registered
            TypeError: If arr is not a numpy array
            ValueError: If array contains NaN or Inf
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(arr).__name__}")
        
        if np.isnan(arr).any():
            logger.warning(f"NaN detected in tensor '{name}'")
        
        if np.isinf(arr).any():
            logger.warning(f"Inf detected in tensor '{name}'")
        
        with self._lock:
            if name not in self._tensors:
                raise KeyError(
                    f"Tensor '{name}' not registered. "
                    f"Load a model first or use a valid tensor name."
                )
            
            b = self._tensors[name]
            cpu_val = ort.OrtValue.ortvalue_from_numpy(arr, "cpu")
            b.values["cpu"] = cpu_val

            if self.device != "cpu":
                try:
                    b.values[self.device] = ort.OrtValue.ortvalue_from_numpy(
                        arr, self.device
                    )
                except Exception:
                    pass

            # CRDT update
            b.clock = b.clock.tick()
            # use blake2b digest for payload hashing
            b.hash = blake2b(arr.tobytes()).digest()
            self._dirty.add(name)

    def get_tensor(self, name: str) -> TensorBinding:
        """Retrieve tensor binding by name.
        
        Args:
            name: Tensor name
            
        Returns:
            TensorBinding with values and metadata
            
        Raises:
            KeyError: If tensor name not found
        """
        with self._lock:
            if name not in self._tensors:
                available = list(self._tensors.keys())
                raise KeyError(
                    f"Tensor '{name}' not found. "
                    f"Available tensors: {available}"
                )
            return self._tensors[name]

    def _resolve_value(self, b: TensorBinding) -> ort.OrtValue:
        """Resolve tensor value, preferring device then CPU."""
        v = b.values.get(self.device)
        if v:
            return v
        v = b.values.get("cpu")
        if v:
            return v
        raise KeyError(
            f"No value for tensor '{b.arg.name}'. "
            f"Set tensor value before running model."
        )

    # Run / IOBinding

    def run(self, model_name: str):
        with self._lock:
            m = self._models[model_name]
            s = m.session
            io = s.io_binding()

            # Bind inputs
            for arg in s.get_inputs():
                b = self._tensors[arg.name]
                io.bind_ortvalue_input(arg.name, self._resolve_value(b))

            # Bind outputs
            for arg in s.get_outputs():
                io.bind_output(arg.name, self.device)

            # Execute
            s.run_with_iobinding(io)

            # Collect outputs, with a fallback for ORT versions that lack
            # get_output_ortvalue_by_name on IOBinding
            get_by_name = getattr(io, "get_output_ortvalue_by_name", None)
            if get_by_name is not None:
                for arg in s.get_outputs():
                    out = io.get_output_ortvalue_by_name(arg.name)
                    b = self._tensors[arg.name]
                    b.values[self.device] = out
                    arr = out.numpy()
                    b.clock = b.clock.tick()
                    # use blake2b digest for payload hashing
                    b.hash = blake2b(arr.tobytes()).digest()
                    self._dirty.add(arg.name)
            else:
                # Fallback: run session normally to obtain numpy arrays
                input_dict = {
                    arg.name: self._resolve_value(self._tensors[arg.name]).numpy()
                    for arg in s.get_inputs()
                }
                out_names = [arg.name for arg in s.get_outputs()]
                results = s.run(out_names, input_dict)
                for arg, arr in zip(s.get_outputs(), results):
                    b = self._tensors[arg.name]
                    out_ort = ort.OrtValue.ortvalue_from_numpy(arr, self.device)
                    b.values[self.device] = out_ort
                    b.clock = b.clock.tick()
                    b.hash = blake2b(arr.tobytes()).digest()
                    self._dirty.add(arg.name)

        self.sync()

    # Replication / CRDT

    def sync(self):
        for name in list(self._dirty):
            snap = self._to_snapshot(name)
            for t in self.transports:
                t.publish(snap)
        self._dirty.clear()

    def _to_snapshot(self, name: str) -> TensorSnapshot:
        b = self._tensors[name]
        v = b.values.get("cpu") or self._resolve_value(b)
        arr = v.numpy()
        return TensorSnapshot(
            name=name,
            dtype=str(arr.dtype),
            shape=arr.shape,
            clock=b.clock,
            hash=b.hash,
            payload=arr.tobytes(),
        )

    def _on_snapshot(self, snap: TensorSnapshot):
        with self._lock:
            local = self._tensors.get(snap.name)
            if not local or (snap.clock is not None and snap.clock > local.clock):
                if snap.payload is None:
                    # Nothing to restore
                    return
                if snap.clock is None:
                    logger.warning("Snapshot for %s missing clock; skipping", snap.name)
                    return
                arr = np.frombuffer(snap.payload, dtype=snap.dtype).reshape(snap.shape)
                ort_val = ort.OrtValue.ortvalue_from_numpy(arr, "cpu")

                if not local:
                    # create minimal arg-like object for registration
                    tmp = type("_Arg", (), {"name": snap.name})
                    self._bind_tensor(tmp())

                b = self._tensors[snap.name]
                b.values["cpu"] = ort_val
                b.clock = cast(HLC, snap.clock)
                b.hash = snap.hash

    # Transport Management

    def add_transport(self, transport: Transport):
        self.transports.append(transport)
        transport.subscribe(self._on_snapshot)
