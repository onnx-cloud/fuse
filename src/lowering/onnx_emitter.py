import json
from typing import Dict

import onnx


class ONNXEmitter:
    """Abstract emitter interface."""

    def save_model_bytes(self, model: onnx.ModelProto) -> bytes:
        """Return serialized bytes for the model."""
        raise NotImplementedError

    def save_model(self, model: onnx.ModelProto, path: str) -> None:
        """Save model to path (may copy external files)."""
        raise NotImplementedError


class InMemoryONNXEmitter(ONNXEmitter):
    """In-memory emitter storing model bytes and external files in dictionaries."""

    def __init__(self):
        self.models: Dict[str, bytes] = {}
        self.external_files: Dict[str, bytes] = {}

    def save_model_bytes(self, model: onnx.ModelProto) -> bytes:
        # Normalize by serializing deterministically where available
        try:
            data = model.SerializeToString(deterministic=True)
        except TypeError:
            data = onnx._serialize(model)
        return data

    def save_model(self, model: onnx.ModelProto, path: str) -> None:
        data = self.save_model_bytes(model)
        self.models[path] = data
        # Copy external files listed in metadata_props if present
        try:
            external_json = None
            for e in model.metadata_props:
                if e.key == "external_files":
                    external_json = e.value
                    break
            if external_json:
                files = json.loads(external_json)
                for entry in files:
                    src = entry.get("src")
                    dest = entry.get("dest")
                    # If the src is a path that was previously provided to the
                    # emitter (via register_external), copy it into memory.
                    if src in self.external_files:
                        self.external_files[dest] = self.external_files[src]
        except Exception:
            pass

    def register_external(self, src_path: str, data: bytes):
        self.external_files[src_path] = data
