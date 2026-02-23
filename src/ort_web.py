import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import onnx

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "third_party" / "ort_web"


class ORTWebError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_runtime_assets(vendor_dir: Optional[Path] = None) -> dict:
    """Look for vendored ORT Web runtime assets in `vendor_dir` or default `third_party/ort_web`.

    Returns dict with keys 'js' and 'wasm' pointing to Path objects.
    Raises ORTWebError if assets are missing.
    """
    vendor = Path(vendor_dir) if vendor_dir is not None else VENDOR_DIR
    js = vendor / "ort-wasm.js"
    wasm = vendor / "ort-wasm.wasm"
    if not js.exists() or not wasm.exists():
        raise ORTWebError(
            f"ORT Web runtime assets not found under {vendor}. "
            "Place `ort-wasm.js` and `ort-wasm.wasm` there, or run the install script."
        )
    return {"js": js, "wasm": wasm}


def build_ort_web_bundle(
    model: onnx.ModelProto,
    out_dir: str,
    *,
    model_filename: str = "model.onnx",
    vendor_dir: Optional[str] = None,
) -> None:
    """Create a deterministic ORT Web bundle in `out_dir`.

    Layout (deterministic):
      <out_dir>/model.onnx
      <out_dir>/ort-wasm.js
      <out_dir>/ort-wasm.wasm
      <out_dir>/model.json

    The manifest contains checksums and simple metadata (opset, model metadata).

    `vendor_dir` may be provided to look for runtime assets in a non-default location.
    (This is used by tests.)
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Write model
    model_path = out_dir_p / model_filename
    onnx.save(model, str(model_path))

    # Copy runtime assets from vendor dir
    assets = _find_runtime_assets(
        Path(vendor_dir) if vendor_dir is not None else None
    )
    js_dst = out_dir_p / assets["js"].name
    wasm_dst = out_dir_p / assets["wasm"].name
    shutil.copy(assets["js"], js_dst)
    shutil.copy(assets["wasm"], wasm_dst)

    # Build manifest
    manifest = {
        "model": {
            "path": model_filename,
            "sha256": _sha256(model_path),
            "opsets": [
                {"domain": oi.domain, "version": oi.version}
                for oi in model.opset_import
            ],
            "metadata": (
                {kv.key: kv.value for kv in model.metadata_props}
                if model.metadata_props
                else {}
            ),
        },
        "runtime": {
            "js": {"path": js_dst.name, "sha256": _sha256(js_dst)},
            "wasm": {"path": wasm_dst.name, "sha256": _sha256(wasm_dst)},
        },
    }

    manifest_path = out_dir_p / "model.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Sane permissions
    for p in (model_path, js_dst, wasm_dst, manifest_path):
        try:
            p.chmod(0o644)
        except Exception:
            pass

    return None
