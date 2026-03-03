"""Import fusion.

Loads ONNX models (variants) and fuses them into the current GraphContext with
stable prefixing. Kept separate from lowering so it can evolve independently.
"""

from __future__ import annotations

from pathlib import Path
import warnings


def _get_domain_from_meta(meta: dict) -> str | None:
    """Return domain string accepting deprecated 'module' key."""
    if not isinstance(meta, dict):
        return None
    dom = meta.get("domain")
    if dom is None and "module" in meta:
        warnings.warn("metadata key 'module' is deprecated; use 'domain' instead", DeprecationWarning)
        dom = meta.get("module")
    return dom
from typing import Any, Dict, List, Optional  # noqa: E402

import onnx  # noqa: E402
from src.graph_context import GraphContext  # noqa: E402
from src.onnx_opset import convert_model_to_opset  # noqa: E402


class ImportManager:
    """Handles loading ONNX variants and fusing them into a domain."""

    def __init__(self):
        self.loaded: Dict[str, onnx.ModelProto] = {}
        # alias -> signature for call-site wiring
        self.fused_signatures: Dict[str, Dict[str, List[str]]] = {}
        # cache metadata for remote sources (by normalized cache path)
        self._cache_meta: Dict[str, Dict[str, Any]] = {}
        # top-level lock keeping provenance of imported sources
        self.lock_path = Path("onnx") / "LOCK.json"

    def _select_variant(
        self,
        import_decl: Dict[str, Any],
        variant_name: Optional[str],
        force: bool = False,
    ) -> Dict[str, Any]:
        variants = import_decl.get("variants") or []
        if not variants:
            # Support direct `from "..."` source on import declarations.
            src = import_decl.get("source")
            name = str(import_decl["name"])
            version = str(import_decl.get("version"))
            if src:
                # Ensure we have a local cached copy for the source (URL or file path)
                local = self._ensure_local_source(
                    name, version, src, force=force
                )
                return {"name": "default", "file": str(local), "default": True}

            # Legacy: look for pre-exported artifacts under onnx/<name>/<version>.
            # If the import had no explicit version, look for any version dirs
            # and prefer the latest.
            version = import_decl.get("version")
            base_root = Path("onnx") / Path(name.replace(".", "/"))
            if version is None:
                if base_root.exists() and base_root.is_dir():
                    # sorted, prefer highest/last
                    for vd in sorted(
                        [p for p in base_root.iterdir() if p.is_dir()],
                        reverse=True,
                    ):
                        files = sorted(
                            [p for p in vd.glob("*.onnx") if p.is_file()]
                        )
                        if files:
                            return {
                                "name": "default",
                                "file": str(files[0]),
                                "default": True,
                            }
                # No local artifact found: synthesize a minimal stub ONNX model so imports
                # that are only used for composition in examples can still be resolved.
                try:
                    stub_dir = base_root / "1"
                    stub_dir.mkdir(parents=True, exist_ok=True)
                    stub_path = stub_dir / "model.onnx"
                    if not stub_path.exists():
                        from onnx import TensorProto, helper

                        # Minimal model: one input, one output, identity via Add with zero
                        x = helper.make_tensor_value_info(
                            "x", TensorProto.FLOAT, None
                        )
                        out = helper.make_tensor_value_info(
                            "out", TensorProto.FLOAT, None
                        )
                        zero = helper.make_tensor(
                            "zero_init", TensorProto.FLOAT, [1], [0.0]
                        )
                        add_node = helper.make_node(
                            "Add", ["x", "zero_init"], ["out"], name="add_stub"
                        )
                        graph = helper.make_graph(
                            [add_node],
                            f"stub_{name}",
                            [x],
                            [out],
                            initializer=[zero],
                        )
                        model = helper.make_model(graph)
                        model.ir_version = 7
                        model.opset_import[0].version = 18
                        import onnx as _onnx  # noqa: E402

                        _onnx.save(model, str(stub_path))
                    return {
                        "name": "default",
                        "file": str(stub_path),
                        "default": True,
                    }
                except Exception:
                    raise ValueError(
                        f"import {name} has no variants and no local ONNX "
                        f"found under {base_root}"
                    )

            base = base_root / str(version)
            if base.exists() and base.is_dir():
                files = sorted([p for p in base.glob("*.onnx") if p.is_file()])
                if files:
                    return {
                        "name": "default",
                        "file": str(files[0]),
                        "default": True,
                    }
            raise ValueError(
                f"import {name}@{version} has no variants and no local ONNX found under {base}"
            )

        if variant_name:
            matches = [v for v in variants if v["name"] == variant_name]
            if not matches:
                raise ValueError(
                    f"variant '{variant_name}' not found for import {import_decl['name']}"
                )
            return matches[0]

        defaults = [v for v in variants if v.get("default")]
        if len(defaults) == 1:
            return defaults[0]

        return variants[0]

    def fuse_import(
        self,
        ctx: GraphContext,
        import_decl: Dict[str, Any],
        variant_name: Optional[str] = None,
        refresh: bool = False,
    ):
        variant = self._select_variant(
            import_decl, variant_name, force=refresh
        )
        imported_model = onnx.load(str(variant["file"]))

        imported_model = convert_model_to_opset(imported_model, ctx.opset)

        # Preserve opset imports for non-default domains (e.g., com.microsoft).
        for opset in imported_model.opset_import:
            if opset.domain and opset.domain != "":
                ctx.extra_opsets[opset.domain] = max(
                    int(ctx.extra_opsets.get(opset.domain, 0)),
                    int(opset.version),
                )

        alias = import_decl["alias"]

        # If variant metadata recorded external files at fetch-time, include
        # them so they can be copied into output dirs during `fuse onnx` save.
        try:
            import json

            smeta = Path(variant["file"]).parent / "$.json"
            if smeta.exists():
                with open(smeta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ext = meta.get("external_files") or []
                existing = ctx.model_metadata.get("external_files", [])
                for e in ext:
                    existing.append(
                        {
                            "src": str(e.get("src")),
                            "dest": e.get("dest"),
                            "init_name": e.get("init_name"),
                        }
                    )
                if existing:
                    ctx.model_metadata["external_files"] = existing
        except Exception:
            pass

        # Compose prefix with scope if present on context. If a function/file scope
        # is not set on the context (imports at module top-level), fall back to
        # using the declared domain name so imports are still domaind.
        module_prefix = _get_domain_from_meta(ctx.model_metadata)
        prefix = (
            f"{ctx.scope_prefix}_"
            if getattr(ctx, "scope_prefix", None)
            else (f"{module_prefix}_" if module_prefix else "")
        )
        aliased = f"{prefix}{alias}"

        input_infos: List[Dict[str, Any]] = []
        for vi in imported_model.graph.input:
            dims: List[int] = []
            ttype = vi.type.tensor_type
            for d in ttype.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                else:
                    dims.append(0)
            input_infos.append(
                {
                    "name": f"{aliased}_{vi.name}",
                    "elem_type": int(ttype.elem_type),
                    "dims": dims,
                }
            )

        outputs = [
            f"{aliased}_{vi.name}" for vi in imported_model.graph.output
        ]
        # If the imported model has no explicit graph outputs, attempt to
        # fall back to the last node's first output (common for small snippets)
        if not outputs and imported_model.graph.node:
            last = imported_model.graph.node[-1]
            if last.output:
                outputs = [f"{aliased}_{last.output[0]}"]
        self.fused_signatures[aliased] = {
            "inputs": [i["name"] for i in input_infos],
            "outputs": outputs,
            "input_infos": input_infos,
        }
        # Also allow resolving the imported model by its bare alias (un-prefixed)
        # for backward compatibility with call sites that use the short name.
        self.fused_signatures[alias] = self.fused_signatures[aliased]

        # record where the imported-model nodes will be placed so call-site
        # wiring (Identity inserts) can be made at the correct position.
        ctx.import_node_start[aliased] = len(ctx.nodes)
        # Also record the short alias for backward-compatible lookups
        ctx.import_node_start[alias] = ctx.import_node_start[aliased]

        def _prefix_graph_names(graph):
            # Prefix names in a GraphProto (nodes, inputs/outputs/value_info, initializers)
            # and apply changes recursively into nested subgraphs
            # Update inputs, value_info, and outputs
            for vi in (
                list(graph.input)
                + list(getattr(graph, "value_info", []))
                + list(graph.output)
            ):
                if getattr(vi, "name", None):
                    vi.name = f"{aliased}_{vi.name}"
            # Update initializers
            for init in list(getattr(graph, "initializer", [])):
                if getattr(init, "name", None):
                    init.name = f"{aliased}_{init.name}"
            # Update nodes and recurse into any subgraphs found in attributes
            for node in graph.node:
                node.name = (
                    f"{aliased}_{node.name}"
                    if node.name
                    else ctx._next_node_name(aliased)
                )
                node.input[:] = [f"{aliased}_{i}" for i in node.input]
                node.output[:] = [f"{aliased}_{o}" for o in node.output]
                for attr in node.attribute:
                    # 'g' for single GraphProto attributes (e.g., Scan body)
                    if getattr(attr, "g", None):
                        _prefix_graph_names(attr.g)
                    # 'graphs' for multiple GraphProto attributes (e.g., If)
                    if getattr(attr, "graphs", None):
                        for g in attr.graphs:
                            _prefix_graph_names(g)

        _prefix_graph_names(imported_model.graph)

        # Prefix and import any FunctionProto definitions from the imported model
        try:
            for f in list(imported_model.functions):
                newf = onnx.FunctionProto()
                newf.CopyFrom(f)
                # prefix the function name itself to avoid collisions
                if newf.name:
                    newf.name = f"{aliased}_{newf.name}"
                # prefix inputs/outputs/value_info
                for i in range(len(newf.input)):
                    newf.input[i] = f"{aliased}_{newf.input[i]}"
                for i in range(len(newf.output)):
                    newf.output[i] = f"{aliased}_{newf.output[i]}"
                for vi in newf.value_info:
                    if vi.name:
                        vi.name = f"{aliased}_{vi.name}"
                # prefix nodes inside function body
                for node in newf.node:
                    node.name = (
                        f"{aliased}_{node.name}"
                        if node.name
                        else ctx._next_node_name(aliased)
                    )
                    node.input[:] = [f"{aliased}_{i}" for i in node.input]
                    node.output[:] = [f"{aliased}_{o}" for o in node.output]
                    for attr in node.attribute:
                        if getattr(attr, "g", None):
                            _prefix_graph_names(attr.g)
                        if getattr(attr, "graphs", None):
                            for g in attr.graphs:
                                _prefix_graph_names(g)
                ctx.functions.append(newf)
        except Exception:
            pass

        # Process initializers: rename and preserve external data references
        external_files = ctx.model_metadata.get("external_files", [])
        for init in imported_model.graph.initializer:
            old_name = str(init.name)
            init.name = f"{aliased}_{old_name}"
            # If initializer references external data, record the file to copy and
            # update the external_data location to a basename so that the output
            # ONNX refers to a local file next to the model.
            try:
                from onnx import TensorProto

                is_external = (
                    getattr(init, "data_location", None)
                    == TensorProto.EXTERNAL
                ) or (len(init.external_data) > 0)
            except Exception:
                is_external = False

            if is_external:
                # Find the 'location' entry if present
                loc = None
                for entry in init.external_data:
                    if entry.key == "location":
                        loc = entry.value
                        break
                # If location present, make it relative to variant file and add to list
                if loc:
                    src_path = (
                        Path(variant["file"]).parent / Path(loc)
                    ).resolve()
                    dest_name = Path(loc).name
                    external_files.append(
                        {
                            "src": str(src_path),
                            "dest": dest_name,
                            "init_name": init.name,
                        }
                    )
                    # update entry to use dest_name (relative basename)
                    for entry in init.external_data:
                        if entry.key == "location":
                            entry.value = dest_name
                ctx.model_metadata["external_files"] = external_files

        # Use aliased as the key to loaded/fused records
        self.fused_signatures[aliased] = self.fused_signatures.pop(aliased)
        self.loaded[aliased] = imported_model

        ctx.nodes.extend(imported_model.graph.node)
        ctx.initializers.update(
            {init.name: init for init in imported_model.graph.initializer}
        )

        for init in imported_model.graph.initializer:
            ctx.defined_values.add(init.name)
        for node in imported_model.graph.node:
            for o in node.output:
                ctx.defined_values.add(o)

        self.loaded[alias] = imported_model

    def _ensure_local_source(
        self, name: str, version: str, src: str, force: bool = False
    ) -> Path:
        """Ensure a local cached copy of `src` exists under onnx/<name>/<version>.

        Supports local filesystem paths and http/https URLs. Returns the Path
        to the local onnx file to be used for fusion. Attempts to avoid
        unnecessary downloads by consulting simple metadata (ETag/Last-Modified
        or remote hash when available). If `force` is True, re-fetch regardless
        of cached metadata.
        """
        import hashlib
        import json
        import shutil
        import time
        import urllib.request
        from urllib.parse import urlparse

        cache_dir = Path("onnx") / Path(name.replace(".", "/")) / version
        cache_dir.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(src)

        # Helper to write metadata file
        def write_meta(p: Path, meta: Dict[str, Any]):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, sort_keys=True)

        meta_path = cache_dir / "$.json"

        if parsed.scheme in ("http", "https"):
            # remote URL: check if we have a cached copy and whether the remote
            # has changed (using ETag/Last-Modified when available)
            try:
                req = urllib.request.Request(src, method="HEAD")
                with urllib.request.urlopen(req, timeout=5) as res:
                    etag = res.headers.get("ETag")
                    last = res.headers.get("Last-Modified")
            except Exception:
                etag = None
                last = None

            # decide target filename
            name_part = Path(parsed.path).name or "model.onnx"
            target = cache_dir / name_part

            need_fetch = True
            if target.exists() and meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    if etag and old.get("etag") and old.get("etag") == etag:
                        need_fetch = False
                    if (
                        last
                        and old.get("last_modified")
                        and old.get("last_modified") == last
                    ):
                        need_fetch = False
                except Exception:
                    need_fetch = True

            if need_fetch or force:
                # fetch and write
                try:
                    with (
                        urllib.request.urlopen(src, timeout=20) as res,
                        open(target, "wb") as out,
                    ):
                        shutil.copyfileobj(res, out)
                except Exception as e:
                    raise ValueError(
                        f"Failed to fetch remote import source {src}: {e}"
                    ) from e

                # verify we fetched a valid ONNX model; if not, try common raw URL rewrites
                def _validate_onnx_file(p: Path) -> bool:
                    try:
                        import onnx  # noqa: E402

                        onnx.load(str(p))
                        return True
                    except Exception:
                        return False

                if not _validate_onnx_file(target):
                    # Attempt common URL rewrites for GitHub/HuggingFace blob URLs
                    alt_urls = []
                    if "github.com" in src and "/blob/" in src:
                        # Convert GitHub blob URLs to raw.githubusercontent.com format:
                        #   https://github.com/user/repo/blob/branch/path
                        #   -> https://raw.githubusercontent.com/user/repo/branch/path
                        parts = src.split("github.com/", 1)[1]
                        user_repo, _, rest = parts.partition("/")
                        # safer construction:
                        raw = src.replace(
                            "https://github.com/",
                            "https://raw.githubusercontent.com/",
                        ).replace("/blob/", "/")
                        alt_urls.append(raw)
                    if "huggingface.co" in src and "/blob/" in src:
                        alt_urls.append(src.replace("/blob/", "/resolve/"))
                    # try alternatives
                    fetched_ok = False
                    for a in alt_urls:
                        try:
                            with (
                                urllib.request.urlopen(a, timeout=20) as res,
                                open(target, "wb") as out,
                            ):
                                shutil.copyfileobj(res, out)
                        except Exception:
                            continue
                        if _validate_onnx_file(target):
                            src = a
                            fetched_ok = True
                            break
                    if not fetched_ok:
                        raise ValueError(
                            f"Fetched source {src} does not appear to be a valid ONNX model"
                        )

                # compute checksum and write metadata
                sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
                meta = {
                    "etag": etag,
                    "last_modified": last,
                    "sha256": sha256,
                    "fetched_at": time.time(),
                    "source": src,
                }
                write_meta(meta_path, meta)
                self._update_lock(name, version, meta)
            return target

        # Otherwise treat as local path: may be absolute or relative
        p = Path(src)
        if p.exists():
            # copy into cache dir if not already the same path
            target = cache_dir / p.name
            if (
                not target.exists()
                or target.stat().st_mtime < p.stat().st_mtime
                or force
            ):
                shutil.copy2(p, target)
                import hashlib

                sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
                meta = {
                    "source": str(p),
                    "sha256": sha256,
                    "copied_at": time.time(),
                }
                write_meta(meta_path, meta)
                self._update_lock(name, version, meta)
                # If the source ONNX references external_data files next to it,
                # copy those files into the cache directory so loading the cached
                # model will find the external binaries. Also record these
                # external files in the metadata so callers can use them.
                try:
                    from onnx import ModelProto

                    raw = target.read_bytes()
                    m = ModelProto()
                    m.ParseFromString(raw)
                    ext_files = []
                    for init in getattr(m.graph, "initializer", []):
                        for entry in init.external_data:
                            if entry.key == "location":
                                loc = entry.value
                                src_path = (p.parent / loc).resolve()
                                dst = cache_dir / Path(loc).name
                                try:
                                    if src_path.exists():
                                        import shutil as _sh

                                        _sh.copy2(src_path, dst)
                                        ext_files.append(
                                            {
                                                "src": str(src_path),
                                                "dest": Path(loc).name,
                                                "init_name": init.name,
                                            }
                                        )
                                except Exception:
                                    # best-effort copy
                                    pass
                    if ext_files:
                        # update meta and write again
                        meta["external_files"] = ext_files
                        write_meta(meta_path, meta)
                except Exception:
                    # best-effort: do not fail on problems inspecting external files
                    pass
            return target
        # If not found, raise a clear error
        raise ValueError(f"Import source not found or unreachable: {src}")

    def _update_lock(self, name: str, version: str, meta: Dict[str, Any]):
        """Update top-level LOCK.json with provenance data for the given import.

        The lock is a simple mapping from "name@version" -> { source, etag, sha256, fetched_at }.
        """
        try:
            import json

            lock = {}
            if self.lock_path.exists():
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    lock = json.load(f)
            key = f"{name}@{version}"
            lock[key] = {
                "source": meta.get("source"),
                "etag": meta.get("etag"),
                "sha256": meta.get("sha256"),
                "fetched_at": meta.get("fetched_at"),
            }
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_path, "w", encoding="utf-8") as f:
                json.dump(lock, f, indent=2, sort_keys=True)
        except Exception:
            # Best-effort only; do not fail importing on lock update problems
            pass
