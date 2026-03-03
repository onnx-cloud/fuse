"""Remote import support for the CLI.

This is separate from lowering's import fusion (which asserts local ONNX files).
The CLI import command supports URL-based variants with local caching.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnx
from src.graph_context import GraphContext
from src.security import validate_import_url
from src.errors import E003_ImportNotFound, E051_InvalidONNXModel


class ImportCache:
    def __init__(
        self, 
        cache_dir: Optional[str] = None,
        url_whitelist: Optional[List[str]] = None,
        unsafe_imports: bool = False
    ):
        self.cache_dir = Path(
            cache_dir or os.path.expanduser("~/.fuse/onnx_cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.url_whitelist = url_whitelist
        self.unsafe_imports = unsafe_imports

    def _hash_url(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    def fetch(self, url: str) -> Path:
        # Security: validate URL against whitelist
        validate_import_url(url, whitelist=self.url_whitelist, unsafe=self.unsafe_imports)
        
        h = self._hash_url(url)
        cached_file = self.cache_dir / f"{h}.onnx"
        if cached_file.exists():
            return cached_file
        print(f"[INFO] Downloading {url}")
        try:
            urllib.request.urlretrieve(url, cached_file)
        except Exception as e:
            raise E051_InvalidONNXModel(url, f"Download failed: {e}")
        return cached_file


class RemoteImportManager:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        url_whitelist: Optional[List[str]] = None,
        unsafe_imports: bool = False
    ):
        self.cache = ImportCache(cache_dir, url_whitelist, unsafe_imports)
        self.loaded: Dict[str, onnx.ModelProto] = {}
        self.unsafe_imports = unsafe_imports

    def fuse_import(
        self,
        ctx: GraphContext,
        import_decl: Dict[str, Any],
        variant_name: Optional[str] = None,
    ):
        variants = import_decl.get("variants") or []
        if not variants:
            raise ValueError(
                "RemoteImportManager requires explicit variants with file=URL/path"
            )

        variant_name = (
            variant_name
            or import_decl.get("default_variant")
            or variants[0]["name"]
        )
        variant = next(v for v in variants if v["name"] == variant_name)
        url_or_path = variant["file"]

        if isinstance(url_or_path, str) and url_or_path.startswith("http"):
            local_path = self.cache.fetch(url_or_path)
        else: # Security: validate local file path
            local_path = Path(url_or_path)
            if not local_path.exists():
                raise E003_ImportNotFound(import_decl["name"], variant=variant_name)

        try:
            model = onnx.load(str(local_path))
        except Exception as e:
            raise E051_InvalidONNXModel(str(local_path), f"Load failed: {e}")
        alias = import_decl["alias"]

        for node in model.graph.node:
            node.name = f"{alias}_{node.name}" if node.name else ""
            node.input[:] = [f"{alias}_{i}" for i in node.input]
            node.output[:] = [f"{alias}_{o}" for o in node.output]

        for init in model.graph.initializer:
            init.name = f"{alias}_{init.name}"

        ctx.nodes.extend(model.graph.node)
        ctx.initializers.update(
            {init.name: init for init in model.graph.initializer}
        )
        self.loaded[alias] = model
        print(
            f"[INFO] Imported {import_decl['name']} variant={variant_name} as {alias}"
        )
