"""Install helper for ORT Web runtime assets.

Provides a Python API used by tests and a thin shell wrapper `scripts/install_ort_web.sh`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Optional


def _sha256_of_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as res, open(dest, "wb") as out:
        out.write(res.read())


class InstallError(Exception):
    pass


def install_ort_web(
    dest_dir: str,
    js_url: str,
    wasm_url: str,
    js_sha256: Optional[str] = None,
    wasm_sha256: Optional[str] = None,
) -> Dict[str, str]:
    """Download and install ORT Web runtime assets into `dest_dir`.

    Returns a dict with keys 'js' and 'wasm' mapping to their sha256.
    Raises InstallError on mismatch or download error.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    js_path = dest / "ort-wasm.js"
    wasm_path = dest / "ort-wasm.wasm"

    try:
        _download(js_url, js_path)
        _download(wasm_url, wasm_path)
    except Exception as e:
        raise InstallError(f"Failed to download runtime assets: {e}") from e

    js_hash = _sha256_of_path(js_path)
    wasm_hash = _sha256_of_path(wasm_path)

    if js_sha256 and js_hash != js_sha256:
        raise InstallError(
            f"JS checksum mismatch: asserted {js_sha256}, got {js_hash}"
        )
    if wasm_sha256 and wasm_hash != wasm_sha256:
        raise InstallError(
            f"WASM checksum mismatch: asserted {wasm_sha256}, got {wasm_hash}"
        )

    return {"js": js_hash, "wasm": wasm_hash}


def _fetch_github_latest_release(repo: str = "microsoft/onnxruntime") -> dict:
    """Return the JSON payload of the latest GitHub release for `repo`.

    Uses the public GitHub API. Raises InstallError on network or API errors.
    """
    import json
    from urllib.request import Request, urlopen

    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = Request(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "fuse-install",
        },
    )
    try:
        with urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except Exception as e:
        raise InstallError(
            f"Failed to fetch latest release metadata from {url}: {e}"
        ) from e


def _fetch_github_release_by_tag(
    tag: str, repo: str = "microsoft/onnxruntime"
) -> dict:
    """Return the JSON payload for a specific release tag using the GitHub API.

    Raises InstallError on network/API errors or if the tag is not found.
    """
    import json
    from urllib.request import Request, urlopen

    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = Request(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "fuse-install",
        },
    )
    try:
        with urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except Exception as e:
        raise InstallError(
            f"Failed to fetch release metadata for tag {tag} from {url}: {e}"
        ) from e


def install_latest_ort_web(
    dest_dir: str,
    repo: str = "microsoft/onnxruntime",
    *,
    asset_js_substr: str = "ort",
    asset_wasm_substr: str = "wasm",
) -> Dict[str, str]:
    """Fetch latest release assets that look like ORT Web runtime and install them.

    This inspects release assets and selects the first JS asset containing `asset_js_substr`
    and the first WASM asset containing `asset_wasm_substr` in their names. It downloads
    them and returns their sha256s.

    Note: This function is best-effort and will raise InstallError if it can't find or
    download suitable assets.
    """
    rel = _fetch_github_latest_release(repo)
    assets = rel.get("assets", [])
    js_asset = None
    wasm_asset = None
    for a in assets:
        name = a.get("name", "")
        if (
            js_asset is None
            and asset_js_substr in name
            and name.endswith(".js")
        ):
            js_asset = a
        if (
            wasm_asset is None
            and asset_wasm_substr in name
            and name.endswith(".wasm")
        ):
            wasm_asset = a
    if js_asset is None or wasm_asset is None:
        raise InstallError(
            "Could not find suitable ORT Web runtime assets in latest release assets"
        )

    js_url = js_asset.get("browser_download_url")
    wasm_url = wasm_asset.get("browser_download_url")

    # Download and return hashes
    return install_ort_web(dest_dir, js_url, wasm_url)


def install_release_by_tag(
    dest_dir: str,
    tag: str,
    repo: str = "microsoft/onnxruntime",
    *,
    asset_js_substr: str = "ort",
    asset_wasm_substr: str = "wasm",
    pin_lock: bool = True,
) -> Dict[str, str]:
    """Fetch a specific release by `tag` and install ORT Web runtime assets.

    This finds the JS/WASM assets similarly to `install_latest_ort_web` and downloads them.
    When `pin_lock` is True, a `LOCK.json` file is written into `dest_dir` containing
    the tag, asset names, download URLs and computed sha256 checksums for reproducibility.

    Returns a dict with keys 'js' and 'wasm' mapping to their sha256.
    """
    rel = _fetch_github_release_by_tag(tag, repo)
    assets = rel.get("assets", [])
    js_asset = None
    wasm_asset = None
    for a in assets:
        name = a.get("name", "")
        if (
            js_asset is None
            and asset_js_substr in name
            and name.endswith(".js")
        ):
            js_asset = a
        if (
            wasm_asset is None
            and asset_wasm_substr in name
            and name.endswith(".wasm")
        ):
            wasm_asset = a
    if js_asset is None or wasm_asset is None:
        raise InstallError(
            f"Could not find suitable ORT Web runtime assets in release tag {tag}"
        )

    js_url = js_asset.get("browser_download_url")
    wasm_url = wasm_asset.get("browser_download_url")

    res = install_ort_web(dest_dir, js_url, wasm_url)

    if pin_lock:
        lock = {
            "tag": tag,
            "repo": repo,
            "assets": {
                "js": {
                    "name": js_asset.get("name"),
                    "url": js_url,
                    "sha256": res["js"],
                },
                "wasm": {
                    "name": wasm_asset.get("name"),
                    "url": wasm_url,
                    "sha256": res["wasm"],
                },
            },
        }
        lock_path = Path(dest_dir) / "LOCK.json"
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(lock, f, indent=2, sort_keys=True)
    return res


def install_from_npm(
    dest_dir: str,
    package: str = "onnxruntime-web",
    version: str = "1.23.2",
    *,
    wasm_candidates=None,
    pin_lock: bool = True,
) -> Dict[str, str]:
    """Download an npm package tarball for `package@version` and extract ORT Web runtime assets.
    Install the runtime into `dest_dir`.
    Select an appropriate WASM variant from `wasm_candidates`.

    Writes `LOCK.json` containing package, version, chosen asset names and sha256 checksums
    when `pin_lock` is True. Returns a dict with keys 'js' and 'wasm' mapping to their sha256.
    """
    import tarfile
    import tempfile
    from urllib.request import urlopen

    if wasm_candidates is None:
        wasm_candidates = [
            "ort-wasm-simd-threaded.wasm",
            "ort-wasm-simd-threaded.jsep.wasm",
            "ort-wasm-simd-threaded.asyncify.wasm",
            "ort.wasm",
            "ort-wasm.wasm",
        ]

    tarball_url = (
        f"https://registry.npmjs.org/{package}/-/{package}-{version}.tgz"
    )
    try:
        with urlopen(tarball_url, timeout=30) as r:
            data = r.read()
    except Exception as e:
        raise InstallError(
            f"Failed to download npm tarball {tarball_url}: {e}"
        ) from e

    # Extract the tarball and find dist JS and preferred wasm
    with tempfile.TemporaryDirectory() as td:
        import os

        tb_path = os.path.join(td, "pkg.tgz")
        with open(tb_path, "wb") as f:
            f.write(data)
        with tarfile.open(tb_path, "r:gz") as tf:
            members = tf.getnames()
            js_name = None
            wasm_name = None
            for m in members:
                if m.endswith("/dist/ort.wasm.js") or m.endswith(
                    "/dist/ort.wasm.min.js"
                ):
                    js_name = m
                    break
            # find wasm by candidates
            for candidate in wasm_candidates:
                cand_path = None
                for m in members:
                    if m.endswith(f"/dist/{candidate}"):
                        cand_path = m
                        break
                if cand_path:
                    wasm_name = cand_path
                    break
            if js_name is None or wasm_name is None:
                raise InstallError(
                    "Could not locate runtime JS or WASM artifacts in npm package tarball"
                )

            # extract and write to dest_dir
            dest = Path(dest_dir)
            dest.mkdir(parents=True, exist_ok=True)
            js_out = dest / "ort-wasm.js"
            wasm_out = dest / "ort-wasm.wasm"
            with tf.extractfile(js_name) as fh:
                js_out.write_bytes(fh.read())
            with tf.extractfile(wasm_name) as fh:
                wasm_out.write_bytes(fh.read())

    js_hash = _sha256_of_path(js_out)
    wasm_hash = _sha256_of_path(wasm_out)

    if pin_lock:
        lock = {
            "npm_package": package,
            "npm_version": version,
            "assets": {
                "js": {
                    "name": Path(js_name).name if js_name else js_out.name,
                    "sha256": js_hash,
                },
                "wasm": {
                    "name": (
                        Path(wasm_name).name if wasm_name else wasm_out.name
                    ),
                    "sha256": wasm_hash,
                },
            },
        }
        lock_path = Path(dest_dir) / "LOCK.json"
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(lock, f, indent=2, sort_keys=True)

    return {"js": js_hash, "wasm": wasm_hash}


def _cli():
    parser = argparse.ArgumentParser(prog="install_ort_web")
    parser.add_argument(
        "--dest",
        default="third_party/ort_web",
        help="Destination folder for runtime assets",
    )
    parser.add_argument(
        "--js-url", required=True, help="URL to download ort-wasm.js"
    )
    parser.add_argument(
        "--wasm-url", required=True, help="URL to download ort-wasm.wasm"
    )
    parser.add_argument(
        "--js-sha256", default=None, help="Optional asserted sha256 for JS`)"
    )
    parser.add_argument(
        "--wasm-sha256", default=None, help="Optional asserted sha256 for WASM"
    )
    args = parser.parse_args()

    res = install_ort_web(
        args.dest,
        args.js_url,
        args.wasm_url,
        js_sha256=args.js_sha256,
        wasm_sha256=args.wasm_sha256,
    )
    print("Installed ORT Web assets:")
    print(f"  js  -> {res['js']}")
    print(f"  wasm -> {res['wasm']}")


if __name__ == "__main__":
    _cli()
