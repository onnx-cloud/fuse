"""Check local developer environment and print actionable setup steps.

Run with: python tools/check_env.py
"""

import shutil
import sys
import importlib


def main():
    """Check basic environment information and presence of required Python packages.

    This script is safe to run in the system environment or inside the project
    virtualenv. It returns non-zero if required packages are missing.
    """

    py = shutil.which("python3") or shutil.which("python")
    pip_cmd = shutil.which("pip") or shutil.which("pip3")
    uv_cmd = shutil.which("uv")

    print("Environment check:")
    print(f"  python: {py}")
    print(f"  pip:    {pip_cmd}")
    print(f"  uv:     {uv_cmd or '<not found>'}")

    if py is None:
        print("ERROR: No python interpreter found. Install Python 3.")
        sys.exit(2)

    if uv_cmd is None:
        print(
            "WARNING: 'uv' runner not found. You can install it with: python3 -m pip install uv",
            file=sys.stderr,
        )

    # Check for important packages used by tests and tooling
    required = {
        "lark-parser": "lark",
        "onnx": "onnx",
        "pytest": "pytest",
        "pygls": "pygls",
        "pydantic": "pydantic",
        "IPython": "IPython",
    }
    missing = []
    for pkg_name, mod in required.items():
        try:
            importlib.import_module(mod)
            print(f"OK: {pkg_name} (module {mod})")
        except Exception:
            missing.append(pkg_name)

    if missing:
        print("\nMissing required Python packages: " + ", ".join(missing), file=sys.stderr)
        print("Install with: uv pip install " + " ".join(missing), file=sys.stderr)
        sys.exit(1)

    print('\nAll required packages are present.')

if __name__ == "__main__":
    main()
