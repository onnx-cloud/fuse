"""
Helper to install the Fuse Jupyter kernel programmatically.
"""
import sys
from pathlib import Path

def install_kernel(user=True, name="fuse", display_name="Fuse (ONNX)", make_default=False):
    try:
        from ipykernel.kernelspec import KernelSpecManager
    except ImportError:
        print("ipykernel is required to install the kernel.", file=sys.stderr)
        sys.exit(1)
    ksm = KernelSpecManager()
    # Find the kernelspec directory (may be in jupyter/ or parent of src/)
    src_path = Path(__file__)
    candidates = [
        src_path.parent.parent.parent / "jupyter" / "kernelspec",  # when installed via pip install -e .
        Path("/fused") / "jupyter" / "kernelspec",  # Docker image location
    ]
    kernelspec_path = None
    for candidate in candidates:
        if candidate.exists():
            kernelspec_path = candidate
            break
    if not kernelspec_path:
        print(f"ERROR: kernelspec not found in any of {candidates}", file=sys.stderr)
        sys.exit(1)
    
    # Install the kernel spec
    ksm.install_kernel_spec(str(kernelspec_path), kernel_name=name, user=user)
    print(f"Installed Jupyter kernel '{display_name}' as '{name}' from {kernelspec_path}")
    
    # Optionally install an additional copy under 'python3' so Fuse becomes the default 'Python 3' kernel
    if make_default and name != "python3":
        try:
            ksm.install_kernel_spec(str(kernelspec_path), kernel_name="python3", user=user)
            print("Also installed kernel as 'python3' to make Fuse the default Python kernel")
        except Exception as e:
            print(f"Warning: could not install kernel as 'python3': {e}", file=sys.stderr)

    # Also create an IPython profile startup directory with the Fuse extension loader
    try:
        from jupyter_client.kernelspec import KernelSpecManager as KSM
        # Get the installed kernel spec
        kernel_spec = ksm.get_kernel_spec(name)
        # Create startup dir in the kernel's resource directory
        profile_dir = Path.home() / ".ipython" / "profile_kernel" / "startup"
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Write startup script that loads Fuse magics
        startup_script = profile_dir / "00_fuse_magics.py"
        startup_script.write_text(
            "# Fuse IPython kernel startup script\n"
            "# Loads the Fuse IPython extension to make %%fuse and other magics available.\n"
            "try:\n"
            "    from src.jupyter.magics import load_ipython_extension\n"
            "    from IPython import get_ipython\n"
            "    ip = get_ipython()\n"
            "    if ip is not None:\n"
            "        load_ipython_extension(ip)\n"
            "except Exception as e:\n"
            "    import sys, traceback\n"
            "    print('Failed to load Fuse IPython extension at kernel startup. The `src` package is not importable in this kernel environment. Please install the package in the kernel environment (e.g., `pip install -e .`).', file=sys.stderr)\n"
            "    traceback.print_exc(file=sys.stderr)\n"
            "    raise\n"
        )
        print(f"Created IPython startup script at {startup_script}")
        # Strict validation: ensure the extension is importable in this environment.
        try:
            from src.jupyter.magics import load_ipython_extension  # type: ignore
        except Exception as e:
            import sys, traceback
            print(
                "ERROR: Fuse IPython extension could not be imported in this environment. "
                "Install the package in the Python environment used by the kernel (e.g., `pip install -e .`).",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        # This is not critical; the extension can still be loaded manually
        print(f"Note: Could not create IPython startup script: {e}", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Install the Fuse Jupyter kernel and optional startup hooks')
    parser.add_argument('--user', dest='user', action='store_true', help='Install for the current user (default)')
    parser.add_argument('--no-user', dest='user', action='store_false', help='Install system-wide')
    parser.add_argument('--name', default='fuse', help='Kernel name to install (default: fuse)')
    parser.add_argument('--display-name', default='Fuse (ONNX)', help='Kernel display name')
    parser.add_argument('--make-default', action='store_true', help='Also install a copy under the name "python3" so Fuse becomes the default Python kernel')
    args = parser.parse_args()
    install_kernel(user=args.user, name=args.name, display_name=args.display_name, make_default=args.make_default)

