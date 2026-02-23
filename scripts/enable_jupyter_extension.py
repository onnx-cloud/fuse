"""Enable the Fuse Jupyter server extension in this environment.

Attempts `jupyter server extension enable --sys-prefix --py fuse_server` first. If that
is not available, writes a minimal config fragment into the system/user jupyter config
location so the extension is enabled.

Run this during setup or as a troubleshooting helper.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

def try_enable_with_jupytercli():
    try:
        # Prefer the jupyter CLI to enable the extension (works on most installs)
        cmd = [sys.executable, "-m", "jupyter", "server", "extension", "enable", "--sys-prefix", "--py", "fuse_server"]
        subprocess.check_call(cmd)
        print("Enabled fuse_server via jupyter CLI")
        return True
    except Exception:
        return False

def write_config_fragment():
    # Determine site-wide config directory (sys.prefix + etc/jupyter) or user config
    # Prefer sys-prefix location when writable
    candidates = [Path(sys.prefix) / "etc" / "jupyter" / "jupyter_server_config.d",
                  Path.home() / ".jupyter" / "jupyter_server_config.d"]
    fragment = {"ServerApp": {"jpserver_extensions": {"fuse_server": True}}}
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            p = d / "fuse.json"
            p.write_text(json.dumps(fragment))
            print(f"Wrote config fragment to {p}")
            return True
        except Exception as e:
            print(f"Failed to write to {d}: {e}")
    return False

if __name__ == "__main__":
    if try_enable_with_jupytercli():
        sys.exit(0)
    if write_config_fragment():
        print("Enabled fuse_server by writing config fragment")
        sys.exit(0)
    print("Failed to enable the Fuse Jupyter server extension; try running with elevated permissions or enabling manually.")
    sys.exit(1)
