#!/usr/bin/env python3
"""Create a new Fuse notebook project scaffold from a template.

Usage:
  ./scripts/scaffold_project.py <project-name>

Will create a new directory under `projects/<project-name>` with a starter
notebook, data folder and README.
"""
from __future__ import annotations

import sys
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "jupyter" / "notebooks" / "new_project_template.ipynb"


def scaffold(name: str):
    projects = ROOT / "projects"
    projects.mkdir(exist_ok=True)
    dest = projects / name
    if dest.exists():
        print(f"Project {name} already exists at {dest}")
        return 1
    dest.mkdir()
    # copy notebook
    nb_dir = dest / "notebooks"
    nb_dir.mkdir()
    shutil.copy(TEMPLATE, nb_dir / "main.ipynb")
    # data
    d = dest / "data"
    d.mkdir()
    print(f"Created project scaffold at {dest}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: scaffold_project.py <name>")
        sys.exit(2)
    sys.exit(scaffold(sys.argv[1]))
