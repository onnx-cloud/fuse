from src.util.project_version import get_project_version as project_fuse_version


# Thin wrapper used by tests to read the authoritative project version.
# Using the centralized helper ensures consistent behavior and a single
# place to adapt env/pyproject/importlib fallbacks.

