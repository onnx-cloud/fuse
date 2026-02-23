"""Local `jupyter` package used for Fuse tests and development.

This small __init__ ensures the repository-provided `jupyter` package is
importable during test runs where the top-level `jupyter` namespace might
otherwise resolve to the system-installed package.
"""
__all__ = []
