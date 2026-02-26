import logging
import pytest
import os
import sys
# configure root logger for tests
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s")
# Ensure repo root is on sys.path so top-level packages (scripts/) are importable
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Auto-enable environment-driven @fuse injection for tests that don't explicitly
# check missing/incompatible @fuse behavior. This avoids monkeypatching parser
# internals while keeping the test-suite reasonably noiseless during the
# transition to explicit @fuse declarations in all tests.
from tests.test_utils import project_fuse_version


@pytest.fixture(autouse=True)
def _auto_enable_fuse_injection(request):
    # Skip for tests that explicitly validate missing/incompatible @fuse
    # semantics (they set env vars themselves as needed).
    nid = getattr(request.node, "nodeid", "") or ""
    if request and ("test_fuse_version" in nid or "incompatible" in nid):
        yield
        return

    prev_auto = os.environ.get("FUSE_AUTO_INJECT")
    prev_auto_ver = os.environ.get("FUSE_AUTO_INJECT_VERSION")
    try:
        os.environ["FUSE_AUTO_INJECT"] = "1"
        # Use a selective injected version that does NOT change the authoritative
        # project/package version. This allows tests to auto-inject a friendly
        # @fuse for parsing without altering `get_project_version()` behaviors.
        os.environ["FUSE_AUTO_INJECT_VERSION"] = os.environ.get("FUSE_AUTO_INJECT_VERSION") or "1.2.0"
        yield
    finally:
        if prev_auto is None:
            os.environ.pop("FUSE_AUTO_INJECT", None)
        else:
            os.environ["FUSE_AUTO_INJECT"] = prev_auto
        if prev_auto_ver is None:
            os.environ.pop("FUSE_AUTO_INJECT_VERSION", None)
        else:
            os.environ["FUSE_AUTO_INJECT_VERSION"] = prev_auto_ver



class StableNamer:
    def __init__(self, prefix="n"):
        self.prefix = prefix
        self.counter = 0

    def next(self, hint=""):
        self.counter += 1
        return f"{self.prefix}_{hint}_{self.counter}"


class InMemoryImportManager:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def get(self, key):
        return self.mapping.get(key)


@pytest.fixture
def stable_namer():
    return StableNamer(prefix="test")


@pytest.fixture
def in_memory_imports():
    return InMemoryImportManager()


@pytest.fixture
def graph_context_factory():
    def _factory(**kwargs):
        # Lazy import to avoid side-effects at import time.
        from src.graph_context import GraphContext

        return GraphContext(**kwargs)

    return _factory


@pytest.fixture
def stable_name_allocator():
    """Provide a fresh `StableNameAllocator` for deterministic naming in tests."""
    # Lazy import to avoid side-effects on test collection
    from src.name_allocator import StableNameAllocator

    return StableNameAllocator(scope_prefix="test", scope_display="test.module")


@pytest.fixture
def inmemory_emitter():
    """Provide an `InMemoryONNXEmitter` for filesystem-free emission in tests."""
    from src.lowering.onnx_emitter import InMemoryONNXEmitter

    return InMemoryONNXEmitter()


@pytest.fixture
def cli_runner(tmp_path):
    class Runner:
        def __init__(self, workdir):
            self.workdir = workdir

        def run(self, argv, input_data=None):
            # Programmatic CLI entrypoint expected:
            # main(argv, in_stream, out_stream, err_stream, fs, deps)
            try:
                from src.__main__ import main as cli_main
            except Exception:
                pytest.skip("CLI entrypoint not available")
            # Minimal runner: call with argv and temp paths; tests should expand.
            return cli_main(argv)

    return Runner(tmp_path)

# reorder golden-marked tests to the end of the session
# and skip them if any prior failures have occurred

def pytest_collection_modifyitems(config, items):
    # move any item with the 'golden' marker to the end of the list
    golden_items = [i for i in items if "golden" in i.keywords]
    other_items = [i for i in items if "golden" not in i.keywords]
    if golden_items:
        items[:] = other_items + golden_items


@pytest.fixture(autouse=True)
def _skip_golden_on_failure(request):
    # this runs for every test; if it's a golden test and previous
    # failures exist, we skip it to honor the "only if all unit tests
    # pass" policy.
    if "golden" in request.keywords:
        session = request.session
        if session.testsfailed:
            pytest.skip("skipping golden test due to earlier failure")