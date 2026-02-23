import pytest
from src import parser as fuse_parser


def test_input_output_not_allowed_on_node_parsing():
    src = """
    @input { x: { bus = "bus.in" } }
    node foo(x: f32) -> f32 { x }
    """
    with pytest.raises(Exception):
        fuse_parser.fuse_parser.parse(src)
