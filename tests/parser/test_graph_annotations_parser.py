from src import parser as fuse_parser


def test_parse_input_output_annotations_on_graph():
    src = """
    @domain example.test.graph_annotations

    @persistent {
      input {
        x: "bus.in"
      }
      output {
        y:  "bus.out"
      }
    }
    graph demo(x: f32[1]) -> f32[1] {
      return { y: x }
    }
    """
    ast = fuse_parser.fuse_parser.parse(src)
    # find graph decl for demo
    decls = [d for d in ast if isinstance(d, dict) and d.get("type") == "model"]
    assert any(d.get("name") == "demo" for d in decls)
    demo = next(d for d in decls if d.get("name") == "demo")
    assert "input" in demo and isinstance(demo["input"], dict)
    assert "x" in demo["input"] and demo["input"]["x"] == "bus.in"
    assert "output" in demo and isinstance(demo["output"], dict)
    assert "y" in demo["output"] and demo["output"]["y"] == "bus.out"
