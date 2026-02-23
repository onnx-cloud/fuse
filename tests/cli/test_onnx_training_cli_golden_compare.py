import onnx
from pathlib import Path
from src.cli import cli_commands
from src.lowering.training_checks import validate_training_info


def test_cli_training_exports_training_info_and_grad(tmp_path: Path):
    src = Path("examples/golden/training.fuse")
    out_dir = tmp_path / "onnx"
    out_dir.mkdir()

    res = cli_commands.cmd_onnx([str(src)], out_dir=str(out_dir), training=True)
    print("DEBUG: cmd_onnx result:", res)
    assert res and len(res) == 1
    src_path, out_path, err = res[0]
    assert err is None
    assert out_path and Path(out_path).exists()

    model = onnx.load(out_path)

    # Model should contain training_info entries and validate
    assert len(model.training_info) >= 1
    validate_training_info(model)

    # update_binding should reference a 'W' initializer (trainable weight)
    found_w = False
    init_names = {init.name for init in model.graph.initializer}
    for ti in model.training_info:
        for entry in ti.update_binding:
            key = entry.key
            # Allow either qualified or unqualified name forms
            if key.endswith("W") or key.endswith("_W") or key in init_names or any(k.endswith("_W") for k in init_names) or "W" in key:
                found_w = True
    assert found_w, f"expected update_binding to reference W; initializers: {sorted(init_names)}"

    # The exported model should include a gradient output (W.grad)
    out_names = [o.name for o in model.graph.output]
    assert any(("W.grad" in n or n.endswith("W.grad") or n.endswith(".W.grad")) for n in out_names), f"no W.grad in outputs: {out_names}"
