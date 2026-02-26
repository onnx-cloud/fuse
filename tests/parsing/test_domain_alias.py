from src.parser import fuse_parser
import warnings

def test_module_deprecation_alias():
    src = "@module olddom\nnode n() -> f32 { return 1.0 }\n"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        ast = fuse_parser.parse(src)
    # parser should normalize the metadata name to "domain"
    metas = [m for m in ast if isinstance(m, dict) and m.get("type") == "meta"]
    assert any(m.get("name") == "domain" and m.get("value") == "olddom" for m in metas)
    # a deprecation warning regarding 'module' should have been emitted
    assert any("module" in str(warn.message) for warn in w)
