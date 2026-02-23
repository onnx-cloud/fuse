from pathlib import Path

from src.parser import fuse_parser

GOLDEN_DIR = Path(__file__).parent / "golden"


def test_golden_files_parse():
    for p in sorted(GOLDEN_DIR.glob("*.fuse")):
        src = p.read_text()
        ast = fuse_parser.parse(src)
        assert isinstance(ast, list)
        assert ast, f"{p.name} parsed to empty AST"
