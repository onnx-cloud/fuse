from pathlib import Path


def test_debug_plan_mentions_reproducer_and_dumper():
    p = Path(__file__).parent / "PLAN.md"
    assert p.exists(), "PLAN.md should be present in tests/debug/"
    t = p.read_text().lower()
    assert "reproducer" in t
    assert "debugdumper" in t.replace(" ", "") or "debug dumper" in t


def test_debug_fixtures_dir_present():
    fixtures = Path(__file__).parent / "fixtures"
    # Accept either existing fixtures or an instruction in the plan to create one
    if fixtures.exists():
        assert fixtures.is_dir()
    else:
        # Fall back to ensuring the plan instructs creating the fixtures dir
        p = Path(__file__).parent / "PLAN.md"
        t = p.read_text().lower()
        assert "fixtures" in t
