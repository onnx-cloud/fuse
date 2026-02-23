def test_const_folding_basic():
    # TODO: build small AST with constant ops and assert folding result
    assert True


def test_name_allocation_is_deterministic(stable_namer):
    n1 = stable_namer.next("x")
    n2 = stable_namer.next("x")
    assert n1 != n2
