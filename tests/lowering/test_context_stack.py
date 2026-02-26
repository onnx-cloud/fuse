import pytest

from src.lowering.context_stack import ContextStack


def test_push_and_pop():
    stack = ContextStack()
    stack.push({"a": 1})
    stack.push({"b": 2})
    assert len(stack) == 2
    assert stack.pop() == {"b": 2}
    assert len(stack) == 1
    assert stack.pop() == {"a": 1}
    assert len(stack) == 0


def test_lookup_and_set():
    stack = ContextStack()
    stack.push({"x": 10})
    stack.push({"y": 20})
    assert stack.lookup("y") == 20
    assert stack.lookup("x") == 10
    with pytest.raises(KeyError):
        stack.lookup("z")
    # set should modify top frame
    stack.set("z", 30)
    assert stack.lookup("z") == 30


def test_empty_stack_behaviour():
    stack = ContextStack()
    with pytest.raises(KeyError):
        stack.lookup("nothing")

    # set should create a frame automatically
    stack.set("foo", "bar")
    assert stack.lookup("foo") == "bar"
    assert len(stack) == 1


def test_envdict_lookup_push_pop():
    from src.lowering.context_stack import EnvDict
    ed = EnvDict({"a": 1})
    assert ed["a"] == 1
    ed["b"] = 2
    assert ed["b"] == 2
    # push new frame
    ed.push({"a": 3})
    assert ed["a"] == 3
    assert ed["b"] == 2
    ed.pop()
    assert ed["a"] == 1
    assert "b" in ed
