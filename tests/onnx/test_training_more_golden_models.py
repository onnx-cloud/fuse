import pytest
pytestmark = pytest.mark.golden
import onnx
from src.lowering.training_checks import validate_training_info, check_training_model
from examples.golden.generate_training_golden import (
    make_training_with_optimizer_states,
    make_training_with_invalid_duplicate_updates,
    make_training_with_init_with_inputs,
)


def test_optimizer_state_example_valid():
    m = make_training_with_optimizer_states()
    # ensure validate_training_info doesn't raise
    validate_training_info(m)
    # check_training_model should not produce missing-state warnings for this example
    res = check_training_model(m)
    assert not any(w.get("code") == "TRAIN.MISSING_STATE" for w in res.get("warnings", []))


def test_duplicate_updates_invalid():
    m = make_training_with_invalid_duplicate_updates()
    with pytest.raises(ValueError, match="Duplicate update_binding"):
        validate_training_info(m)


def test_initialization_with_inputs_invalid():
    m = make_training_with_init_with_inputs()
    with pytest.raises(ValueError, match="initialization graph should have no inputs"):
        validate_training_info(m)
