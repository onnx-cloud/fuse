import onnx


def validate_training_info(model: onnx.ModelProto):
    """Validate TrainingInfoProto entries on a ModelProto.

    Raises ValueError on fatal validation failures (duplicate update keys, missing initializers,
    update values not present in algorithm outputs, or initialization graph having inputs).
    Returns a dict of diagnostics on success: {"warnings": [...], "errors": [...]}.
    """
    # collect all update keys across training_info entries
    all_keys = []
    for ti in getattr(model, "training_info", []):
        # support two possible representations: map-like or repeated StringStringEntryProto
        try:
            keys = list(ti.update_binding.keys())
        except Exception:
            keys = [entry.key for entry in ti.update_binding]
        all_keys.extend(keys)

    duplicate_keys = {k for k in all_keys if all_keys.count(k) > 1}
    if duplicate_keys:
        raise ValueError(f"Duplicate update_binding key(s) across training_info entries: {sorted(list(duplicate_keys))}")

    # helpers
    model_inits = {init.name for init in getattr(model.graph, "initializer", [])}
    model_outputs = {o.name for o in getattr(model.graph, "output", [])}

    for ti in getattr(model, "training_info", []):
        # initialization graph should have no inputs by convention
        try:
            init_inputs = [i.name for i in ti.initialization.input]
        except Exception:
            init_inputs = []
        if init_inputs:
            raise ValueError(f"TrainingInfoProto.initialization graph should have no inputs, found: {init_inputs}")

        # collect algorithm initializers and outputs
        alg_inits = {init.name for init in getattr(ti.algorithm, "initializer", [])}
        alg_outputs = {o.name for o in getattr(ti.algorithm, "output", [])}

        # iterate entries in update_binding regardless of representation
        try:
            items = list(ti.update_binding.items())
        except Exception:
            items = [(entry.key, entry.value) for entry in ti.update_binding]

        for k, v in items:
            # key must refer to an initializer in model.graph or algorithm.initializer
            if k not in model_inits and k not in alg_inits:
                raise ValueError(f"update_binding key '{k}' not found as an initializer in model.graph or algorithm.initializer")
            # value must be an output of the combined training graph (algorithm outputs or model outputs)
            if v not in alg_outputs and v not in model_outputs:
                raise ValueError(f"update_binding value '{v}' for key '{k}' is not an output of the combined training graph")
    # If we reach here, consider returning a diagnostics dict (no errors)
    return {"warnings": [], "errors": []}