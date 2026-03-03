import json
from typing import List, Dict, Any

import onnx

# Backwards-compatible import: tests expect `validate_training_info` to live
# in this module. Delegate to `training_info_checks.validate_training_info`.
try:
    from .training_info_checks import validate_training_info  # type: ignore
except Exception:
    # keep a fallback that raises if the impl is unavailable
    def validate_training_info(model: onnx.ModelProto):
        raise RuntimeError("validate_training_info implementation not available")

# keys are lowercase optimizer identifiers as they may appear in config
OPTIMIZER_REGISTRY = {
    "adam": {"canon": "Adam", "state_suffixes": ["m", "v"], "require_state": True},
    "adamw": {"canon": "AdamW", "state_suffixes": ["m", "v"], "require_state": True},
    "adagrad": {"canon": "Adagrad", "state_suffixes": ["accum"], "require_state": True},
    "momentum": {"canon": "Momentum", "state_suffixes": ["momentum"], "require_state": True},
    "sgd": {"canon": "SGD", "state_suffixes": [], "require_state": False},
}


def _get_meta(model: onnx.ModelProto, key: str):
    for e in model.metadata_props:
        if e.key == key:
            try:
                return json.loads(e.value)
            except Exception:
                return e.value
    return None

def check_training_model(model: onnx.ModelProto) -> Dict[str, List[Dict[str,Any]]]:
    """Check that a lowered ONNX model contains expected training artifacts.

    Returns a dict with keys 'warnings' and 'errors' containing structured diagnostics:
    { code: str, message: str, param?: str, state?: str }
    """
    warnings: List[Dict[str,Any]] = []
    errors: List[Dict[str,Any]] = []

    trainables = _get_meta(model, "trainables") or {}
    training_cfg = _get_meta(model, "training_config") or {}

    # If no training config or trainables present, nothing to check
    if not training_cfg and not trainables:
        return {"warnings": warnings, "errors": errors}

    # Collect model outputs and node op_types
    outputs = {o.name for o in model.graph.output}
    nodes = list(model.graph.node)
    {n.op_type for n in nodes}

    # Helper: accept broadcasted state shapes for conv-like params and special rules
    def _shapes_compatible(p_shape, s_dims, param_name=None):
        # None shapes are treated as compatible (unknown shapes)
        if p_shape is None or s_dims is None:
            return True
        if p_shape == s_dims:
            return True

        # Load optional shape rules from schemas/training_param_shape_rules.json
        from src.util.config import load_schema
        try:
            rules = load_schema("training_param_shape_rules", {"rules": []})
        except Exception:
            rules = {"rules": []}

        # Check for matching rule by param_name (supports simple glob pattern)
        accept_mode = None
        if param_name and rules.get("rules"):
            import fnmatch

            for r in rules["rules"]:
                pat = r.get("pattern")
                if pat and fnmatch.fnmatch(param_name, pat):
                    accept_mode = r.get("accept")
                    break

        # If batchnorm-like exact 1-D required
        if accept_mode == "exact_1d":
            return s_dims == [p_shape[0]]

        # If leading channel broadcast allowed (conv-like)
        if accept_mode == "leading_channel_broadcast":
            # allow s_dims == [C] matching first dim
            if s_dims == [p_shape[0]]:
                return True
            # allow s_dims == p_shape[:len(s_dims)] with trailing ones in param
            if len(s_dims) < len(p_shape):
                if s_dims == p_shape[: len(s_dims)] and all(d == 1 for d in p_shape[len(s_dims) :]):
                    return True
            return False

        # Default (previous behavior): allow 1-D over leading channel or leading match + trailing ones
        if len(s_dims) < len(p_shape):
            if s_dims == [p_shape[0]]:
                return True
            if s_dims == p_shape[: len(s_dims)] and all(d == 1 for d in p_shape[len(s_dims) :]):
                return True
        return False

    # For each declared trainable param, ensure gradient output exists
    for pname, enabled in (trainables or {}).items():
        if not enabled:
            continue
        gname = f"{pname}.grad"
        if gname not in outputs:
            # Fallback: check for GenerateGradients or Gradient nodes that reference name
            found_grad = any(
                n.op_type in ("GenerateGradients", "Gradient")
                and (pname in ((getattr(a, "s", None) or "") for a in n.attribute))
                for n in nodes
            )
            if not found_grad:
                errors.append({
                    "code": "TRAIN.MISSING_GRADIENT",
                    "message": f"Missing gradient output '{gname}' for trainable parameter '{pname}'",
                    "param": pname,
                    "expected_output": gname,
                })

    # Check optimizer presence if requested
    opt = None
    if isinstance(training_cfg, dict):
        opt = training_cfg.get("optimizer")
    elif isinstance(training_cfg, str):
        opt = training_cfg

    if opt:
        opt_name = None
        if isinstance(opt, dict):
            opt_name = str(opt.get("type") or opt.get("optimizer") or "").lower()
        else:
            opt_name = str(opt).lower()

        # canonical mapping for backwards compatibility and registry loading
        mapping = {
            "adam": "Adam",
            "adamw": "AdamW",
            "adagrad": "Adagrad",
            "momentum": "Momentum",
            "sgd": "SGD",
        }

        # Prefer declarative registry in schemas/training_optimizers.json when available
        try:
            from pathlib import Path as _Path
            import json as _json

            reg_path = _Path(__file__).resolve().parents[1].joinpath("schemas/training_optimizers.json")
            if reg_path.exists():
                loaded_reg = _json.loads(reg_path.read_text())
            else:
                loaded_reg = {}
        except Exception:
            loaded_reg = {}

        # Prefer registry from schema, else fallback to in-code registry
        registry = loaded_reg.get(opt_name) or OPTIMIZER_REGISTRY.get(opt_name)
        canon = None
        if isinstance(registry, dict):
            canon = registry.get("canon") or registry.get("canon_name")
        if not canon:
            canon = mapping.get(opt_name, None)

        # Warn if optimizer op not present (best-effort)
        if canon:
            found = any(n.op_type == canon or n.op_type.lower() == canon.lower() for n in nodes)
            if not found:
                warnings.append({"code": "TRAIN.MISSING_OPTIMIZER", "message": f"Optimizer '{canon}' requested by training_config not found in lowered model"})

        # If registry defines expected state tensors, check presence/shape/dtype
        if registry and isinstance(registry, dict) and registry.get("require_state") and trainables:
            inits = {init.name: init for init in getattr(model.graph, "initializer", [])}

            def _param_shape(name: str):
                if name in inits:
                    return list(inits[name].dims)
                for vi in getattr(model.graph, "value_info", []):
                    if getattr(vi, "name", None) == name:
                        t = vi.type.tensor_type
                        return [d.dim_value if d.HasField("dim_value") else 0 for d in t.shape.dim]
                return None

            for pname, enabled in (trainables or {}).items():
                if not enabled:
                    continue
                p_shape = _param_shape(pname)
                p_dtype = inits.get(pname).data_type if pname in inits else None

                for suff in registry.get("state_suffixes", []):
                    state_name = f"{pname}.{suff}"
                    if state_name not in inits:
                        warnings.append({
                            "code": "TRAIN.MISSING_STATE",
                            "message": f"Optimizer '{registry.get('canon') or canon or opt_name}' expects state initializer '{state_name}' for parameter '{pname}' but it was not found",
                            "param": pname,
                            "state": state_name,
                        })
                        continue
                    state_init = inits[state_name]
                    s_dims = list(state_init.dims)
                    # shape check
                    if p_shape is not None and not _shapes_compatible(p_shape, s_dims, param_name=pname):
                        warnings.append({
                            "code": "TRAIN.STATE_SHAPE_MISMATCH",
                            "message": f"Optimizer state '{state_name}' dims {s_dims} differ from parameter '{pname}' dims {p_shape}",
                            "param": pname,
                            "state": state_name,
                        })
                    # dtype check, if available
                    if p_dtype is not None and getattr(state_init, "data_type", None) is not None:
                        if p_dtype != state_init.data_type:
                            warnings.append({
                                "code": "TRAIN.STATE_DTYPE_MISMATCH",
                                "message": f"Optimizer state '{state_name}' dtype {state_init.data_type} differs from parameter '{pname}' dtype {p_dtype}",
                                "param": pname,
                                "state": state_name,
                            })
        elif not registry:
            known_any = any(n.op_type in mapping.values() for n in nodes)
            if not known_any:
                warnings.append({"code": "TRAIN.UNKNOWN_OPTIMIZER", "message": f"Specified optimizer '{opt}' not recognized and no optimizer nodes found in model"})

    return {"warnings": warnings, "errors": errors}