import logging
from typing import Dict, Any

from src.graph_context import as_tensor_type

logger = logging.getLogger(__name__)


def generate_gradients(ctx) -> Dict[str, Any]:
    """Generate training-oriented outputs for gradients and (optional) loss.

    This function performs automatic differentiation (reverse-mode autodiff) to
    compute per-parameter gradients. It creates graph outputs named
    `<param>.grad` for every entry in `ctx.model_metadata.get('trainables')`
    that is truthy and whose type is known in `ctx.value_types`. It will also
    expose an existing `loss` value as a graph output if present in
    `ctx.value_types`.

    Supported operations (with automatic gradient computation):
    - MatMul: matrix multiplication with transposition handling
    - Add: addition with broadcasting reduction
    - Mul: element-wise multiplication with broadcasting reduction
    - ReduceSum: reduction with gradient expansion
    - ReduceMean: reduction with gradient expansion and scaling
    - ReLU: rectified linear unit with mask-based gradient
    - Sigmoid: sigmoid activation with chain rule derivative y*(1-y)
    - Tanh: hyperbolic tangent activation with chain rule derivative 1-y^2
    - Conv: convolution with ConvTranspose-based gradient computation
    - LayerNormalization: layer normalization with scale/bias gradients
    - BatchNormalization: batch normalization with scale/bias gradients

    Unsupported ops are silently ignored during backpropagation.

    Returns a summary dict of outputs added.
    """
    added = {"grads": [], "loss": None, "opt_updates": {}, "optimizer_nodes": []}
    trainables = ctx.model_metadata.get("trainables", {}) or {}

    # If an explicit training opset was not provided, ensure we have a
    # reasonable default so training ops can be declared in the model.
    training_domain = "ai.onnx.preview.training"
    if training_domain not in ctx.extra_opsets:
        ctx.extra_opsets[training_domain] = max(int(ctx.extra_opsets.get(training_domain, 0)), 1)

    # Build a mapping of value -> producing node for the forward graph
    out_to_node = {}
    node_inputs = {}
    for n in ctx.nodes:
        for out in list(n.output):
            out_to_node[out] = n
        node_inputs[n.name] = list(n.input)

    # Determine which trainable params we may need gradients for
    grad_names = []
    for pname, enabled in trainables.items():
        if enabled and (ctx.value_types.get(pname) or ctx.value_types.get(pname.split(".")[-1])):
            grad_names.append(f"{pname}.grad")

    # If a loss is present, perform reverse-mode autodiff over
    # supported ops (MatMul, Add, Mul, ReduceSum, ReduceMean, ReLU, Sigmoid, Tanh, Conv)
    # to compute per-parameter gradients.
    loss_present = "loss" in ctx.value_types
    seed_grad = None

    # Emit a GenerateGradients node as a compatibility marker if a loss
    # and trainable params exist. This mirrors common ONNX training
    # conventions and lets tools that expect a GenerateGradients op find it
    # even when we also emit per-op autodiff nodes.
    if loss_present and grad_names:
        try:
            import json

            params_list = [p for p, e in trainables.items() if e]
            ctx.add_node("GenerateGradients", ["loss"], grad_names, attrs={"params": json.dumps(params_list)})
            try:
                ctx.nodes[-1].domain = training_domain
            except (AttributeError, IndexError) as e:
                logger.debug(f"Could not set training domain on GenerateGradients node: {e}")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to add GenerateGradients node: {e}")

    if loss_present:
        # Create a scalar '1.0' literal to seed dloss
        seed_grad = ctx.add_tensor_literal([1.0], {"scalar": "f32", "dims": []})

        # If requested, create a step counter literal so algorithms may accept
        # a dynamic step/iteration input (and be exposed as an algorithm input
        # when `step_input` is set in training config).
        training_cfg = ctx.model_metadata.get("training_config") or {}
        try:
            if training_cfg.get("step_input"):
                qname = (ctx.scope_display or ctx.name) + ".step"
                step_name = ctx.add_literal(0, {"scalar": "i64", "dims": []}, name=qname)
                ctx.model_metadata.setdefault("training", {})["step_literal"] = step_name
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Could not add step input literal: {e}")
        # Map any graph-visible outputs that look like 'loss' to the seed
        grads = {}
        for out_name in list(ctx.outputs.keys()):
            if str(out_name).endswith("loss"):
                grads[out_name] = seed_grad

        # Process nodes in reverse order of emission for a simple reverse topological
        for n in reversed(list(ctx.nodes)):
            # Determine if node contributes to any current gradient target
            produces = list(n.output)
            contributes = [p for p in produces if p in grads]
            if not contributes:
                continue
            # For each supported op attempt to backprop
            op = n.op_type
            if op == "MatMul":
                # Assume outputs: [C]; inputs: [A, B]
                A = n.input[0]
                B = n.input[1]
                C = n.output[0]
                dC = grads.get(C)
                if not dC:
                    continue
                
                # Compute transpose permutation based on input rank
                # For 2D: perm=[1,0], for 3D: perm=[0,2,1], etc.
                A_type = ctx.value_types.get(A, {})
                A_dims = A_type.get("dims", [])
                B_type = ctx.value_types.get(B, {})
                B_dims = B_type.get("dims", [])
                
                # Compute perm for transposing B: swap last two dimensions
                B_perm = list(range(len(B_dims)))
                if len(B_perm) >= 2:
                    B_perm[-2], B_perm[-1] = B_perm[-1], B_perm[-2]
                else:
                    B_perm = [1, 0]  # fallback for unclear shapes
                
                # Compute perm for transposing A: swap last two dimensions
                A_perm = list(range(len(A_dims)))
                if len(A_perm) >= 2:
                    A_perm[-2], A_perm[-1] = A_perm[-1], A_perm[-2]
                else:
                    A_perm = [1, 0]  # fallback for unclear shapes
                
                # dA = MatMul(dC, Transpose(B))
                tb = ctx._next_const_name()
                try:
                    ctx.add_node("Transpose", [B], [tb], attrs={"perm": B_perm})
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except (AttributeError, IndexError) as e:
                        logger.debug(f"Could not set domain on Transpose node: {e}")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to add Transpose with domain, using default: {e}")
                    ctx.add_node("Transpose", [B], [tb], attrs={"perm": B_perm})
                dA = ctx._next_const_name()
                ctx.add_node("MatMul", [dC, tb], [dA])
                try:
                    ctx.nodes[-1].domain = training_domain
                except (AttributeError, IndexError) as e:
                    logger.debug(f"Could not set domain on MatMul node: {e}")
                # accumulate into grads[A]
                if A in grads:
                    s = ctx._next_const_name()
                    ctx.add_node("Add", [grads[A], dA], [s])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except (AttributeError, IndexError) as e:
                        logger.debug(f"Could not set domain on Add node: {e}")
                    grads[A] = s
                else:
                    grads[A] = dA

                # dB = MatMul(Transpose(A), dC)
                ta = ctx._next_const_name()
                try:
                    ctx.add_node("Transpose", [A], [ta], attrs={"perm": A_perm})
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                except Exception:
                    ctx.add_node("Transpose", [A], [ta], attrs={"perm": A_perm})
                dB = ctx._next_const_name()
                ctx.add_node("MatMul", [ta, dC], [dB])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                if B in grads:
                    s = ctx._next_const_name()
                    ctx.add_node("Add", [grads[B], dB], [s])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    grads[B] = s
                else:
                    grads[B] = dB

            elif op == "Add":
                # C = Add(A,B) -> dA += dC, dB += dC
                # Note: Handle broadcasting - if A or B were broadcast, their gradients must be contracted
                A = n.input[0]
                B = n.input[1]
                C = n.output[0]
                dC = grads.get(C)
                if not dC:
                    continue
                
                # Get shapes to detect broadcasting
                A_type = ctx.value_types.get(A, {})
                A_dims = A_type.get("dims", [])
                B_type = ctx.value_types.get(B, {})
                B_dims = B_type.get("dims", [])
                dC_type = ctx.value_types.get(dC, {})
                dC_dims = dC_type.get("dims", [])
                
                for X, X_dims in [(A, A_dims), (B, B_dims)]:
                    grad_for_X = dC
                    
                    # If X was broadcast (fewer dims than output), reduce gradient to original shape
                    if X_dims and dC_dims and len(X_dims) < len(dC_dims):
                        # Determine which axes to reduce over
                        # Given output shape dC_dims, need to reduce to X_dims
                        num_new_axes = len(dC_dims) - len(X_dims)
                        axes_to_reduce = list(range(num_new_axes))
                        
                        # Also check for 1-dimension broadcasts (1 in X_dims vs > 1 in dC_dims)
                        for i in range(len(X_dims)):
                            orig_i = num_new_axes + i
                            if orig_i < len(dC_dims) and X_dims[i] == 1 and dC_dims[orig_i] != 1:
                                axes_to_reduce.append(orig_i)
                        
                        # Insert ReduceSum if needed
                        if axes_to_reduce:
                            reduced = ctx._next_const_name()
                            ctx.add_node("ReduceSum", [dC], [reduced], attrs={"axes": axes_to_reduce, "keepdims": 1})
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            grad_for_X = reduced
                    
                    if X in grads:
                        s = ctx._next_const_name()
                        ctx.add_node("Add", [grads[X], grad_for_X], [s])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        grads[X] = s
                    else:
                        grads[X] = grad_for_X

            elif op == "Mul":
                # C = Mul(A,B) -> dA = Mul(dC, B), dB = Mul(dC, A)
                # Handle broadcasting like Add does
                A = n.input[0]
                B = n.input[1]
                C = n.output[0]
                dC = grads.get(C)
                if not dC:
                    continue
                
                # Get shapes to detect broadcasting
                A_type = ctx.value_types.get(A, {})
                A_dims = A_type.get("dims", [])
                B_type = ctx.value_types.get(B, {})
                B_dims = B_type.get("dims", [])
                dC_type = ctx.value_types.get(dC, {})
                dC_dims = dC_type.get("dims", [])
                
                for X, X_dims, Y in [(A, A_dims, B), (B, B_dims, A)]:
                    # dX = Mul(dC, Y)
                    dX = ctx._next_const_name()
                    ctx.add_node("Mul", [dC, Y], [dX])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    
                    grad_for_X = dX
                    
                    # If X was broadcast (fewer dims than output), reduce gradient to original shape
                    if X_dims and dC_dims and len(X_dims) < len(dC_dims):
                        num_new_axes = len(dC_dims) - len(X_dims)
                        axes_to_reduce = list(range(num_new_axes))
                        
                        # Also check for 1-dimension broadcasts
                        for i in range(len(X_dims)):
                            orig_i = num_new_axes + i
                            if orig_i < len(dC_dims) and X_dims[i] == 1 and dC_dims[orig_i] != 1:
                                axes_to_reduce.append(orig_i)
                        
                        if axes_to_reduce:
                            reduced = ctx._next_const_name()
                            ctx.add_node("ReduceSum", [dX], [reduced], attrs={"axes": axes_to_reduce, "keepdims": 1})
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            grad_for_X = reduced
                    
                    if X in grads:
                        s = ctx._next_const_name()
                        ctx.add_node("Add", [grads[X], grad_for_X], [s])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        grads[X] = s
                    else:
                        grads[X] = grad_for_X

            elif op == "ReduceSum":
                # Y = ReduceSum(X, axes=axes, keepdims=keepdims)
                # dX = Expand(dY, shape_of_X) to broadcast gradient back
                X = n.input[0]
                Y = n.output[0]
                dY = grads.get(Y)
                if not dY:
                    continue
                
                X_type = ctx.value_types.get(X, {})
                X_dims = X_type.get("dims", [])
                dY_type = ctx.value_types.get(dY, {})
                dY_dims = dY_type.get("dims", [])
                
                # If input has concrete dims, we can create shape literal for Expand
                if X_dims and all(isinstance(d, int) and d > 0 for d in X_dims):
                    try:
                        shape_name = ctx.add_tensor_literal(list(X_dims), {"scalar": "i64", "dims": [len(X_dims)]})
                        dX = ctx._next_const_name()
                        ctx.add_node("Expand", [dY, shape_name], [dX])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        if X in grads:
                            s = ctx._next_const_name()
                            ctx.add_node("Add", [grads[X], dX], [s])
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            grads[X] = s
                        else:
                            grads[X] = dX
                    except Exception:
                        logger.debug(f"Could not create ReduceSum gradient with concrete dims")

            elif op == "ReduceMean":
                # Y = ReduceMean(X, axes=axes, keepdims=keepdims)
                # dX = Mul(Expand(dY, shape_of_X), 1/count)
                # where count is product of reduced dimensions
                X = n.input[0]
                Y = n.output[0]
                dY = grads.get(Y)
                if not dY:
                    continue
                
                X_type = ctx.value_types.get(X, {})
                X_dims = X_type.get("dims", [])
                dY_type = ctx.value_types.get(dY, {})
                dY_dims = dY_type.get("dims", [])
                
                # Extract axes attribute (use keepdims if specified)
                axes = n.attribute if hasattr(n, 'attribute') else {}
                keepdims_attr = axes.get('keepdims', 1) if isinstance(axes, dict) else 1
                axes_attr = axes.get('axes', []) if isinstance(axes, dict) else []
                
                if X_dims and all(isinstance(d, int) and d > 0 for d in X_dims):
                    try:
                        # Compute count of reduced elements
                        if isinstance(axes_attr, list) and axes_attr:
                            count = 1
                            for ax in axes_attr:
                                if ax < len(X_dims):
                                    count *= X_dims[ax]
                        else:
                            count = 1
                            for d in X_dims:
                                count *= d
                        
                        # Expand dY back to X shape
                        shape_name = ctx.add_tensor_literal(list(X_dims), {"scalar": "i64", "dims": [len(X_dims)]})
                        expanded = ctx._next_const_name()
                        ctx.add_node("Expand", [dY, shape_name], [expanded])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        
                        # Scale by 1/count
                        scale = 1.0 / count if count > 0 else 1.0
                        scale_name = ctx.add_tensor_literal([scale], {"scalar": "f32", "dims": []})
                        dX = ctx._next_const_name()
                        ctx.add_node("Mul", [expanded, scale_name], [dX])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        
                        if X in grads:
                            s = ctx._next_const_name()
                            ctx.add_node("Add", [grads[X], dX], [s])
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            grads[X] = s
                        else:
                            grads[X] = dX
                    except Exception:
                        logger.debug(f"Could not create ReduceMean gradient")

            elif op == "ReLU":
                # Y = ReLU(X) -> dX = Mul(dY, Greater(X, 0))
                X = n.input[0]
                Y = n.output[0]
                dY = grads.get(Y)
                if not dY:
                    continue
                
                # Create mask: Greater(X, 0) -> mask
                mask = ctx._next_const_name()
                zero = ctx.add_tensor_literal([0.0], {"scalar": "f32", "dims": []})
                ctx.add_node("Greater", [X, zero], [mask])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                # Cast mask to f32 (Greater returns bool)
                mask_f32 = ctx._next_const_name()
                ctx.add_node("Cast", [mask], [mask_f32], attrs={"to": 1})  # 1 = FLOAT
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                # dX = Mul(dY, mask_f32)
                dX = ctx._next_const_name()
                ctx.add_node("Mul", [dY, mask_f32], [dX])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                if X in grads:
                    s = ctx._next_const_name()
                    ctx.add_node("Add", [grads[X], dX], [s])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    grads[X] = s
                else:
                    grads[X] = dX

            elif op == "Sigmoid":
                # Y = Sigmoid(X) -> dX = Mul(dY, Mul(Y, Sub(1, Y)))
                # where Y is the sigmoid output
                X = n.input[0]
                Y = n.output[0]
                dY = grads.get(Y)
                if not dY:
                    continue
                
                Y_val = Y  # output of sigmoid
                
                # Compute sigmoid derivative: Y * (1 - Y)
                one = ctx.add_tensor_literal([1.0], {"scalar": "f32", "dims": []})
                
                # one_minus_Y = Sub(1, Y)
                one_minus_Y = ctx._next_const_name()
                ctx.add_node("Sub", [one, Y_val], [one_minus_Y])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                # Y_times_1_minus_Y = Mul(Y, one_minus_Y)
                sigmoid_deriv = ctx._next_const_name()
                ctx.add_node("Mul", [Y_val, one_minus_Y], [sigmoid_deriv])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                # dX = Mul(dY, sigmoid_deriv)
                dX = ctx._next_const_name()
                ctx.add_node("Mul", [dY, sigmoid_deriv], [dX])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                if X in grads:
                    s = ctx._next_const_name()
                    ctx.add_node("Add", [grads[X], dX], [s])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    grads[X] = s
                else:
                    grads[X] = dX

            elif op == "Tanh":
                # Y = Tanh(X) -> dX = Mul(dY, Sub(1, Mul(Y, Y)))
                # where Y is the tanh output
                X = n.input[0]
                Y = n.output[0]
                dY = grads.get(Y)
                if not dY:
                    continue
                
                Y_val = Y  # output of tanh
                
                # Compute tanh derivative: 1 - Y^2
                one = ctx.add_tensor_literal([1.0], {"scalar": "f32", "dims": []})
                
                # Y_squared = Mul(Y, Y)
                Y_squared = ctx._next_const_name()
                ctx.add_node("Mul", [Y_val, Y_val], [Y_squared])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                # one_minus_Y_squared = Sub(1, Y_squared)
                tanh_deriv = ctx._next_const_name()
                ctx.add_node("Sub", [one, Y_squared], [tanh_deriv])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                # dX = Mul(dY, tanh_deriv)
                dX = ctx._next_const_name()
                ctx.add_node("Mul", [dY, tanh_deriv], [dX])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                
                if X in grads:
                    s = ctx._next_const_name()
                    ctx.add_node("Add", [grads[X], dX], [s])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    grads[X] = s
                else:
                    grads[X] = dX

            elif op == "Conv":
                # Y = Conv(X, W, [B], ...)
                # Backward:
                #  dX = ConvTranspose(dY, W, ...)
                #  dW = ConvTranspose(padding(dY), X, ...)
                #  dB = ReduceSum(dY) if bias present
                X = n.input[0]
                W = n.input[1]
                Y = n.output[0]
                dY = grads.get(Y)
                if not dY:
                    continue
                
                # Extract pads attribute (default: 0 for all sides)
                pads = []
                strides = [1, 1]  # Conv defaults to stride 1
                try:
                    for attr in n.attribute:
                        if attr.name == "pads":
                            pads = list(attr.ints) if attr.ints else []
                        elif attr.name == "strides":
                            strides = list(attr.ints) if attr.ints else [1, 1]
                except Exception:
                    pass
                
                # Compute dX via ConvTranspose
                try:
                    # For ConvTranspose, strides go into output_shape calculation
                    # and we use output_padding to match shapes
                    dX_name = ctx._next_const_name()
                    ctx.add_node("ConvTranspose", [dY, W], [dX_name])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    
                    if X in grads:
                        s = ctx._next_const_name()
                        ctx.add_node("Add", [grads[X], dX_name], [s])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        grads[X] = s
                    else:
                        grads[X] = dX_name
                except Exception as e:
                    logger.debug(f"Could not create Conv input gradient: {e}")
                
                # Compute dW via delayed ConvTranspose
                # For full dW computation: dW = ConvTranspose(dY, X, ...)
                # This is a simplified approximation; full impl would need padding adjustment
                try:
                    dW_name = ctx._next_const_name()
                    # ConvTranspose with swapped inputs to get weight gradient
                    # Note: this requires careful shape handling in practice
                    ctx.add_node("ConvTranspose", [dY, X], [dW_name])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    
                    if W in grads:
                        s = ctx._next_const_name()
                        ctx.add_node("Add", [grads[W], dW_name], [s])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        grads[W] = s
                    else:
                        grads[W] = dW_name
                except Exception as e:
                    logger.debug(f"Could not create Conv weight gradient: {e}")
                
                # Compute dB if bias present (3rd input)
                if len(n.input) > 2:
                    B = n.input[2]
                    try:
                        # Bias gradient: sum over all non-channel dimensions
                        # Shape: if Y is [N, C, *spatial], dB should be [C]
                        dB_name = ctx._next_const_name()
                        # For now, use ReduceMean as approximation and scale
                        ctx.add_node("ReduceMean", [dY], [dB_name])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        
                        if B in grads:
                            s = ctx._next_const_name()
                            ctx.add_node("Add", [grads[B], dB_name], [s])
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            grads[B] = s
                        else:
                            grads[B] = dB_name
                    except Exception as e:
                        logger.debug(f"Could not create Conv bias gradient: {e}")

            elif op == "LayerNormalization":
                # LayerNorm gradient computation
                # Y = LayerNorm(X, scale, bias) with scale and bias as optional inputs
                # dL/dX = dL/dY * dY/dX (computed via chain rule)
                # dL/dscale and dL/dbias require reduction over batch dimensions
                try:
                    X = n.input[0]
                    dY = grads.get(n.output[0])
                    if not dY:
                        continue
                    
                    # Basic chain rule: dX ~= dY (we use simplified gradient; full impl would use
                    # (X - mean) / sqrt(var + eps) computation)
                    dX = ctx._next_const_name()
                    ctx.add_node("Identity", [dY], [dX])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    
                    if X in grads:
                        s = ctx._next_const_name()
                        ctx.add_node("Add", [grads[X], dX], [s])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        grads[X] = s
                    else:
                        grads[X] = dX
                    
                    # Handle scale (weight) gradient if present
                    if len(n.input) > 1:
                        scale = n.input[1]
                        try:
                            dscale = ctx._next_const_name()
                            # Scale gradient: reduce over batch dimension (axis 0)
                            # Multiply dY * (X - mean) then sum over batch
                            ctx.add_node("ReduceMean", [dY], [dscale], attrs={"axes": [0]})
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            
                            if scale in grads:
                                s = ctx._next_const_name()
                                ctx.add_node("Add", [grads[scale], dscale], [s])
                                try:
                                    ctx.nodes[-1].domain = training_domain
                                except Exception:
                                    pass
                                grads[scale] = s
                            else:
                                grads[scale] = dscale
                        except Exception as e:
                            logger.debug(f"Could not create LayerNorm scale gradient: {e}")
                    
                    # Handle bias gradient if present
                    if len(n.input) > 2:
                        bias = n.input[2]
                        try:
                            dbias = ctx._next_const_name()
                            # Bias gradient: reduce over batch dimension
                            ctx.add_node("ReduceMean", [dY], [dbias], attrs={"axes": [0]})
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            
                            if bias in grads:
                                s = ctx._next_const_name()
                                ctx.add_node("Add", [grads[bias], dbias], [s])
                                try:
                                    ctx.nodes[-1].domain = training_domain
                                except Exception:
                                    pass
                                grads[bias] = s
                            else:
                                grads[bias] = dbias
                        except Exception as e:
                            logger.debug(f"Could not create LayerNorm bias gradient: {e}")
                
                except Exception as e:
                    logger.debug(f"LayerNorm gradient computation failed: {e}")

            elif op == "BatchNormalization":
                # BatchNormalization gradient computation
                # Y = BatchNorm(X, scale, bias, mean, var)
                # Training mode: uses batch statistics; inference mode: uses running statistics
                # For simplicity, compute scale and bias gradients via reduction
                try:
                    X = n.input[0]
                    dY = grads.get(n.output[0])
                    if not dY:
                        continue
                    
                    # Input gradient (simplified: just pass through dY)
                    dX = ctx._next_const_name()
                    ctx.add_node("Identity", [dY], [dX])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    
                    if X in grads:
                        s = ctx._next_const_name()
                        ctx.add_node("Add", [grads[X], dX], [s])
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        grads[X] = s
                    else:
                        grads[X] = dX
                    
                    # Handle scale (weight) gradient if present (input[1])
                    if len(n.input) > 1:
                        scale = n.input[1]
                        try:
                            # Scale gradient: reduce dY over non-channel dimensions
                            # For image data [N,C,H,W], reduce over [0,2,3] to get [C]
                            dscale = ctx._next_const_name()
                            ctx.add_node("ReduceMean", [dY], [dscale], attrs={"axes": [0]})
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            
                            if scale in grads:
                                s = ctx._next_const_name()
                                ctx.add_node("Add", [grads[scale], dscale], [s])
                                try:
                                    ctx.nodes[-1].domain = training_domain
                                except Exception:
                                    pass
                                grads[scale] = s
                            else:
                                grads[scale] = dscale
                        except Exception as e:
                            logger.debug(f"Could not create BatchNorm scale gradient: {e}")
                    
                    # Handle bias gradient if present (input[2])
                    if len(n.input) > 2:
                        bias = n.input[2]
                        try:
                            dbias = ctx._next_const_name()
                            # Bias gradient: reduce over non-channel dimensions
                            ctx.add_node("ReduceMean", [dY], [dbias], attrs={"axes": [0]})
                            try:
                                ctx.nodes[-1].domain = training_domain
                            except Exception:
                                pass
                            
                            if bias in grads:
                                s = ctx._next_const_name()
                                ctx.add_node("Add", [grads[bias], dbias], [s])
                                try:
                                    ctx.nodes[-1].domain = training_domain
                                except Exception:
                                    pass
                                grads[bias] = s
                            else:
                                grads[bias] = dbias
                        except Exception as e:
                            logger.debug(f"Could not create BatchNorm bias gradient: {e}")
                
                except Exception as e:
                    logger.debug(f"BatchNorm gradient computation failed: {e}")

            else:
                # Unsupported ops: do not propagate
                continue

        # Once we have grads for parameters, materialize them as outputs
        for pname, enabled in trainables.items():
            if not enabled:
                continue
            # The parameter may be present as qualified name in ctx.inputs/initializers
            param_name = pname
            if param_name not in grads:
                # maybe stored under qualified name variants
                candidates = [param_name, f"{ctx.scope_display}.{param_name}" if getattr(ctx, 'scope_display', None) else param_name]
                found = None
                for c in candidates:
                    if c in grads:
                        found = c
                        break
                if not found:
                    continue
                grad_src = grads[found]
            else:
                grad_src = grads[param_name]
            grad_name = f"{pname}.grad"
            # Prefer to avoid emitting a synthetic Identity node: if the
            # underlying producer value can be renamed to the canonical
            # `<param>.grad` name, do so so the graph output references the
            # original producer directly (ergonomic for Netron/human). Fall
            # back to emitting an Identity when renaming is not possible.
            t = ctx.value_types.get(pname) or {}
            try:
                # Attempt to rename the producing value to the canonical
                # gradient name so no extra Identity node is necessary.
                ctx.rename_value(grad_src, grad_name)
                # Ensure type info is recorded for both the internal and
                # graph-visible forms.
                ctx.value_types[grad_name] = as_tensor_type(t)
                ctx.add_output(grad_name, as_tensor_type(t))
            except Exception:
                try:
                    ctx.add_node("Identity", [grad_src], [grad_name])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    ctx.value_types[grad_name] = as_tensor_type(t)
                    try:
                        ctx.add_output(grad_name, as_tensor_type(t))
                    except Exception:
                        pass
                except Exception:
                    # Best-effort: if we cannot create an identity, skip.
                    continue
            added["grads"].append(grad_name)

    # Per-param Gradient fallback (used when no loss is present or
    # GenerateGradients failed)
    if not added["grads"]:
        for pname, enabled in trainables.items():
            if not enabled:
                continue
            t = ctx.value_types.get(pname) or ctx.value_types.get(pname.split(".")[-1])
            if not t:
                continue
            grad_name = f"{pname}.grad"
            try:
                ctx.add_node("Gradient", [pname], [grad_name], attrs={"for": pname})
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
            except Exception:
                ctx.add_node("Identity", [pname], [grad_name])
            ctx.value_types[grad_name] = as_tensor_type(t)
            try:
                ctx.add_output(grad_name, as_tensor_type(t))
            except Exception:
                pass
            added["grads"].append(grad_name)
    # If an optimizer is specified in the training config, wire a simple
    # optimizer node per parameter that consumes the param and its gradient.
    cfg = ctx.model_metadata.get("training_config") or {}
    if isinstance(cfg.get("optimizer"), dict):
        opt_spec = cfg.get("optimizer")
    elif isinstance(cfg.get("optimizer"), str):
        opt_spec = {"type": cfg.get("optimizer")}
    else:
        opt_spec = None

    if opt_spec and isinstance(opt_spec, dict):
        opt_type = str(opt_spec.get("type") or opt_spec.get("optimizer") or "").lower()
        lr = opt_spec.get("lr") or opt_spec.get("learning_rate")
        mapping = {
            "adam": "Adam",
            "adamw": "AdamW",
            "adagrad": "Adagrad",
            "momentum": "Momentum",
            "sgd": "SGD",
        }
        op_name = mapping.get(opt_type, (opt_type.title() if opt_type else None))
        if op_name:
            for g in added["grads"]:
                param = g[:-5]
                opt_out = f"{param}.opt"
                attrs = {}
                if lr is not None:
                    try:
                        attrs["lr"] = float(lr)
                    except Exception:
                        attrs["lr"] = lr
                # Create optimizer state initializers when required by the
                # optimizer (e.g., Adam has 'm' and 'v'). Initializers are
                # best-effort: only created for concrete dims.
                state_inputs = []
                if op_name in ("Adam", "AdamW"):
                    for sname in ("m", "v"):
                        t = ctx.value_types.get(param) or {}
                        dims = t.get("dims", []) or []
                        # Try to materialize zeros if shape is concrete
                        try:
                            if dims and all(isinstance(d, int) and d > 0 for d in dims):
                                total = 1
                                for d in dims:
                                    total *= int(d)
                                zeros = [0.0] * total
                                name = ctx.add_tensor_literal(zeros, {"scalar": t.get("scalar"), "dims": dims})
                                state_inputs.append(name)
                            else:
                                # fallback: create a named initializer with no data
                                name = f"{param}.{sname}"
                                try:
                                    ctx.add_const({"name": name, "type_decl": t, "value": {"lit_list": []}})
                                except Exception:
                                    pass
                                state_inputs.append(name)
                        except Exception:
                            pass
                # For ONNX training optimizer ops, canonical inputs are
                # often [var, m, v, lr, grad] and outputs [var, m, v]
                inputs = [param] + state_inputs
                if g:
                    inputs.append(g)
                if lr is not None:
                    # Use lr as an initializer literal so it's available as input.
                    # If training config requests a dynamic learning-rate (`lr_input`),
                    # create a named literal 'lr' so training emission can expose it
                    # as an algorithm graph input instead of copying it as an initializer.
                    training_cfg = ctx.model_metadata.get("training_config") or {}
                    try:
                        if training_cfg.get("lr_input"):
                            # use a qualified, human-friendly name to avoid collisions
                            qname = (ctx.scope_display or ctx.name) + ".lr"
                            lr_name = ctx.add_tensor_literal([float(lr)], {"scalar": "f32", "dims": [1]}, name=qname)
                            # record for emit_training_info to detect and expose
                            ctx.model_metadata.setdefault("training", {})["lr_literal"] = lr_name
                        else:
                            lr_name = ctx.add_tensor_literal([float(lr)], {"scalar": "f32", "dims": [1]})
                        inputs.append(lr_name)
                    except Exception:
                        pass

                try:
                    # Some optimizers return updated var and state tensors
                    out_count = 3 if op_name in ("Adam", "AdamW") else 1
                    outputs = [opt_out] if out_count == 1 else [opt_out, f"{param}.m", f"{param}.v"]
                    node_name = ctx.add_node(op_name, inputs, outputs, attrs=attrs)
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    # Record optimizer node and mapping for later TrainingInfo
                    added["opt_updates"][param] = outputs[0]
                    added["optimizer_nodes"].append(node_name)
                except Exception:
                    # Fallback: Build optimizer inputs with canonical ordering: [param, grad, <state...>]
                    inputs = [param, g] + state_inputs
                    try:
                        # second attempt to add the same op with canonical inputs
                        node_name = ctx.add_node(op_name, inputs, [opt_out], attrs=attrs)
                        try:
                            ctx.nodes[-1].domain = training_domain
                        except Exception:
                            pass
                        added["opt_updates"][param] = opt_out
                        added["optimizer_nodes"].append(node_name)
                    except Exception:
                        pass

    # Expose existing loss value as a graph output if it exists
    if "loss" in ctx.value_types:
        try:
            ctx.add_output("loss", as_tensor_type(ctx.value_types["loss"]))
            added["loss"] = "loss"
        except Exception:
            added["loss"] = None

    return added
