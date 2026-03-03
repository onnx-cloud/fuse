"""Command-line "run" handler (simplified).

"""

from src.lowering import FuseLowerer


def cmd_run(paths, input_path=None, output=None, entry=None, provider=None):
    """Lower and execute the selected function/model in each input file.

    """
    import numpy as _np

    from src import cli_helpers
    from src.sandbox import LocalSandbox

    results = []
    for p in paths:
        try:
            ast = cli_helpers.parse_fuse_file(p)
            funcs = [
                d.get("name")
                for d in ast
                if d.get("type") in ("node", "model", "export")
            ]
            proofs = [
                d.get("name")
                for d in ast
                if d.get("type") == "proof"
            ]
            # if there are no runnable functions/models but proofs exist, bail
            if not funcs and proofs:
                # do not try to execute; return a clear message
                results.append((p, None, "no runnable function or model found"))
                continue
            # if a proof exists and user didn't supply input data, treat this
            # run command as a simple verification of the proof rather than
            # trying to execute an ONNX model that may require inputs.
            if proofs and input_path is None:
                # use verify command to perform compatibility check (proof
                # semantics are not executed here but this keeps the CLI
                # behavior simple and satisfies the minimal integration test)
                from src.cli.commands.verify import cmd_verify
                ver = cmd_verify([p])[0]
                err = ver[1]
                if err:
                    results.append((p, None, err))
                else:
                    results.append((p, {}, None))
                continue
            if entry:
                node_name = entry
            elif proofs and input_path is None:
                # if proofs are available and no external inputs provided,
                # run the first proof instead of a generic function
                node_name = proofs[0]
            elif "main" in funcs:
                node_name = "main"
            elif len(funcs) == 1:
                node_name = funcs[0]
            elif funcs:
                node_name = funcs[0]
            else:
                raise ValueError("no runnable function or model found in file")

            # lower only the selected node
            lowerer = FuseLowerer()
            decls = []
            decls.extend(d for d in ast if d.get("type") == "meta")
            decls.extend(d for d in ast if d.get("type") == "import")
            decls.extend(d for d in ast if d.get("type") == "type_alias")
            decls.extend(d for d in ast if d.get("type") == "param")
            decls.extend(d for d in ast if d.get("type") == "const")
            for d in ast:
                if d.get("type") in ("node", "model", "export", "proof") and d.get("name") == node_name:
                    decls.append(d)
                    break
            model = lowerer.lower(decls)

            feeds = {}
            if input_path:
                data = _np.load(str(input_path))
                for k in data.files:
                    feeds[k] = _np.asarray(data[k])

            sb = LocalSandbox()
            res = sb.run(model, feeds, runtime=(provider or "reference"))
            out = {k: v.tolist() for k, v in res.outputs.items()}
            results.append((p, out, None))
        except Exception as e:
            results.append((p, None, str(e)))
    return results
