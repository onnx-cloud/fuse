from typing import Any, Dict

import importlib

try:
    # Preferred import (works for most lark-parser versions)
    from lark import Lark, Token, Transformer, v_args
except Exception:
    # Some packaging layouts present lark as a namespace package or place the
    # implementation in submodules. Try a few fallback import locations before
    # giving up so the Docker image and other environments are more tolerant.
    try:
        from lark.lark import Lark, Token, Transformer, v_args  # type: ignore
    except Exception:
        # Fallback: attempt to locate attributes dynamically from loaded subpackages
            # Try to import top-level package if possible; otherwise use find_spec
        lark_mod = None
        try:
            lark_mod = importlib.import_module('lark')
        except Exception:
            # If direct import failed, see if a package location exists on disk
            try:
                spec = importlib.util.find_spec('lark')
                if spec and spec.submodule_search_locations:
                    # Do not raise yet; we'll try to load by path below
                    lark_mod = None
                else:
                    lark_mod = None
            except Exception:
                lark_mod = None

        if lark_mod is not None:
            Lark = getattr(lark_mod, 'Lark', None)
            Token = getattr(lark_mod, 'Token', None)
            Transformer = getattr(lark_mod, 'Transformer', None)
            v_args = getattr(lark_mod, 'v_args', None)
        if not Lark:
            # Try known subpackage locations as a last resort
            for candidate in ('lark.lark', 'lark.parsers.lark', 'lark.lark_parser'):
                try:
                    sub = importlib.import_module(candidate)
                    Lark = getattr(sub, 'Lark', None)
                    Token = getattr(sub, 'Token', None)
                    Transformer = getattr(sub, 'Transformer', None)
                    v_args = getattr(sub, 'v_args', None)
                    if Lark:
                        break
                except Exception:
                    continue
            # If still not found, attempt to directly load the implementation file
            if not Lark:
                try:
                    spec = importlib.util.find_spec('lark')
                    if spec and spec.submodule_search_locations:
                        for p in spec.submodule_search_locations:
                            candidate = os.path.join(p, 'lark.py')
                            if os.path.exists(candidate):
                                from importlib.machinery import SourceFileLoader
                                mod = SourceFileLoader('lark._impl', candidate).load_module()
                                Lark = getattr(mod, 'Lark', None)
                                Token = getattr(mod, 'Token', None)
                                Transformer = getattr(mod, 'Transformer', None)
                                v_args = getattr(mod, 'v_args', None)
                                if Lark:
                                    break
                    # If spec didn't help, scan sys.path for a lark/lark.py candidate
                    if not Lark:
                        for base in sys.path:
                            candidate = os.path.join(base, 'lark', 'lark.py')
                            if os.path.exists(candidate):
                                from importlib.machinery import SourceFileLoader
                                mod = SourceFileLoader('lark._impl', candidate).load_module()
                                Lark = getattr(mod, 'Lark', None)
                                Token = getattr(mod, 'Token', None)
                                Transformer = getattr(mod, 'Transformer', None)
                                v_args = getattr(mod, 'v_args', None)
                                if Lark:
                                    break
                except Exception:
                    pass
        if not Lark:
            # Last-resort fallback: attempt to load the 'lark.py' implementation
            # directly from the package location (helps with unusual packaging
            # layouts where submodule resolution fails but files exist on disk).
            try:
                import importlib.util
                spec = importlib.util.find_spec('lark')
                if spec and spec.submodule_search_locations:
                    for p in spec.submodule_search_locations:
                        candidate = os.path.join(p, 'lark.py')
                        if os.path.exists(candidate):
                            from importlib.machinery import SourceFileLoader
                            mod = SourceFileLoader('lark._impl', candidate).load_module()
                            Lark = getattr(mod, 'Lark', None)
                            Token = getattr(mod, 'Token', None)
                            Transformer = getattr(mod, 'Transformer', None)
                            v_args = getattr(mod, 'v_args', None)
                            if Lark:
                                break
            except Exception:
                pass
        if not Lark:
            raise ImportError('Could not import Lark from lark or known submodules')


GRAMMAR = r"""
// -----
// Start
// -----
start: file
file: (meta | decl | COMMENT)*

// -----
// Metadata / module
// -----
meta: meta_fuse | meta_opset | meta_module | meta_kv | meta_id | meta_type | meta_training | meta_version | meta_persistent
meta_type: "@type" STRING
meta_version: "@version" VERSION
meta_training: "@training" "{" training_kv_list? "}"
meta_persistent: "@persistent" "{" (input_entry | output_entry | kwarg | COMMENT)* "}"
training_kv_list: training_kv ("," training_kv)*
training_kv: IDENT ":" "{" (kwarg ("," kwarg)*)? "}" | kwarg

meta_fuse: "@fuse" VERSION

VERSION: /\d+(\.\d+){0,2}/

WEIGHTS: "weights"

// Inline attribute assignment token: e.g., gamma@=LN1_gamma
ATTR_ASSIGN: /[A-Za-z_][A-Za-z0-9_]*@=[A-Za-z_][A-Za-z0-9_]*/
meta_opset: "@opset" IDENT NUMBER
meta_module: "@domain" IDENT | "@domain" IDENT
meta_kv: "@meta" IDENT "=" value_expr
meta_id: "@id" STRING

// -----
// Top-level declarations
// -----
decl: import_decl
    | export_decl
    | test_node_decl
    | golden_decl
    | decorated_node
    | decorated_model
    | node_decl
    | model_decl
    | param_decl
    | const_decl
    | type_alias_decl
    | train_decl
    | frozen_decl
    | persistent_decl

// Inline persistent form: `@persistent weights name: Type = value`
// Use an explicit optional group for the 'weights' literal so it is preserved
// in the parse tree and visible to the transformer implementation.
persistent_decl: "@persistent" (WEIGHTS)? IDENT ":" type_expr param_default?

decorated_model: annotation* model_decl

train_decl: "@train" param_decl
frozen_decl: "@frozen" (param_decl | const_decl)


annotation: "@quantize" quantize_args    -> quantize_annot
          | "@dequantize" dequantize_args? -> dequantize_annot
          | "@proof"                         -> proof_annot
          | "@golden" golden_args? -> golden_annot
          | "@loss"                          -> loss_annot
          | "@algorithm"                     -> algorithm_annot
          | "@input" "{" input_entry* "}" -> input_annot
          | "@output" "{" output_entry* "}" -> output_annot
          | "@" IDENT "{" (input_entry | output_entry | kwarg | COMMENT)* "}" -> nested_annot

// Input/output entry forms: `name { k = v, ... }` or `name: { ... }`
output_entry: IDENT ":" "{" (kwarg ("," kwarg)*)? "}"
            | IDENT "{" (kwarg ("," kwarg)*)? "}"
input_entry: IDENT ":" "{" (kwarg ("," kwarg)*)? "}"
           | IDENT "{" (kwarg ("," kwarg)*)? "}"

golden_args: "(" (kwarg ("," kwarg)*)? ")"

decorated_node: annotation* node_decl

quantize_args: "(" STRING ("," kwarg)* ")"
dequantize_args: "(" (kwarg ("," kwarg)*)? ")"

type_alias_decl: "type" IDENT ("=" type_expr | type_expr)

import_decl: "@import" IDENT ["@" NUMBER] ["as" IDENT] ["from" STRING] import_variants?
import_variants: "{" variant_decl* "}"
variant_decl: "@variant" IDENT "file" "=" STRING variant_opt_default? variant_opt_keep_external?
variant_opt_default: "default"
variant_opt_keep_external: "@keep_external"

// External data loader syntax used for large constants
imported_tensors: "@import" "(" STRING ("," kwarg)* ")"

// Extend value expressions to allow external load forms
value_expr: literal | list_lit | imported_tensors

node_decl: ("fn" | "node" | "block")? IDENT generic? "(" param_list? ")" ret_annot? node
test_node_decl: "@proof" node_decl
golden_decl: "@golden" node_decl
model_decl: ("model" | "graph") IDENT "(" param_list? ")" ret_annot? node
export_decl: "export" IDENT "(" param_list? ")" ret_annot? node

ret_annot: "->" (type_expr | "(" ret_item ("," ret_item)* ")") meta_ann*

ret_item: IDENT ":" type_expr | type_expr

param_list: param ("," param)*
param: IDENT param_type? param_default?
param_type: ":" type_expr meta_ann*
param_default: "=" (value_expr | expr)
param_decl: ("param" | "weight") IDENT ":" type_expr param_default?
const_decl: "const" IDENT ":" type_expr "=" value_expr

meta_ann: "@meta" IDENT "=" value_expr

// -----
// Blocks & statements
// -----

node: "{" (stmt | expr | COMMENT | SEMICOLON)* "}"
stmt: let_stmt | assign_stmt | assert_stmt | doc_stmt | annot_stmt | return_stmt | COMMENT
let_stmt: (IDENT | ident_tuple) "=" expr

// tuple-like LHS without surrounding brackets: `a, b = ...`
ident_tuple: IDENT ("," IDENT)+
assign_stmt: IDENT ":" type_expr "=" expr
// Support both `assert <expr>` and `assert <expr> == <expr>`
assert_stmt: ("assert" | "expect") expr ["==" expr]
doc_stmt: "@note" STRING
annot_stmt: "@" IDENT "(" value_expr ")"
// Allow `return a, b` (tuple return) — accepted by tests/examples.
return_stmt: "return" expr ("," expr)*

// -----
// Expressions
// -----
?expr: as_expr | infix
as_expr: infix "as" (type_expr | scalar)
infix: primary (operator primary)*
primary: atom subscript*
// Support inline arrow/lambda expressions used in terse examples: `(a, b) => expr`
// Allow parenthesized tuples/expressions (e.g., `(true, Add(...))`) and inline lambdas
atom: call | cast_expr | ident_list | list_lit | literal | IDENT | paren_expr | lambda_expr | map_lit | loop_expr | if_expr | scan_expr

map_lit: "{" (kwarg ("," kwarg)*)? "}"
subscript: "[" subscript_inner "]"
subscript_inner: (slice_expr | expr) ("," (slice_expr | expr))*
slice_expr: expr? ":" expr?
call: IDENT generic? "(" args? ")"
cast_expr: "<" type_expr ">" "(" expr ")"
args: arg ("," arg)*
arg: kwarg | attrarg | attrarg2 | posarg | star_arg | ATTR_ASSIGN
kwarg: IDENT ("=" expr | ":" "=" expr | ":" expr)
attrarg: "@" IDENT "=" value_expr
attrarg2: IDENT "@" "=" (value_expr | IDENT)
star_arg: "*" expr
posarg: expr

generic: "<" (kwarg | decl_item | scalar) ("," (kwarg | decl_item | scalar))* ">"

decl_item: IDENT ":" type_expr

OP: "+" | "-" | "*" | "/" | "@" | "⊕" | "??"
operator: OP

// Control flow expressions: loop, if, scan
loop_expr: "loop" "(" loop_args ")" "{" loop_body_stmts "return" loop_body_return "}"
loop_args: expr ("," expr)*
loop_body_stmts: (stmt | COMMENT)*
loop_body_return: expr ("," expr)*

if_expr: "static" "if" expr node ["else" node]
    | "if" expr node ["else" node]

scan_expr: "scan" "(" scan_args ")" "{" scan_body_stmts "return" scan_body_return "}"
scan_args: expr ("," expr)*
scan_body_stmts: (stmt | COMMENT)*
scan_body_return: expr ("," expr)*

// -----
// Types
// -----
// Supported type forms (angle-scalar and shorthand).
// Examples: `f32[2,3]` or `f32[2,3]` or `MyType[3]`.
type_expr: angle_scalar | array_scalar | scalar | ident_array | IDENT
// Allow both plain-scalar and angle-scalar shorthand to be followed by
// dimensions (e.g. `f32[3]` and `f32[2,2]`).
array_scalar: (scalar | angle_scalar) "[" dims "]"
ident_array: IDENT "[" dims "]"
angle_scalar: "<" scalar ">"
dims: dim ("," dim)*
dim: INT | IDENT | "_" | ELLIPSIS | ":" | dim_expr

dim_expr: dim_term (ADDOP dim_term)*
?dim_term: INT | IDENT

ADDOP: "+" | "-" | "//" | "*"

ELLIPSIS: "..."

SCALAR: "f32" | "f64" | "i64" | "i32" | "i16" | "i8"
    | "u64" | "u32" | "u16" | "u8" | "bool"
    | "bf16" | "f16" | "complex64" | "complex128"
scalar: SCALAR

// -----
// Literals
// -----
ident_list: "[" IDENT ("," IDENT)* "]"
list_lit: "[" [expr ("," expr)*] "]"
// Allow nested list literals as values (e.g., [[1.0, 2.0]])
literal: NUMBER | STRING | BOOLEAN | list_lit
NUMBER: /[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?/
STRING: /"([^"\\]|\\.)*"/
BOOLEAN: "true" | "false"
SEMICOLON: ";"

// -----

// -----
// Lambda / inline functions
// -----
lambda_expr: "(" lambda_args? ")" "=>" expr
lambda_args: IDENT ("," IDENT)*

// Parenthesized expressions and expression tuples (e.g., `(true, Add(...))`)
paren_expr: "(" expr ("," expr)* ")"

// -----
// Identifiers
// -----
IDENT: /[A-Za-z_][A-Za-z0-9_.]*/
INT: /\d+/

COMMENT: /#[^\n]*/

%ignore " "
%ignore "\t"
%ignore /(\r?\n)+/
%ignore /#[^\n]*/
// Support C/C++-style single-line comments `// ...`
%ignore /\/\/[^\n]*/
"""


# -----
# AST Transformer
# -----
@v_args(inline=True)
class FuseTransformer(Transformer):
    def __init__(self):
        super().__init__()
        # deterministic temporary name counter used for AST rewrites
        self._tmp_counter = 0

    def _next_tmp(self) -> str:
        n = f"__tmp_{self._tmp_counter}"
        self._tmp_counter += 1
        return n

    def start(self, *nodes):
        # Lark wraps the `file` production as a single child; unwrap to avoid
        # an extra list layer in the returned AST.
        if len(nodes) == 1 and isinstance(nodes[0], list):
            return list(nodes[0])
        return list(nodes)

    def meta(self, item):
        return item

    def meta_opset(self, domain, version):
        return {
            "type": "meta",
            "name": "opset",
            "value": [str(domain), int(version)],
        }

    def meta_module(self, name):
        return {"type": "meta", "name": "module", "value": str(name)}

    def meta_id(self, value):
        # sugar form: @id "examples:..."
        import re

        v = str(value).strip()
        # Accept absolute IRIs or CURIEs (prefix:local)
        if v.startswith("http://") or v.startswith("https://"):
            return {"type": "meta", "name": "id", "value": v}
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*:[^\s]+", v):
            return {"type": "meta", "name": "id", "value": v}
        # Provide a helpful parse-time error rather than deferring to later
        raise Exception("invalid @id value: must be absolute IRI (http(s)://...) or CURIE (prefix:local)")

    def meta_kv(self, key, value):
        return {"type": "meta", "name": "meta", "value": {str(key): value}}

    def meta_fuse(self, version):
        return {"type": "meta", "name": "fuse", "value": str(version)}

    def meta_version(self, version):
        return {"type": "meta", "name": "version", "value": str(version)}

    def meta_training(self, items=None):
        # Merge parsed kwarg entries and nested training blocks produced by
        # `training_kv_list` into a single metadata dict.
        out = {}
        if not items:
            return {"type": "meta", "name": "fuse.training", "value": out}
        # items may be a list or a single list wrapped as first element
        itlist = (
            items
            if isinstance(items, list)
            else (
                items[0] if items and isinstance(items[0], list) else [items]
            )
        )
        for it in itlist:
            if isinstance(it, dict):
                for k, v in it.items():
                    if (
                        isinstance(v, dict)
                        and k in out
                        and isinstance(out[k], dict)
                    ):
                        out[k].update(v)
                    else:
                        out[k] = v
        return {"type": "meta", "name": "fuse.training", "value": out}

    def training_kv_list(self, *items):
        # Return list of training kv dicts/kwarg dicts
        return list(items)

    def training_kv(self, *args):
        # forms:
        #  - kwarg -> ('k', v) already transformed into dict by kwarg()
        #  - IDENT ':' '{' (kwarg (, kwarg)*) '}' ->
        #    args: name, list-of-kwarg-dicts OR name followed by multiple dict args
        if len(args) == 1:
            # kwarg dict
            return args[0]
        name = str(args[0])
        kv_items = []
        # Collect kv dicts from remaining args
        for a in args[1:]:
            if isinstance(a, list):
                kv_items.extend(a)
            else:
                kv_items.append(a)
        out = {}
        for kv in kv_items:
            if isinstance(kv, dict):
                out.update(kv)
        return {name: out}

    def training_block(self, name, *kvlist):
        # Convert training nested block `name: { k1=..., k2=... }` into
        # { name: {k1: val, k2: val} }
        out = {}
        # kvlist may be a single list child
        items = (
            kvlist[0]
            if len(kvlist) == 1 and isinstance(kvlist[0], list)
            else list(kvlist)
        )
        for kv in items:
            if isinstance(kv, dict):
                out.update(kv)
        return {str(name): out}

    def import_variants(self, *variants):
        return list(variants)

    def import_decl(self, name, *rest):
        # Grammar changed to allow optional version/alias/source/variants.
        # Parse positional `rest` defensively. It may contain:
        #  - (NUMBER,)
        #  - ("as" IDENT,)
        #  - ("from" STRING,)
        #  - variants list
        from lark import Token

        version = None
        alias = None
        src = None
        variants = None

        for item in rest:
            if isinstance(item, Token):
                if item.type == "NUMBER":
                    version = float(item)
                elif item.type == "IDENT":
                    alias = str(item)
                elif item.type == "STRING":
                    src = str(item)[1:-1]
            elif isinstance(item, list):
                variants = item

        return {
            "type": "import",
            "name": str(name),
            "version": version,
            "alias": alias if alias else str(name),
            "source": src,
            "variants": variants or [],
        }

    def variant_decl(self, name, file, default=None, keep_external=None):
        return {
            "name": str(name),
            "file": str(file),
            "default": bool(default),
            "keep_external": bool(keep_external),
        }

    def variant_opt_default(self):
        return True

    def variant_opt_keep_external(self):
        return True

    def imported_tensors(self, filename, *kwargs):
        # filename may already be unquoted by STRING handler; handle Token or plain string
        if isinstance(filename, Token):
            nodeame = str(filename)[1:-1]
        else:
            nodeame = str(filename)
        out = {"file": nodeame}
        for k in kwargs:
            if isinstance(k, dict):
                out.update(k)
        return {"imported_tensors": out}

    def type_alias_decl(self, name, typ):
        return {"type": "type_alias", "name": str(name), "type_decl": typ}

    def dim(self, *v):
        if not v:
            return ":"
        val = v[0]
        from lark import Token, Tree

        if isinstance(val, Token):
            return str(val)
        if isinstance(val, Tree):
            # Delegate to dim_expr-like flattening
            return "".join(self._flatten_dim_tree(val))
        return val

    def return_stmt(self, first, *rest):
        # Normalize return forms. Fix cases where a leading IDENT token is
        # emitted (e.g., 'Cast<f32>(x)' may produce separate 'Cast' IDENT and
        # a call node). If we detect a leading IDENT followed by a call node
        # with the same name, fold them into a single returned expression.
        if (
            rest
            and isinstance(first, str)
            and isinstance(rest[0], dict)
            and rest[0].get("call") == first
        ):
            # Replace the leading IDENT with the call node
            items = [rest[0]] + list(rest[1:])
            if len(items) == 1:
                return {"return": items[0]}
            return {"return": items}
        if not rest:
            return {"return": first}
        return {"return": [first] + list(rest)}

    def dim_expr(self, first, *rest):
        from lark import Token, Tree

        def _tok_text(x):
            if isinstance(x, Token):
                return x.value
            if isinstance(x, Tree):
                return "".join(_tok_text(c) for c in x.children)
            return str(x)

        parts = [_tok_text(first)]
        for i in range(0, len(rest), 2):
            op = rest[i]
            term = rest[i + 1] if i + 1 < len(rest) else None
            parts.append(_tok_text(op))
            if term is not None:
                parts.append(_tok_text(term))
        return "".join(parts)

    def _flatten_dim_tree(self, tree):
        from lark import Token, Tree

        out = []
        for c in tree.children:
            if isinstance(c, Token):
                out.append(c.value)
            elif isinstance(c, Tree):
                out.extend(self._flatten_dim_tree(c))
            else:
                out.append(str(c))
        return out

    def ret_annot(self, typ, *meta):
        # Normalize return annotations. Support both unnamed forms
        # `-> (type, type)` and named forms `-> (name: type, name2: type)`.
        try:
            from lark import Token, Tree

            if isinstance(typ, Tree):
                # If it's a single child, it may be a lone `type_expr`.
                if len(typ.children) == 1 and not isinstance(
                    typ.children[0], Tree
                ):
                    typ = typ.children[0]
                else:
                    items = []
                    for c in typ.children:
                        if isinstance(c, Tree):
                            # ret_item: either `IDENT ":" type_expr` or just `type_expr`
                            if (
                                len(c.children) == 2
                                and isinstance(c.children[0], Token)
                                and isinstance(c.children[1], dict)
                            ):
                                name = str(c.children[0])
                                t = dict(c.children[1])
                                t["name"] = name
                                items.append(t)
                            elif len(c.children) == 1:
                                items.append(c.children[0])
                            else:
                                items.append(str(c))
                        elif isinstance(c, dict):
                            items.append(c)
                        else:
                            items.append(str(c))
                    typ = items
            # fallback normalization for singletons
            if isinstance(typ, (list, tuple)) and len(typ) == 1:
                typ = typ[0]
        except Exception:
            # lark not present or unexpected shape; continue defensively
            if isinstance(typ, (list, tuple)) and len(typ) == 1:
                typ = typ[0]

        merged: Dict[str, Any] = {}
        for m in meta:
            # Meta items may sometimes be untransformed Tree nodes; only merge
            # dict-shaped meta annotations produced by the transformer.
            if isinstance(m, dict):
                merged.update(m)
        if merged and isinstance(typ, dict):
            typ = dict(typ)
            typ["meta"] = merged
        return typ

    def _is_param_list(self, value):
        # Param lists are lists of {name, type, value} dicts.
        if not isinstance(value, list):
            return False
        if not value:
            return True
        first = value[0]
        return (
            isinstance(first, dict)
            and "name" in first
            and ("type" in first or "value" in first)
        )

    def node_decl(self, name, *rest):
        # Grammar: ["node"] IDENT decl-generic? "(" param_list? ")" ret_annot? block
        # Earley optional arity means we can see:
        # - (name, block)
        # - (name, params, block)
        # - (name, ret_type, block)
        # - (name, params, ret_type, block)
        # - (name, generics, params, ret_type, block)  (with decl-site generics)
        params = []
        ret_type = None
        body = None
        generics = None

        if len(rest) == 1:
            (body,) = rest
        elif len(rest) == 2:
            a, b = rest
            if self._is_param_list(a):
                params, body = a, b
            else:
                ret_type, body = a, b
        elif len(rest) == 3:
            params, ret_type, body = rest
        elif len(rest) == 4 and isinstance(rest[0], dict):
            # generics + (params|ret_type) + (ret_type|body)
            g, a, b, c = rest
            generics = g
            if self._is_param_list(a):
                params, ret_type, body = a, b, c
            else:
                ret_type, body = a, b
        elif len(rest) == 5 and isinstance(rest[0], dict):
            # generics + params + ret_type + body
            generics, params, ret_type, body = rest
        else:
            raise ValueError(f"node_decl: unasserted arity {1 + len(rest)}")

        out = {
            "type": "node",
            "name": str(name),
            "params": params or [],
            "ret_type": ret_type,
            "body": body,
        }
        # Attach parser position info for better diagnostics
        try:
            if isinstance(name, Token):
                out["__pos__"] = {
                    "filename": getattr(self, "_filename", None),
                    "line": getattr(name, "line", None),
                    "column": getattr(name, "column", None),
                }
        except Exception:
            pass
        if generics:
            out["generics"] = generics
        return out

    def test_node_decl(self, node_decl):
        d = dict(node_decl)
        d["type"] = "proof"
        return d

    def golden_decl(self, node_decl):
        d = dict(node_decl)
        d["type"] = "golden"
        d["golden"] = True
        return d

    def quantize_args(self, *items):
        return list(items)

    def dequantize_args(self, *items):
        return list(items)

    def quantize_annot(self, args):
        target_token = args[0]
        target = (
            str(target_token)[1:-1]
            if isinstance(target_token, Token)
            else str(target_token)
        )
        opts = {}
        for a in args[1:]:
            if isinstance(a, dict):
                opts.update(a)
        return {"quantize": {"target": target, **opts}}

    def dequantize_annot(self, args=None):
        opts = {}
        if args:
            for a in args:
                if isinstance(a, dict):
                    opts.update(a)
        return {"dequantize": opts}

    def proof_annot(self):
        return {"proof_annot": True}

    def loss_annot(self):
        return {"loss_annot": True}

    def algorithm_annot(self):
        return {"algorithm_annot": True}

    def golden_args(self, *items):
        if not items:
            return {}
        out: Dict[str, Any] = {}
        for kv in items:
            if isinstance(kv, dict):
                out.update(kv)
        return out

    def golden_annot(self, args=None):
        opts = args or {}
        return {"golden": opts}

    def input_annot(self, *entries):
        out = {}
        for e in entries:
            if isinstance(e, dict):
                out.update(e)
        return {"input": out}

    def output_annot(self, *entries):
        out = {}
        for e in entries:
            if isinstance(e, dict):
                out.update(e)
        return {"output": out}

    def nested_annot(self, name, *entries):
        # Generic annotation block such as `@persistent` or `@external` or `@training`.
        # Merge contained annotation dicts (input/output/kwargs) into a single dict
        # so decorated_model can apply them uniformly. When inner entries are
        # scalar (e.g., `x: "bus.in"`) synthesize the wrapper name as the
        # kv key (e.g., `x: {bus: "bus.in"}`) to preserve source intent.
        out = {}
        wrapper = str(name)
        for e in entries:
            if isinstance(e, dict):
                for k, v in e.items():
                    # If the inner value is a dict (e.g., input: { x: {...} })
                    # propagate inner mappings, converting scalar leaf values
                    # into wrapper-mapped dicts when necessary.
                    if isinstance(v, dict):
                        inner = {}
                        for ik, iv in v.items():
                            if isinstance(iv, dict):
                                inner[ik] = iv
                            else:
                                inner[ik] = {wrapper: iv}
                        out.setdefault(k, {}).update(inner)
                    else:
                        # Top-level scalar kv -> attach under wrapper name
                        out.setdefault(k, {}).update({wrapper: v})
        return out

    def meta_persistent(self, *entries):
        # Top-level `@persistent { ... }` metadata. Normalize inner
        # `input`/`output` blocks into a single mapping in the meta value.
        # When inner entries use scalar shorthand (e.g., `x: "bus.in"`)
        # normalize them into an explicit `{bus: <value>}` mapping so later
        # consumers can expect a consistent shape: `input: { x: { bus: "..." } }`.
        out = {}
        for e in entries:
            if isinstance(e, dict):
                for k, v in e.items():
                    if isinstance(v, dict):
                        # Convert inner scalar values like {'x': 'bus.in'} into
                        # {'x': {'bus': 'bus.in'}} for consistent downstream shape.
                        conv = {}
                        for ik, iv in v.items():
                            if isinstance(iv, dict):
                                conv[ik] = iv
                            else:
                                conv[ik] = {"bus": iv}
                        out.setdefault(k, {}).update(conv)
                    else:
                        out.setdefault(k, {}).update({"persistent": v})
        return {"type": "meta", "name": "persistent", "value": out}

    def output_entry(self, *items):
        # items: IDENT, maybe dict list
        name = str(items[0])
        val = {}
        for it in items[1:]:
            if isinstance(it, dict):
                val.update(it)
        return {name: val}

    def input_entry(self, *items):
        name = str(items[0])
        val = {}
        for it in items[1:]:
            if isinstance(it, dict):
                val.update(it)
        return {name: val}

    def decorated_node(self, *items):
        # last item is node_decl, preceding items are annotation dicts
        node = dict(items[-1])
        for ann in items[:-1]:
            if isinstance(ann, dict):
                # Disallow graph-only annotations on functions
                if "input" in ann or "output" in ann:
                    raise ValueError("@input/@output annotations are only valid on graph declarations")
                if "quantize" in ann:
                    node["quantize"] = ann["quantize"]
                if "dequantize" in ann:
                    node["dequantize"] = ann["dequantize"]
                if "proof_annot" in ann:
                    node["type"] = "proof"
                if "loss_annot" in ann:
                    node["loss"] = True
                if "algorithm_annot" in ann:
                    node["algorithm"] = True
                if "golden" in ann:
                    node["type"] = "golden"
                    node["golden"] = ann["golden"]
        return node

    def model_decl(self, name, *rest):
        params = []

    def decorated_model(self, *items):
        # Find the model declaration among items (annotations may be present)
        md = None
        for it in reversed(items):
            if isinstance(it, dict) and it.get("type") == "model":
                md = dict(it)
                break
            if isinstance(it, dict) and it.get("name") and it.get("params") is not None:
                md = dict(it)
                break
        if md is None:
            # fall back to any dict-like last item
            for it in reversed(items):
                if isinstance(it, dict):
                    md = dict(it)
                    break
        if md is None:
            return None
        for ann in items:
            if not isinstance(ann, dict):
                continue
            # Direct input/output annotations
            if "input" in ann:
                md.setdefault("input", {}).update(ann["input"])
            if "output" in ann:
                md.setdefault("output", {}).update(ann["output"])
            # Nested wrapper forms: e.g., @persistent { input {...} }
            for v in ann.values():
                if isinstance(v, dict):
                    if "input" in v:
                        md.setdefault("input", {}).update(v["input"])
                    if "output" in v:
                        md.setdefault("output", {}).update(v["output"])
        return md

    def model_decl(self, name, *rest):
        params = []
        ret_type = None
        body = None

        if len(rest) == 1:
            (body,) = rest
        elif len(rest) == 2:
            a, b = rest
            if self._is_param_list(a):
                params, body = a, b
            else:
                ret_type, body = a, b
        elif len(rest) == 3:
            params, ret_type, body = rest
        else:
            raise ValueError(f"model_decl: unasserted arity {1 + len(rest)}")

        out = {
            "type": "model",
            "name": str(name),
            "params": params or [],
            "ret_type": ret_type,
            "body": body,
        }
        try:
            if isinstance(name, Token):
                out["__pos__"] = {
                    "filename": getattr(self, "_filename", None),
                    "line": getattr(name, "line", None),
                    "column": getattr(name, "column", None),
                }
        except Exception:
            pass
        return out

    def export_decl(self, name, *rest):
        params = []
        ret_type = None

        if len(rest) == 1:
            (body,) = rest
        elif len(rest) == 2:
            a, b = rest
            if self._is_param_list(a):
                params, body = a, b
            else:
                ret_type, body = a, b
        elif len(rest) == 3:
            params, ret_type, body = rest
        else:
            raise ValueError(f"export_decl: unasserted arity {1 + len(rest)}")

        return {
            "type": "export",
            "name": str(name),
            "params": params or [],
            "ret_type": ret_type,
            "body": body,
        }

    def meta_ann(self, key, value):
        return {str(key): value}

    def param_type(self, typ, *meta):
        merged: Dict[str, Any] = {}
        for m in meta:
            merged.update(m)
        if merged and isinstance(typ, dict):
            typ = dict(typ)
            typ["meta"] = merged
        return typ

    def param_default(self, value):
        return value

    def param(self, name, typ=None, value=None):
        return {"name": str(name), "type": typ, "value": value}

    def param_decl(self, name, typ, value=None):
        # Include explicit training-related fields (defaults) for downstream
        # passes: `trainable` indicates whether this param is trainable and
        # `train_meta` may hold optional training metadata later.
        out = {
            "type": "param",
            "name": str(name),
            "type_decl": typ,
            "trainable": None,
            "train_meta": None,
        }
        if value is not None:
            out["value"] = value
        return out

    def persistent_decl(self, *args):
        # Inline form: `@persistent [weights] NAME: TYPE [= VALUE]`.
        # Normalize into a `param` declaration with a `persistent` flag so
        # downstream passes that expect params can handle it uniformly.
        # Args shapes:
        #  - (name, type)
        #  - (name, type, value)
        #  - ('weights', name, type)
        #  - ('weights', name, type, value)
        kind = None
        name = None
        typ = None
        val = None
        # Lark may pass a Token; be robust and compare string value.
        if len(args) >= 1 and str(args[0]) == "weights":
            kind = "weights"
            # shift
            name = args[1]
            typ = args[2]
            if len(args) > 3:
                val = args[3]
        else:
            name = args[0]
            typ = args[1]
            if len(args) > 2:
                val = args[2]

        p = {
            "type": "param",
            "name": str(name),
            "type_decl": typ,
            "trainable": None,
            "train_meta": None,
            "persistent": True,
        }
        if kind is not None:
            p["persistent_kind"] = kind
        if val is not None:
            p["value"] = val
        return p

    def const_decl(self, name, typ, value):
        if isinstance(typ, str):
            typ = {"type": "tensor", "scalar": typ, "dims": [1]}
        return {
            "type": "const",
            "name": str(name),
            "type_decl": typ,
            "value": value,
        }

    def train_decl(self, param_decl):
        # Sugar: `@train param ...` marks the parameter as trainable (able to
        # receive gradients). Preserve shape of param_decl for downstream and
        # maintain backwards-compatible `trainable` flag.
        if isinstance(param_decl, dict):
            d = dict(param_decl)
            d["trainable"] = True
            d["trainable"] = True
            return d
        return param_decl

    def frozen_decl(self, item):
        # `@frozen` marks a param or const as not trainable (trainable == False).
        if isinstance(item, dict):
            d = dict(item)
            d["trainable"] = False
            d["trainable"] = False
            return d
        return item

    def stmt(self, item):
        return item

    def node(self, *stmts):
        # Normalize node bodies into a clean list and fold occasional
        # leading IDENT + arg-list sequences into proper call nodes. Lark
        # can sometimes emit a bare IDENT followed by a list of args as
        # separate children; this mirrors the special-case handling in
        # :meth:`return_stmt` to preserve a consistent AST shape.
        items = [s for s in stmts if not isinstance(s, Token)]
        out: list = []
        i = 0
        while i < len(items):
            cur = items[i]
            # Fold patterns like: IDENT, [args...] -> {"call": IDENT, "args": [...]}
            if (
                isinstance(cur, str)
                and i + 1 < len(items)
                and isinstance(items[i + 1], list)
                and not isinstance(items[i + 1], dict)
            ):
                out.append({"call": cur, "args": items[i + 1]})
                i += 2
                continue
            out.append(cur)
            i += 1
        return out

    def file(self, *items):
        # Merge a preceding module-level `@id` meta into the following
        # declaration so each declaration's AST includes an explicit
        # `@id` key (set to the provided id or None). This keeps the
        # AST self-contained for downstream passes.
        out: list = []
        pending_id = None
        pending_persistent = None
        for item in items:
            # skip comment tokens entirely
            if isinstance(item, Token):
                continue
            # capture a standalone @id meta for the next declaration
            if (
                isinstance(item, dict)
                and item.get("type") == "meta"
                and item.get("name") == "id"
            ):
                pending_id = item.get("value")
                # do not emit this meta as a top-level node (it will be
                # attached to the following decl), continue to next item
                continue
            # capture a standalone @persistent { ... } meta for the following decl
            if (
                isinstance(item, dict)
                and item.get("type") == "meta"
                and item.get("name") == "persistent"
            ):
                pending_persistent = item.get("value")
                # do not emit this meta as a top-level node (it will be
                # attached to the following decl), continue to next item
                continue
            if isinstance(item, dict) and item.get("type") in (
                "node",
                "model",
                "export",
                "param",
                "const",
                "type_alias",
                "import",
                "proof",
                "golden",
            ):
                # attach the pending id (or None) to the declaration
                d = dict(item)
                d.setdefault("@id", pending_id)
                pending_id = None
                        # attach any preceding @persistent mapping to the declaration
                if pending_persistent and d.get("type") != "const":
                    # merge graph-level input/output annotations when present
                    if isinstance(pending_persistent, dict):
                        if "input" in pending_persistent:
                            d.setdefault("input", {}).update(pending_persistent.get("input", {}))
                        if "output" in pending_persistent:
                            d.setdefault("output", {}).update(pending_persistent.get("output", {}))
                        # preserve any other persistent metadata under 'persistent'
                        other = {k: v for k, v in pending_persistent.items() if k not in ("input", "output")}
                        if other:
                            d.setdefault("persistent", {}).update(other)
                    else:
                        d.setdefault("persistent", {}).update({"value": pending_persistent})
                    pending_persistent = None
                # Do not attach top-level @persistent to const declarations; consts
                # represent fixed tensors and are not the same as persistent params.
                out.append(d)
                continue
            # other metadata/decls are emitted unchanged; ensure they are
            # not left without an explicit @id when they are declarations
            out.append(item)
        # Ensure top-level function/model/export decls have an '@id' key
        for d in out:
            if isinstance(d, dict) and d.get("type") in (
                "node",
                "model",
                "export",
            ):
                d.setdefault("@id", None)
        return out

    def decl(self, item):
        return item

    def param_list(self, *params):
        return list(params)

    def _dim_legacy(self, v):
        return v

    def type_expr(self, v):
        return v

    def ident_tuple(self, first, *rest):
        return [str(first)] + [str(r) for r in rest]

    def let_stmt(self, name, expr):
        # If this is a tuple-destructuring LHS (e.g. `_, x = foo()`), preserve
        # the tuple-form in the AST rather than expanding it; lowering will
        # expand into per-target assignments. Tests expect the tuple-form to
        # be present in the parsed AST.
        if isinstance(name, list):
            return {"let": [str(n) for n in name], "expr": expr}
        return {"let": str(name), "expr": expr}

    def assign_stmt(self, name, typ, expr):
        return {"assign": str(name), "type": typ, "expr": expr}

    def assert_stmt(self, *args):
        # Handle both `assert expr` and `assert left == right` forms.
        if len(args) == 1:
            return {"assert": args[0]}
        if len(args) == 2:
            left, right = args
            return {"assert": {"left": left, "right": right}}
        raise ValueError("assert_stmt: unexpected arity")

    def doc_stmt(self, text):
        return {"note": text}

    def annot_stmt(self, name, value):
        return {"annot": str(name), "value": value}

    def call(self, name, *rest):
        # Support an optional `generic` (angle-bracket kwargs) between the
        # IDENT and the argument list: `Zeros<like=x>(...)`.
        generics = None
        args = None
        if not rest:
            args = []
        elif len(rest) == 1:
            if isinstance(rest[0], list):
                args = rest[0]
            else:
                generics = rest[0]
                args = []
        elif len(rest) == 2:
            generics, args = rest
        else:
            # defensive fallback
            args = (
                [r for r in rest if isinstance(r, list)][0]
                if any(isinstance(r, list) for r in rest)
                else []
            )

        # Preserve backwards compatibility: surface dict-shaped generic
        # kwargs as ordinary entries in the `args` list so existing
        # lowering logic continues to see them as attributes/kwargs. For
        # scalar-only generics (e.g., `Cast<f32>`), do not insert them into
        # the positional args list — they should be visible on the
        # `generics` field only to avoid being treated as literal args.
        if isinstance(generics, dict):
            args = [generics] + (args or [])
        out = {"call": str(name), "args": args or []}
        # Attach call site position when available for precise diagnostics
        try:
            if isinstance(name, Token):
                out["__pos__"] = {
                    "filename": getattr(self, "_filename", None),
                    "line": getattr(name, "line", None),
                    "column": getattr(name, "column", None),
                }
        except Exception:
            pass
        if generics is not None:
            out["generics"] = generics
        return out

    def cast_expr(self, typ, expr):
        t = None
        if isinstance(typ, str):
            t = str(typ)
        elif isinstance(typ, dict):
            t = typ.get("scalar")
        else:
            try:
                t = str(typ)
            except Exception:
                t = "f32"
        return {"call": "Cast", "args": [expr], "generics": {"to": t}}

    def lambda_args(self, *idents):
        # Return a list of identifier names for lambda arguments
        return [str(i) for i in idents]

    def lambda_expr(self, *items):
        # Forms:
        #  - (args...) => expr  -> items: (list_of_idents, expr)
        #  - () => expr         -> items: (expr,)
        if len(items) == 1:
            body = items[0]
            args = []
        else:
            args, body = items
        args = [str(a) for a in args] if args else []
        return {"lambda": {"args": args, "body": body}}

    def paren_expr(self, *items):
        # Parenthesized expression or tuple: (expr) -> expr, (a, b, ...) -> [a, b, ...]
        def _norm_bools(x):
            if isinstance(x, str) and x in ("true", "false"):
                return x == "true"
            if isinstance(x, list):
                return [_norm_bools(i) for i in x]
            if isinstance(x, dict):
                out = {}
                for k, v in x.items():
                    out[k] = _norm_bools(v)
                return out
            return x

        if len(items) == 1:
            return _norm_bools(items[0])
        return _norm_bools(list(items))

    def generic(self, *items):
        # Merge kwarg pairs into a single dict for ease of consumption by
        # downstream lowering (e.g. {'like': 'x'}).
        # Allow scalar-only generics such as `<f32>` to be returned as a
        # bare string, so call-level generics like `Cast<f32>(x)` are
        # represented as `generics: 'f32'` for downstream lowering. If a
        # dict-shaped kwarg is present, merge into a dict as before.
        out = {}
        scalar_items = []
        for it in items:
            if isinstance(it, dict):
                out.update(it)
            else:
                # Tokens or plain scalars (IDENT/STRING)
                scalar_items.append(it)
        if out:
            return out
        if len(scalar_items) == 1:
            # normalize Token to string when necessary
            val = scalar_items[0]
            return str(val) if not isinstance(val, dict) else val
        # Fallback: merge into a dict if multiple scalars present
        return {str(i): True for i in scalar_items}

    def as_expr(self, left, typ):
        # Cast expressions: `expr as TYPE` -> lower to a Cast call AST so
        # lowering can handle it uniformly via the existing Cast lowering.
        t = None
        if isinstance(typ, str):
            t = str(typ)
        elif isinstance(typ, dict):
            t = typ.get("scalar")
        else:
            try:
                t = str(typ)
            except Exception:
                t = "f32"
        # Represent as an explicit Cast call so lowerer can pick it up.
        try:
            return {"call": "Cast", "args": [left], "generics": {"to": t}}
        except Exception:
            return left

    def args(self, *args):
        return list(args)

    def attrarg(self, name, val):
        # Support explicit '@name=value' form -> {'@name': <value>}
        return {"@" + str(name): val}

    def attrarg2(self, name, val):
        # Keep support for spaced form: IDENT @= value
        return {"@" + str(name): val}

    def arg(self, item):
        return item

    def ATTR_ASSIGN(self, tok):
        # Token form: 'gamma@=LN1_gamma' -> {'@gamma': 'LN1_gamma'}
        s = str(tok)
        k, v = s.split("@=")
        return {"@" + k: v}

    def map_lit(self, *kwargs):
        out = {}
        for k in kwargs:
            if isinstance(k, dict):
                out.update(k)
        return out

    def star_arg(self, val):
        return {"*": val}

    def kwarg(self, name, val):
        # Normalize boolean-like strings to Python booleans for kwarg values
        if isinstance(val, str) and val in ("true", "false"):
            val = val == "true"
        return {str(name): val}

    def attrarg2(self, name, val):
        # Postfix attribute syntax: IDENT@=value -> produce internal form {'@name': value}
        return {"@" + str(name): val}

    def posarg(self, val):
        # Normalize inline boolean tokens appearing as positional args
        if isinstance(val, str) and val in ("true", "false"):
            return val == "true"
        return val

    def infix(self, first, *rest):
        # flatten operators: [op, operand] pairs.
        from lark import Token, Tree

        if not rest:
            return first
        it = iter(rest)
        ops = []
        for op, val in zip(it, it):
            if isinstance(op, Token):
                opname = op.value
            elif isinstance(op, Tree):
                # find first Token child if present
                tok = next(
                    (c for c in op.children if isinstance(c, Token)), None
                )
                if tok is not None:
                    opname = tok.value
                else:
                    s = str(op)
                    import re

                    m = re.search(r"(\+|\-|\*|/|@|⊕)", s)
                    if m:
                        opname = m.group(1)
                    else:
                        # fallback heuristic based on operand shapes
                        is_simple_right = isinstance(
                            val, (str, int, float)
                        ) or (
                            isinstance(val, dict)
                            and ("left" not in val and "call" not in val)
                        )
                        is_call_like_left = isinstance(first, dict) and (
                            "call" in first or "left" in first
                        )
                        if is_call_like_left and is_simple_right:
                            opname = "+"
                        else:
                            opname = s
            else:
                opname = str(op)

            # Salvage cases where the operator transformer returned an empty
            # string (Lark/Earley can sometimes produce an empty Tree). In
            # those situations re-apply the call-like / simple-right
            # heuristic so expressions such as `matmul(a,b) + c` inside a
            # call arg retain the '+' operator instead of becoming ''.
            if not opname:
                is_simple_right = isinstance(val, (str, int, float)) or (
                    isinstance(val, dict)
                    and ("left" not in val and "call" not in val)
                )
                is_call_like_left = isinstance(first, dict) and (
                    "call" in first or "left" in first
                )
                if is_call_like_left and is_simple_right:
                    opname = "+"

            ops.append({"op": opname, "right": val})
        return {"left": first, "ops": ops}

    def loop_expr(self, args, stmts, returns):
        """Transform loop (args) { stmts; return exprs }"""
        return {
            "call": "loop",
            "args": list(args) if args else [],
            "body": {
                "type": "block",
                "stmts": list(stmts) if stmts else [],
                "returns": list(returns) if returns else [],
            },
        }

    def loop_args(self, first, *rest):
        return [first] + list(rest)

    def loop_body_stmts(self, *stmts):
        return list(stmts) if stmts else []

    def loop_body_return(self, first, *rest):
        return [first] + list(rest)

    def if_expr(self, *parts):
        return {"if": parts}

    def scan_expr(self, args, stmts, returns):
        """Transform scan (args) { stmts; return exprs }"""
        return {
            "call": "scan",
            "args": list(args) if args else [],
            "body": {
                "type": "block",
                "stmts": list(stmts) if stmts else [],
                "returns": list(returns) if returns else [],
            },
        }

    def scan_args(self, first, *rest):
        return [first] + list(rest)

    def scan_body_stmts(self, *stmts):
        return list(stmts) if stmts else []

    def scan_body_return(self, first, *rest):
        return [first] + list(rest)

    def array_scalar(self, scalar, dims):
        # allow shorthand like `f32[1]` to be parsed as a tensor type
        return {"type": "tensor", "scalar": scalar, "dims": dims}

    def ident_array(self, name, dims):
        # Allow alias-like forms such as `MyType[3]` to be treated as tensor
        return {"type": "tensor", "scalar": str(name), "dims": dims}

    def angle_scalar(self, s):
        return s

    def dims(self, first, *rest):
        return [first] + list(rest)

    def scalar(self, name):
        return str(name)

    def list_lit(self, *items):
        # Use an explicit node to avoid ambiguity with node statement lists.
        return {"lit_list": list(items)}

    def ident_list(self, *items):
        strs = [str(i) for i in items]
        # If any item is a boolean literal or looks like a numeric literal,
        # treat the whole list as a literal list.
        for s in strs:
            if s in ("true", "false"):
                return {
                    "lit_list": [True if x == "true" else False for x in strs]
                }
            try:
                # numeric check
                float(s)
                return {
                    "lit_list": [
                        float(x) if (x and any(c.isdigit() for c in x)) else x
                        for x in strs
                    ]
                }
            except Exception:
                continue
        return {"ident_list": strs}

    def subscript(self, item):
        # subscript_inner already transformed; return it directly for use by `primary`
        return item

    def slice_expr(self, *parts):
        # parts are transformed expr children (start/stop may be None)
        if len(parts) == 2:
            return {"slice": (parts[0], parts[1])}
        if len(parts) == 1:
            # single part -> could be `:stop` or `start:` depending on parse
            return {"slice": (None, parts[0])}
        return {"slice": (None, None)}

    def atom(self, x):
        # unwrap atom wrapper so downstream transformers see canonical nodes
        return x

    def primary(self, atom, *subs):
        # Apply subscripts (if any) as nested index/slice AST nodes
        out = atom
        for s in subs:
            out = {"index": out, "selector": s}
        return out

    def value_expr(self, v):
        return v

    def literal(self, val):
        # Normalize boolean-like literal tokens that sometimes reach this
        # point as plain strings (e.g., 'true'/'false') so downstream code
        # receives proper Python booleans rather than raw strings.
        if isinstance(val, str) and val in ("true", "false"):
            return val == "true"
        return val

    def STRING(self, s):
        return str(s)[1:-1]

    def NUMBER(self, n):
        s = str(n)
        if "." in s or "e" in s or "E" in s:
            return float(s)
        return int(s)

    def BOOLEAN(self, b):
        # Be tolerant: Lark may pass a Token or a plain string depending on
        # version/production rules; coerce to string before checking value.
        return str(b) == "true"

    def IDENT(self, s):
        return str(s)

    def INT(self, s):
        return int(s)

    def operator(self, tok=None):
        """Normalize operator tokens/trees to a short string (e.g. '+', '*').

        Lark/Earley can sometimes call this rule with no children (empty
        Tree) or wrap the terminal in a tiny Tree; accept Token/Tree or
        None and return a plain string when possible. If the tree is
        empty try to extract an operator character from its string
        representation so downstream consumers (the infix transformer)
        receive a usable operator token instead of an empty string.
        """
        import re

        from lark import Token, Tree

        if tok is None:
            return ""
        if isinstance(tok, Token):
            return tok.value
        if isinstance(tok, Tree):
            # prefer a Token child when present
            tok_child = next(
                (c for c in tok.children if isinstance(c, Token)), None
            )
            if tok_child is not None:
                return tok_child.value
            # try to salvage an operator char from the Tree string
            s = str(tok)
            m = re.search(r"(\+|\-|\*|/|@|⊕)", s)
            if m:
                return m.group(1)
            # fallback to the Tree text so callers can decide
            return s


# -----
# Parser instance
# -----
# Newer lark versions disallow passing an embedded
# transformer when using Earley, so we wrap the parser + transformer behind a
# small helper preserving the `fuse_parser.parse(src)` API.


class ParseError(Exception):
    """Raised when parsing fails. Carries filename, line, column and a small
    source context snippet (three lines) if available."""

    def __init__(
        self, message, filename=None, line=None, column=None, context=None
    ):
        super().__init__(message)
        self.filename = filename
        self.line = line
        self.column = column
        self.context = context


class _FuseParserWrapper:
    def __init__(self, grammar):
        self._parser = Lark(grammar, parser="earley")
        self._transformer = FuseTransformer()

    def parse(self, s, filename=None):
        """Parse the Fuse source `s`. If `filename` is provided, include it
        in the raised ParseError so callers can show file/line context.

        This wrapper also performs a lightweight normalization pass that converts
        colon-terminated, indented blocks into explicit brace-delimited blocks
        (e.g., `model foo(...):\n  ...`) so the core Lark grammar can remain
        brace-oriented and examples/tests using the convenient colon style
        continue to work.
        """

        def _normalize_colon_blocks(src: str) -> str:
            lines = src.splitlines(keepends=True)
            out: list[str] = []
            stack: list[int] = []  # indentation levels where a node was opened
            for i, line in enumerate(lines):
                # Preserve blank lines
                if line.strip() == "":
                    out.append(line)
                    continue
                indent = len(line) - len(line.lstrip(" "))
                # If the previous output line ended with ':' and the current
                # line is indented more than that previous header, treat it as
                # an indented node header and open a brace-delimited block.
                if out and out[-1].rstrip().endswith(":"):
                    prev_indent = len(out[-1]) - len(out[-1].lstrip(" "))
                    if indent > prev_indent:
                        # replace trailing ':' with ' {' on the header line
                        out[-1] = out[-1].rstrip()[:-1] + " {\n"
                        stack.append(prev_indent)
                # If indentation decreased, close any open blocks whose
                # header indent is >= current indent
                while stack and indent <= stack[-1]:
                    close_indent = stack.pop()
                    out.append(" " * close_indent + "}\n")
                out.append(line)
            # Close any remaining open blocks at EOF
            while stack:
                close_indent = stack.pop()
                out.append(" " * close_indent + "}\n")
            return "".join(out)

        # Attach filename to transformer so it may embed location info
        self._transformer._filename = filename
        try:
            # Optionally auto-inject a top-level @fuse declaration when the
            # environment requests it (tests/CI use-case). This avoids
            # fragile test monkeypatches and keeps behavior opt-in.
            import os

            auto_inject = str(os.environ.get("FUSE_AUTO_INJECT", "")).lower() in (
                "1",
                "true",
                "yes",
            )
            if isinstance(s, str) and auto_inject:
                # detect any non-empty line starting with '@fuse'
                has = any(line.strip().startswith("@fuse") for line in s.splitlines() if line.strip())
                if not has:
                    try:
                        # Allow a dedicated override for the injected version so
                        # tests can opt-in without affecting authoritative package
                        # version resolution used elsewhere.
                        ver = os.environ.get("FUSE_AUTO_INJECT_VERSION")
                        if not ver:
                            from src.util.project_version import get_project_version

                            ver = get_project_version() or os.environ.get("FUSE_PROJECT_VERSION")
                        if ver:
                            s = f"@fuse {ver}\n" + s
                    except Exception:
                        # best-effort only; fall back to not injecting
                        pass

            s2 = _normalize_colon_blocks(s)
            # Normalize compact attr-assign forms like `gamma@=LN1_gamma`
            # into spaced form `gamma @= LN1_gamma` so the arg grammar can
            # consistently recognize it without depending on tokenization
            # ambiguity between the '@' operator and attribute syntax.
            import re

            # Prefer prefix form '@name=val' to avoid lexer ambiguity
            # caused by '@' being also an infix operator token. Convert
            # `name@=val` -> `@name=val` which is recognized by the
            # existing `attrarg` rule.
            s3 = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)@=", r"@\1=", s2)
            tree = self._parser.parse(s3)
            return self._transformer.transform(tree)
        except Exception as e:
            # Try to obtain a helpful snippet/context from the Lark exception
            ctx = None
            ln = None
            col = None
            try:
                # Lark's UnassertedInput variants have a helpful get_context method
                get_ctx = getattr(e, "get_context", None)
                if get_ctx is not None:
                    # Prefer the context from the normalized source so the
                    # line/column mapping remains useful to users.
                    ctx = get_ctx(s2)
                ln = getattr(e, "line", None)
                col = getattr(e, "column", None)
            except Exception:
                pass
            msg = f"Parse error: {e}"
            raise ParseError(
                msg, filename=filename, line=ln, column=col, context=ctx
            ) from e
        finally:
            # Clean up filename attachment
            try:
                del self._transformer._filename
            except Exception:
                pass


fuse_parser = _FuseParserWrapper(GRAMMAR)
