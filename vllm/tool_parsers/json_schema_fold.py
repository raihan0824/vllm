# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fold JSON-schema combinators into plain nodes.

Ported from the SGLang line (constrained/json_schema_fold.py, commits 5d77d32d59,
248413bbd3, 8a347aef5b), which measured these shapes against xgrammar 0.2.7. Here the
result feeds the Gemma 4 tool declaration in the prompt, which only renders a node's own
``properties``, ``required`` and ``type``.

``fold_schema_combinators`` returns an equivalent (or, where noted, slightly looser)
schema:

- ``allOf`` is merged into its node: properties united (a property in both is merged the
  same way), required united, ``additionalProperties: false`` if either side says so,
  types / enums intersected, numeric and length bounds tightened. A local ``$ref``
  branch
  is inlined for the merge.
- ``anyOf`` / ``oneOf`` next to the node's own constraint keywords: those keywords are
  merged into every branch (A and (B or C) = (A and B) or (A and C)); a branch that
  cannot hold them is dropped.
- A ``$ref`` next to constraint keywords is merged like a one-branch ``allOf``.
- ``if: {k: const c}, then: T`` (own or in allOf) on a required key k with an enum /
  const: an anyOf with one branch per value of k, each with the matching ``then`` merged
  in.

Looser than the schema: ``additionalProperties: false`` inside one allOf branch also
forbids the other branches' properties; after the merge they are allowed. Other ``if`` /
``then`` shapes and ``not`` are kept where they were.

Anything unexpected (a non-local or missing ``$ref``, a conflict, a result over the size
limit) leaves that node, or the whole schema, as it was.
"""

import contextlib
import copy
import json
from typing import Any

# Words that describe a schema without constraining a value.
_ANNOTATIONS = frozenset(
    (
        "title",
        "description",
        "default",
        "examples",
        "example",
        "$comment",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$id",
        "$schema",
        "$anchor",
        "$defs",
        "definitions",
    )
)
_COMBINATORS = ("anyOf", "oneOf")
_MIN_KEYS = ("minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties")
_MAX_KEYS = ("maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties")
# Keys holding one subschema / a name -> subschema map / a list of subschemas.
_SUB_ONE = (
    "items",
    "additionalProperties",
    "not",
    "contains",
    "propertyNames",
    "if",
    "then",
    "else",
)
_SUB_MAP = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
)
_SUB_LIST = ("anyOf", "oneOf", "allOf", "prefixItems")

_MAX_DEPTH = 64
_MAX_BRANCHES = 64
# The rewritten schema may grow (inlined $refs, distributed anyOf); past this it is not
# used.
_MAX_GROWTH = 8
_MAX_CHARS = 1 << 20


_COND_KEYS = frozenset(("if", "then", "else"))
_MAX_COND_VALUES = 32


def _condition(branch: Any) -> tuple[str, Any, dict] | None:
    """{"if": {"properties": {k: {"const": c}}}, "then": {...}} -> (k, c, then);
    anything else -> None."""
    if not isinstance(branch, dict) or "if" not in branch or "else" in branch:
        return None
    if set(branch) - _COND_KEYS - _ANNOTATIONS:
        return None
    cond, then = branch["if"], branch.get("then", {})
    if not isinstance(cond, dict) or not isinstance(then, dict):
        return None
    props = cond.get("properties")
    if (
        set(cond) - {"properties", "required", "type"}
        or not isinstance(props, dict)
        or len(props) != 1
    ):
        return None
    ((key, sub),) = props.items()
    if not isinstance(sub, dict):
        return None
    if "const" in sub and set(sub) <= {"const", "type"}:
        return key, sub["const"], then
    if (
        isinstance(sub.get("enum"), list)
        and len(sub["enum"]) == 1
        and set(sub) <= {"enum", "type"}
    ):
        return key, sub["enum"][0], then
    return None


class _Conflict(Exception):
    """The two schemas cannot both hold (e.g. disjoint types): no value satisfies the
    merge."""


class _Unsupported(Exception):
    """A shape this rewrite does not handle: keep the node as the client wrote it."""


def _constraint_keys(node: dict) -> list[str]:
    return [k for k in node if k not in _ANNOTATIONS]


def _types(t) -> set | None:
    if t is None:
        return None
    return set(t) if isinstance(t, list) else {t}


def _merge_types(a, b):
    ta, tb = _types(a), _types(b)
    if ta is None:
        return b
    if tb is None:
        return a
    both = ta & tb
    if "integer" in ta and "number" in tb or "number" in ta and "integer" in tb:
        both.add("integer")
    if not both:
        raise _Conflict
    order = [x for x in (a if isinstance(a, list) else [a]) if x in both]
    order += sorted(both - set(order))
    return order[0] if len(order) == 1 else order


def _key(v) -> str:
    return json.dumps(v, sort_keys=True)


class _Folder:
    def __init__(self, root: dict):
        self.root = root
        self.changed = 0

    def resolve(self, node: dict, depth: int) -> dict:
        """A pure local $ref -> a copy of its target; anything else unchanged."""
        seen = 0
        while (
            isinstance(node, dict)
            and "$ref" in node
            and not _constraint_keys({k: v for k, v in node.items() if k != "$ref"})
        ):
            ref = node["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/"):
                raise _Unsupported
            target: Any = self.root
            for part in ref[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, dict) or part not in target:
                    raise _Unsupported
                target = target[part]
            if not isinstance(target, dict):
                raise _Unsupported
            node = target
            seen += 1
            if seen > _MAX_DEPTH:
                raise _Unsupported
        return copy.deepcopy(node)

    def merge(self, a: Any, b: Any, depth: int) -> Any:
        """A schema both a and b accept (allOf [a, b]), without allOf at its top."""
        if depth > _MAX_DEPTH:
            raise _Unsupported
        if a is True or a == {}:
            return copy.deepcopy(b)
        if b is True or b == {}:
            return copy.deepcopy(a)
        if a is False or b is False:
            return False
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise _Unsupported
        a = self.flatten(self.resolve(a, depth), depth + 1)
        b = self.flatten(self.resolve(b, depth), depth + 1)
        if any(k in a or k in b for k in ("$ref",)):
            raise _Unsupported  # a $ref with siblings that flatten could not inline
        out = dict(a)
        conflicted: list[str] = []
        for k, vb in b.items():
            if k not in out:
                out[k] = copy.deepcopy(vb)
                continue
            va = out[k]
            if k in _ANNOTATIONS:
                continue
            if k == "properties":
                props = dict(va)
                for name, sub in vb.items():
                    if name not in props:
                        props[name] = copy.deepcopy(sub)
                        continue
                    try:
                        props[name] = self.merge(props[name], sub, depth + 1)
                    except _Conflict:
                        conflicted.append(name)
                out[k] = props
            elif k == "required":
                out[k] = list(va) + [x for x in vb if x not in va]
            elif k == "additionalProperties" or k == "items":
                out[k] = self.merge(va, vb, depth + 1)
            elif k == "type":
                out[k] = _merge_types(va, vb)
            elif k == "enum":
                keep = {_key(x) for x in vb}
                vals = [x for x in va if _key(x) in keep]
                if not vals:
                    raise _Conflict
                out[k] = vals
            elif k == "const":
                if _key(va) != _key(vb):
                    raise _Conflict
            elif k in _MIN_KEYS:
                out[k] = max(va, vb)
            elif k in _MAX_KEYS:
                out[k] = min(va, vb)
            elif k in _COMBINATORS:
                branches = []
                for x in va:
                    for y in vb:
                        with contextlib.suppress(_Conflict):
                            branches.append(self.merge(x, y, depth + 1))
                if not branches:
                    raise _Conflict
                if len(branches) > _MAX_BRANCHES:
                    raise _Unsupported
                out[k] = branches
            elif _key(va) != _key(vb):
                # pattern / format / if-then / ...: no exact intersection; keep the
                # first.
                continue
        for name in conflicted:
            # No value satisfies both definitions: an optional property can only be
            # absent (strict mode then
            # forbids it); a required one makes the node unsatisfiable, so keep the
            # first definition (best effort).
            if name in (out.get("required") or []):
                out["properties"][name] = a["properties"][name]
            else:
                del out["properties"][name]
        if "const" in out and "enum" in out:
            if _key(out["const"]) not in {_key(x) for x in out["enum"]}:
                raise _Conflict
            del out["enum"]
        self.changed += 1
        return out

    def flatten(self, node: dict, depth: int) -> dict:
        """This node without allOf, a $ref with siblings, or a combinator next to
        constraints."""
        if depth > _MAX_DEPTH:
            raise _Unsupported
        node, conds = self.take_conditions(node)
        all_of = node.get("allOf")
        if isinstance(all_of, list) and all_of:
            own = {k: v for k, v in node.items() if k != "allOf"}
            if (
                len(all_of) == 1
                and not _constraint_keys(own)
                and isinstance(all_of[0], dict)
                and set(all_of[0]) == {"$ref"}
            ):
                # {"allOf": [{"$ref"}], "description": ...}: xgrammar enforces it
                return node
            merged: Any = own
            for branch in all_of:
                try:
                    merged = self.merge(merged, branch, depth + 1)
                except _Conflict:
                    # unsatisfiable together: keep the branches that agree (left as
                    # written, xgrammar would
                    # compile this node to any JSON)
                    continue
            if not isinstance(merged, dict):
                raise _Unsupported
            node = merged
        ref = node.get("$ref")
        if ref is not None and _constraint_keys(
            {k: v for k, v in node.items() if k != "$ref"}
        ):
            own = {k: v for k, v in node.items() if k != "$ref"}
            node = self.merge(own, {"$ref": ref}, depth + 1)
        if conds:
            node = self.apply_conditions(node, conds, depth)
        for comb in _COMBINATORS:
            branches = node.get(comb)
            if not isinstance(branches, list) or not branches:
                continue
            own = {k: v for k, v in node.items() if k != comb}
            constraints = {k: v for k, v in own.items() if k not in _ANNOTATIONS}
            if not constraints:
                continue
            kept = []
            for branch in branches:
                with contextlib.suppress(_Conflict):
                    kept.append(self.merge(constraints, branch, depth + 1))
            if not kept:
                # no branch can hold the node's own keywords (unsatisfiable): keep the
                # own keywords, which
                # xgrammar would otherwise drop
                node = own
                self.changed += 1
                break
            node = {k: v for k, v in own.items() if k in _ANNOTATIONS}
            node[comb] = kept
            self.changed += 1
            break
        return node

    def take_conditions(self, node: dict) -> tuple[dict, list[tuple[str, Any, dict]]]:
        """The node without its "if k == c then ..." conditions (own and in allOf), and
        the conditions."""
        conds, rest = [], []
        for branch in node.get("allOf") or []:
            c = _condition(branch)
            (conds.append(c) if c else rest.append(branch))
        own = _condition({k: node[k] for k in _COND_KEYS if k in node})
        if own:
            conds.append(own)
        if not conds:
            return node, conds
        out = {k: v for k, v in node.items() if k not in _COND_KEYS and k != "allOf"}
        if rest:
            out["allOf"] = rest
        return out, conds

    def apply_conditions(
        self, node: dict, conds: list[tuple[str, Any, dict]], depth: int
    ) -> dict:
        """ "if k == c then T" on a required enum / const key k -> anyOf with one branch
        per value of k (exact).

        xgrammar ignores if / then (a conditionally required key stays optional). Any
        other shape keeps the first
        condition as written, which xgrammar ignores as before.
        """
        keys = {k for k, _, _ in conds}
        key = next(iter(keys))
        prop = (node.get("properties") or {}).get(key)
        values = None
        if (
            len(keys) == 1
            and key in (node.get("required") or [])
            and isinstance(prop, dict)
        ):
            prop = self.resolve(prop, depth)
            if isinstance(prop.get("enum"), list):
                values = prop["enum"]
            elif "const" in prop:
                values = [prop["const"]]
        if not values or len(values) > _MAX_COND_VALUES:
            k, c, then = conds[0]
            return dict(node, **{"if": {"properties": {k: {"const": c}}}, "then": then})
        branches = []
        for v in values:
            branch: Any = {"properties": {key: {"const": v}}}
            for k, c, then in conds:
                if _key(c) == _key(v):
                    branch = self.merge(branch, then, depth + 1)
            branches.append(branch)
        self.changed += 1
        return self.merge(node, {"anyOf": branches}, depth + 1)

    def declare_required(self, node: Any) -> Any:
        """A required key the object does not declare: xgrammar's strict mode forbids
        undeclared keys, so the
        reply could never be valid. Declare it with the additionalProperties schema (any
        value when absent);
        an object with additionalProperties false really is unsatisfiable and is left
        alone."""
        if not isinstance(node, dict):
            return node
        req, props, extra = (
            node.get("required"),
            node.get("properties"),
            node.get("additionalProperties", True),
        )
        t = _types(node.get("type"))
        if (
            not isinstance(req, list)
            or extra is False
            or not (isinstance(props, dict) or (t and "object" in t))
        ):
            return node
        props = props if isinstance(props, dict) else {}
        missing = [k for k in req if isinstance(k, str) and k not in props]
        if not missing:
            return node
        props = dict(props)
        for k in missing:
            props[k] = copy.deepcopy(extra) if isinstance(extra, dict) else {}
        self.changed += 1
        return dict(node, properties=props)

    def narrow_one_of(self, node: Any, depth: int) -> Any:
        """oneOf of objects: drop from each branch the optional keys that other branches
        require, so a reply
        written for one branch cannot also satisfy another (xgrammar compiles oneOf as
        anyOf). Only narrows:
        every reply the result allows is valid for the branch it was written for."""
        one_of = node.get("oneOf") if isinstance(node, dict) else None
        if not isinstance(one_of, list) or len(one_of) < 2:
            return node
        try:
            branches = [
                self.resolve(b, depth) if isinstance(b, dict) else None for b in one_of
            ]
        except _Unsupported:
            return node
        if any(
            not isinstance(b, dict) or not isinstance(b.get("properties"), dict)
            for b in branches
        ):
            return node
        reqs = [set(b.get("required") or []) for b in branches]
        out, changed = [], False
        for i, b in enumerate(branches):
            others = set().union(*(r for j, r in enumerate(reqs) if j != i)) - reqs[i]
            drop = [k for k in b["properties"] if k in others]
            if drop:
                b = dict(
                    b,
                    properties={
                        k: v for k, v in b["properties"].items() if k not in drop
                    },
                )
                changed = True
            out.append(b)
        if not changed:
            return node
        self.changed += 1
        return dict(node, oneOf=out)

    def walk(self, node: Any, depth: int) -> Any:
        if depth > _MAX_DEPTH or not isinstance(node, dict):
            return node
        # On failure keep the node as written; its subschemas are still walked.
        with contextlib.suppress(_Conflict, _Unsupported):
            node = self.flatten(node, depth)
        node = self.declare_required(node)
        node = self.narrow_one_of(node, depth)
        out = dict(node)
        for k in _SUB_ONE:
            if isinstance(out.get(k), dict):
                out[k] = self.walk(out[k], depth + 1)
        for k in _SUB_MAP:
            if isinstance(out.get(k), dict):
                out[k] = {n: self.walk(s, depth + 1) for n, s in out[k].items()}
        for k in _SUB_LIST:
            if isinstance(out.get(k), list):
                out[k] = [self.walk(s, depth + 1) for s in out[k]]
        return out


def fold_schema_combinators(schema_str: str) -> tuple[str, int]:
    """Return (folded schema, number of rewritten nodes); the input if unchanged."""
    try:
        schema = json.loads(schema_str)
    except (json.JSONDecodeError, TypeError):
        return schema_str, 0
    if not isinstance(schema, dict) or not any(
        k in schema_str for k in ('"allOf"', '"anyOf"', '"oneOf"', '"$ref"', '"if"')
    ):
        return schema_str, 0
    folder = _Folder(schema)
    try:
        out = folder.walk(schema, 0)
    except Exception:  # a shape this code did not expect: keep the client's schema
        return schema_str, 0
    if not folder.changed:
        return schema_str, 0
    # $defs stay at the root for the $refs left in place.
    for k in ("$defs", "definitions"):
        if k in schema and k not in out:
            out[k] = schema[k]
    text = json.dumps(out)
    if len(text) > max(_MAX_GROWTH * len(schema_str), 4096) or len(text) > _MAX_CHARS:
        return schema_str, 0
    return text, folder.changed
