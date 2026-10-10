"""Source fingerprints that ignore comments, docstrings, annotations and layout."""

import ast
import hashlib
from typing import Any

IGNORED_FIELDS = {"type_comment", "type_params", "returns", "annotation", "kind"}
TYPING_MODULES = {"typing", "typing_extensions", "collections.abc", "__future__"}
SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_docstring(stmt: ast.stmt) -> bool:
    """Whether the statement is a bare string expression."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_typing_only(stmt: ast.stmt) -> bool:
    """Whether the statement exists only for type checking."""
    if isinstance(stmt, ast.ImportFrom):
        return stmt.module in TYPING_MODULES
    if isinstance(stmt, ast.Import):
        return all(alias.name in TYPING_MODULES for alias in stmt.names)
    if isinstance(stmt, ast.If):
        test = stmt.test
        name = test.id if isinstance(test, ast.Name) else getattr(test, "attr", "")
        return name == "TYPE_CHECKING"
    return False


def _canonical_body(stmts: list[ast.stmt], scope: bool) -> list[Any]:
    """Canonical form of a statement list without docstring or typing code."""
    out = []
    for i, stmt in enumerate(stmts):
        if (scope and i == 0 and _is_docstring(stmt)) or _is_typing_only(stmt):
            continue
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                continue
            stmt = ast.Assign(targets=[stmt.target], value=stmt.value)
        out.append(_canonical(stmt))
    return out


def _canonical(node: Any) -> Any:
    """Nested tuple describing the node without layout or annotations."""
    if not isinstance(node, ast.AST):
        return repr(node)
    parts: list[Any] = [type(node).__name__]
    scope = isinstance(node, SCOPES)
    for name, value in ast.iter_fields(node):
        if name in IGNORED_FIELDS or value is None or value == []:
            continue
        if name in ("body", "orelse", "finalbody") and isinstance(value, list):
            parts.append((name, _canonical_body(value, scope and name == "body")))
        elif isinstance(value, list):
            parts.append((name, [_canonical(v) for v in value]))
        else:
            parts.append((name, _canonical(value)))
    return tuple(parts)


def source_digest(source: str) -> str:
    """Return a digest of the executable content of Python source text."""
    return hashlib.sha256(repr(_canonical(ast.parse(source))).encode()).hexdigest()
