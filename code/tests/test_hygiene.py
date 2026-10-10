import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_TOP = {"data", "results", "reports", ".git"}
SKIP_ANY = {"__pycache__", ".pytest_cache", ".venv", "venv", ".git"}
PATTERN = re.compile(
    "[\u2013\u2014\u2015\u2212\U0001f300-\U0001faff\u2600-\u27bf\u2b50\u2705]"
)


def project_files(*suffixes):
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if parts[0] in SKIP_TOP or set(parts) & SKIP_ANY or not path.is_file():
            continue
        if path.suffix in suffixes:
            yield path


def test_no_long_dashes_or_emojis_in_text_files():
    offenders = []
    for path in project_files(".py", ".md", ".txt", ".yml", ".toml", ".ipynb"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PATTERN.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_no_patch_log_comments():
    offenders = []
    for path in project_files(".py"):
        if path.name == "test_hygiene.py":
            continue
        if re.search(r"#\s*(FIX|PATCH|HOTFIX)\b", path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert not offenders, offenders


def package_functions():
    package = ROOT / "code" / "kairos"
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("__"):
                yield path.name, node


def test_every_package_function_has_a_docstring():
    missing = [
        f"{name}:{node.name}"
        for name, node in package_functions()
        if not ast.get_docstring(node)
    ]
    assert not missing, missing


def test_every_package_function_is_annotated():
    missing = []
    for name, node in package_functions():
        args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        args = [a for a in args if a.arg not in ("self", "cls")]
        if node.returns is None or any(a.annotation is None for a in args):
            missing.append(f"{name}:{node.name}")
    assert not missing, missing
