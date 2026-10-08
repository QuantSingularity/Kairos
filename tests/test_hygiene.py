import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"data", "results", ".git", "__pycache__", ".pytest_cache"}
PATTERN = re.compile(
    "[\u2013\u2014\u2015\u2212\U0001f300-\U0001faff\u2600-\u27bf\u2b50\u2705]"
)


def text_files():
    for path in ROOT.rglob("*"):
        parts = set(path.relative_to(ROOT).parts)
        if parts & SKIP or not path.is_file():
            continue
        if path.suffix in {".py", ".md", ".txt", ".yml", ".ipynb"}:
            yield path


def test_no_long_dashes_or_emojis_in_text_files():
    offenders = []
    for path in text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PATTERN.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_no_patch_log_comments():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if set(path.relative_to(ROOT).parts) & SKIP:
            continue
        if path.name == "test_hygiene.py":
            continue
        if re.search(r"#\s*(FIX|PATCH|HOTFIX)\b", path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert not offenders, offenders
