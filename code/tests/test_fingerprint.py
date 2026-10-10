from kairos.backtest.fingerprint import source_digest

BASE = """
import numpy as np


def score(x, scale=2.0):
    total = np.sum(x) * scale
    return total
"""

REFORMATTED = '''
from __future__ import annotations

from typing import Any

import numpy as np


def score(
    x: Any,
    scale: float = 2.0,
) -> float:
    """Return the scaled total."""
    # a comment that must not matter
    total: float = np.sum(x) * scale
    return total
'''


def test_digest_ignores_comments_docstrings_hints_and_layout():
    assert source_digest(BASE) == source_digest(REFORMATTED)


def test_digest_changes_with_constants_and_operators():
    assert source_digest(BASE) != source_digest(BASE.replace("2.0", "2.5"))
    assert source_digest(BASE) != source_digest(BASE.replace("* scale", "+ scale"))


def test_digest_changes_with_new_logic():
    extra = BASE.replace("    return total", "    total = total + 1\n    return total")
    assert source_digest(BASE) != source_digest(extra)
