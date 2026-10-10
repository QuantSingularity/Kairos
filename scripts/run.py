"""Command line entry point: python scripts/run.py <stage|all> [options]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from kairos.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
