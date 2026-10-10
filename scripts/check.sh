#!/bin/sh
# Usage: scripts/check.sh [--fix]
set -e
cd "$(dirname "$0")/.."
if [ "$1" = "--fix" ]; then
    ruff check --fix .
    ruff format code scripts
else
    ruff check .
    ruff format --check code scripts
fi
mypy
python -m pytest -q
